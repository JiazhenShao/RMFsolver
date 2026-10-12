"""Validate saved results and recover existing figure exports, without rerunning physics."""
import ast
import base64
import runpy
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
candidate = OUT/'26-09-03.recovered.ipynb'
nb = nbformat.read(candidate, as_version=4)
namespace = {'const': SimpleNamespace(**runpy.run_path(str(ROOT/'RMFsolver/constants.py'))),
             'display': lambda value: None}
definitions = ast.parse(nb.cells[1].source)
definitions.body = [node for node in definitions.body if not (
    isinstance(node, ast.ImportFrom) and node.module.startswith(('scipy', 'RMFsolver')))]
exec(compile(definitions, '<restored-definitions-without-solver-imports>', 'exec'), namespace)
exec(compile(nb.cells[3].source, '<restored-cache-loader>', 'exec'), namespace)
assert all(row['action'].startswith('loaded') for row in namespace['report'])


def restore_saved_figure(index, filename):
    image = ROOT/'local_analysis_results/FL26-3'/filename
    with Image.open(image) as rendered:
        assert min(rendered.size) > 500
        rendered.verify()
    nb.cells[index].outputs = [
        nbformat.v4.new_output('stream', name='stdout', text=f'Recovered previously saved figure: local_analysis_results/FL26-3/{filename}\nNo new calculation was performed during recovery.\n'),
        nbformat.v4.new_output('display_data', data={'image/png': base64.b64encode(image.read_bytes()).decode('ascii')}, metadata={}),
    ]


assert set(namespace['boundary_sets']) == {2., 3., 4.}
for payload in namespace['boundary_sets'].values():
    for records in (payload['stable_records'], payload['A_records']):
        assert len(records) >= 2
        angles = [row['theta_rad'] for row in records]
        assert np.isclose(min(angles), 0.) and np.isclose(max(angles), np.pi/2)
restore_saved_figure(4, 'numerical-boundary-comparison-ncrit2-3-4.png')
cached = np.load(ROOT/'local_analysis_results/FL26-3/isothermal-enthalpy-vs-gibbs-ncrit3.npy', allow_pickle=True).item()
assert len(cached['records']) == 52
assert max(r['T_MeV'] for r in cached['records']) == 120.
assert {(r['balance'], r['condition']) for r in cached['records']} == {
    ('gibbs', 'muK'), ('gibbs', 'ns'), ('enthalpy', 'muK'), ('enthalpy', 'ns')}
restore_saved_figure(7, 'isothermal-boundary-test-ncrit3.png')
nbformat.validate(nb)
nbformat.write(nb, candidate)
print('Validated three cached boundary sets and 52 isothermal records; recovered both saved figure exports.')
print('All source syntax checked. No solver or plot recomputation; full execution is blocked by the existing SciPy binary import error.')
