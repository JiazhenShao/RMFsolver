"""Recover original cell sources from this thread, not replacement physics code."""
import ast
import hashlib
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HISTORY = Path.home()/'.codex/sessions/2026/06/22/rollout-2026-06-22T14-05-17-019ef0b9-0552-73b2-ab5c-08380f580125.jsonl'
SNAPSHOT = Path.home()/'Library/Application Support/Code/User/History/2150e119/fcnh.ipynb'
NEEDED = {37690, 37904, 38000, 37723, 37752, 38014, 38036, 38310, 38395, 38314, 38399}
records = {}
for index, line in enumerate(HISTORY.open()):
    if index in NEEDED:
        records[index] = json.loads(line)
    if index > max(NEEDED):
        break
assert records.keys() == NEEDED


def extracted_cells(index):
    result = {}
    for block in records[index]['payload']['output']:
        text = block.get('text', '')
        try:
            text = json.loads(text).get('output', text)
        except (ValueError, AttributeError):
            pass
        assert 'tokens truncated' not in text
        for match in re.finditer(r'===== cell (\d+) \((\w+)\) =====\n(.*?)(?=\n===== cell|\Z)', text, re.S):
            number, kind, source = match.groups()
            result[int(number)] = (kind, source.rstrip()+'\n')
    return result


def script(index, name):
    payload = records[index]['payload']
    text = payload.get('input', payload.get('arguments', ''))
    for match in re.finditer(r'"(?:[^"\\]|\\.)*"', text):
        value = json.loads(match.group())
        if not value.startswith('*** Begin Patch'):
            continue
        marker = '*** Add File: /private/tmp/'+name+'\n'
        if marker in value:
            section = value.split(marker, 1)[1].split('\n*** ', 1)[0]
            assert all(line.startswith('+') for line in section.splitlines())
            return '\n'.join(line[1:] for line in section.splitlines())+'\n'
    raise ValueError((index, name))


def assignments(code, allowed_names, namespace):
    # Replay only reviewed string transformations, never historical shell/kernel calls.
    nodes = [node for node in ast.parse(code).body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in allowed_names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical-string-edits>', 'exec'), namespace)


notebook = nbformat.read(SNAPSHOT, as_version=4)
for number in (0, 2):
    notebook.cells[number].source = extracted_cells(37690)[number][1]
notebook.cells[1].source = extracted_cells(37904)[1][1]
for number in (3, 4):
    notebook.cells[number].source = extracted_cells(38000)[number][1]

comparison_update = ast.parse(script(37723, 'update_fl26_3_comparison.py'))
for node in comparison_update.body:
    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'notebook.cells[2].source':
        notebook.cells[2].source = ast.literal_eval(node.value)
metadata_update = ast.parse(script(37752, 'patch_fl26_3_cold_metadata.py'))
for node in metadata_update.body:
    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'notebook.cells[2].source':
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<historical-markdown>', 'exec'), {'notebook': notebook})

namespace = {'cell': notebook.cells[4], 'nb': notebook}
assignments(script(38014, 'update_fl26_3_contours.py'), {'source'}, namespace)
notebook.cells[4].source = namespace['source']
assignments(script(38036, 'finish_fl26_3_contours.py'), {'source'}, namespace)
notebook.cells[4].source = namespace['source']

notebook.cells.append(nbformat.v4.new_markdown_cell(
    extracted_cells(38310)[5][1].replace('crit}(0)=4n_0', 'crit}(0)=3n_0')))
notebook.cells.append(nbformat.v4.new_code_cell(extracted_cells(38395)[6][1]))
notebook.cells.append(nbformat.v4.new_code_cell(script(38314, 'fl26_3_iso_single_plot.py')))
namespace = {'nb': notebook}
assignments(script(38399, 'extend_fl26_3_iso_window.py'), {'calc', 'start', 'end', 'plot'}, namespace)
notebook.cells[6].source = namespace['calc']
notebook.cells[7].source = namespace['plot']

for index, cell in enumerate(notebook.cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'recovered-cell-{index}', 'exec')
        cell.outputs = []
        cell.execution_count = None
assert len(notebook.cells) == 8
assert 'least_squares(lambda z: z' not in '\n'.join(c.source for c in notebook.cells)
assert 'CONTOUR_PATHS' in notebook.cells[4].source
assert 'ISOTHERMAL_NCRIT_OVER_N0 = 3.0' in notebook.cells[6].source
assert 'np.linspace(0.0, 120.0, 13)' in notebook.cells[6].source
assert 'Y_LIMITS_MEV = (0.0, 120.0)' in notebook.cells[7].source
nbformat.validate(notebook)
candidate = OUT/'26-09-03.recovered.ipynb'
nbformat.write(notebook, candidate)
provenance = {
    'history': str(HISTORY), 'metadata_snapshot': str(SNAPSHOT),
    'latest_recovered_edit_UTC': records[38399]['timestamp'],
    'history_records_zero_based': sorted(NEEDED),
    'note': 'Original sources recovered from logged cell reads and original string-edit scripts; outputs cleared, no physics recomputed.',
    'cell_source_sha256': [hashlib.sha256(c.source.encode()).hexdigest() for c in notebook.cells],
}
(OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print(candidate)
print('Recovered and syntax-checked', len(notebook.cells), 'cells.')
