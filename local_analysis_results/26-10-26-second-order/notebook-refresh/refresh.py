"""Replace the displayed analytical dataset, retaining method-specific provenance."""
from pathlib import Path
import copy
import hashlib
import json
import pickle
import shutil
import numpy as np
import nbformat
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.pylabtools import select_figure_formats
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import flush_figures

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LOCAL=ROOT/'new_paper_calculations/26-10-26/local'
PATH=LOCAL/'adiabatic-T0plus0-xi-minus0p85.npy'
NEW=LOCAL/'adiabatic-T0plus0-xi-minus0p85-second-order.npy'
ARCHIVE=HERE/'adiabatic-T0plus0-xi-minus0p85-leading-density.npy'
NOTEBOOK=ROOT/'26-10-26.ipynb'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if not ARCHIVE.exists():
    shutil.copy2(PATH,ARCHIVE)
if not (HERE/'26-10-26.before.ipynb').exists():
    shutil.copy2(NOTEBOOK,HERE/'26-10-26.before.ipynb')
original=np.load(ARCHIVE,allow_pickle=True).item()
new=np.load(NEW,allow_pickle=True).item()
assert original['complete'] and new['complete']
assert original['identity']['physics']==new['identity']['physics']
assert original['calibration']==new['calibration']
assert list(new['methods'])==['analytic']
merged=copy.deepcopy(original)
merged['methods']['analytic']=new['methods']['analytic']
merged['coverage']['analytic']=new['coverage']['analytic']
merged['angular_checks']=[r for r in original['angular_checks'] if r['method']=='semi_analytic']+new['angular_checks']
merged['method_identities']={'analytic':new['identity'],'semi_analytic':original['identity']}
merged['record_directories']={'analytic':new['record_directory'],'semi_analytic':original['record_directory']}
merged['source_manifests']={'analytic':str(NEW.relative_to(ROOT)),'semi_analytic':str(ARCHIVE.relative_to(ROOT))}
merged['identity']={'format':'cold_adiabatic_contour_collection_v1','physics':new['identity']['physics']}
merged['method_eos_approximations']={'analytic':'quadratic_pressure_second_order_density',
                                    'semi_analytic':'full_neutral_conserved_path'}
merged['usage']='Plotting collection; resume calculations from the individual source manifests.'
for key in ('record_directory','runtime_this_invocation_s','sampling_budget_extension'):
    merged.pop(key,None)
for method,source in (('analytic',new),('semi_analytic',original)):
    assert pickle.dumps(merged['methods'][method])==pickle.dumps(source['methods'][method])
    assert merged['coverage'][method]==source['coverage'][method]
with PATH.with_suffix('.tmp').open('wb') as handle:
    np.save(handle,merged,allow_pickle=True)
PATH.with_suffix('.tmp').replace(PATH)

before=nbformat.read(HERE/'26-10-26.before.ipynb',as_version=4)
notebook=nbformat.read(NOTEBOOK,as_version=4)
notebook.cells[7].source=before.cells[7].source.replace(
    'quadratic EOS, quadratic ceiling, leading diffusion, and closed $I_2$',
    'quadratic EOS with consistent second-order endpoint heat matching, quadratic ceiling, leading diffusion, and closed $I_2$')
notebook.cells[7].source+='\n\nThe full-analytical data and figures below replace the earlier leading-density endpoint calculation. The corrected scan has 61 rays and resolves all seven requested speed levels through 5000 m/s. The semi-analytical data are unchanged.'
notebook.cells[8].source=before.cells[8].source.replace('for method, coverage in cold["coverage"].items():',
    'assert cold["method_eos_approximations"]["analytic"] == "quadratic_pressure_second_order_density"\n'
    'assert all(identity["physics"] == cold["identity"]["physics"] for identity in cold["method_identities"].values())\n'
    'print("Full analytical: corrected second-order endpoint; semi-analytical: original full-EOS scan.")\n'
    'for method, coverage in cold["coverage"].items():')
notebook.cells[15].source=before.cells[15].source+'\n\nThe combined plotting file now contains the corrected full-analytical branch and the unchanged semi-analytical branch. It records separate `method_identities`, `record_directories`, and `source_manifests`; it is not a resumable single-scan checkpoint. Resume the corrected analytic scan with `--method analytic --output new_paper_calculations/26-10-26/local/adiabatic-T0plus0-xi-minus0p85-second-order.npy`. The original combined manifest is archived at `local_analysis_results/26-10-26-second-order/notebook-refresh/adiabatic-T0plus0-xi-minus0p85-leading-density.npy`.'
for index in (10,14):
    notebook.cells[index].source=before.cells[index].source.replace('1000.: 102., 2000.: 90., 5000.: 97.',
                                                                 '1000.: 72., 2000.: 81., 5000.: 90.')
    if index == 14:
        notebook.cells[index].source=notebook.cells[index].source.replace(
            '"analytic": {50.: 33., 100.: 42., 200.: 51., 500.: 61., 1000.: 72., 2000.: 81., 5000.: 90.}',
            '"analytic": {50.: 25., 100.: 31., 200.: 38., 500.: 47., 1000.: 57., 2000.: 45., 5000.: 60.}')
    assert notebook.cells[index].source.startswith('# Tunable plotting parameters')

exports=ROOT/'local_analysis_results/26-10-26'
changed_stems={'adiabatic-full-analytical-cold','adiabatic-cold-analytical-vs-numerical'}
preserved={str(p.relative_to(ROOT)):sha(p) for p in exports.iterdir()
           if p.is_file() and p.suffix in ('.png','.pdf') and p.stem not in changed_stems}
shell=InteractiveShell.instance()
select_figure_formats(shell,{'png'})
shell.execution_count=max(c.get('execution_count') or 0 for c in notebook.cells)+1
with capture_output(stdout=True,stderr=True,display=True):
    initialized=shell.run_cell(notebook.cells[1].source,store_history=False)
if initialized.error_before_exec or initialized.error_in_exec:
    raise initialized.error_before_exec or initialized.error_in_exec
for index in (8,10,14):
    cell=notebook.cells[index]
    print('Executing cell',index,flush=True)
    with capture_output(stdout=True,stderr=True,display=True) as captured:
        result=shell.run_cell(cell.source,store_history=True)
        flush_figures()
    if result.error_before_exec or result.error_in_exec:
        print(captured.stdout,captured.stderr)
        raise result.error_before_exec or result.error_in_exec
    outputs=[]
    for name,value in (('stdout',captured.stdout),('stderr',captured.stderr)):
        if value:
            outputs.append(nbformat.v4.new_output('stream',name=name,text=value))
    for item in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
    cell.outputs=outputs
    cell.execution_count=shell.execution_count-1
for index in (*range(7),11,12):
    assert notebook.cells[index]==before.cells[index],index
for name,digest in preserved.items():
    assert sha(ROOT/name)==digest,name
nbformat.validate(notebook)
nbformat.write(notebook,NOTEBOOK)
report=dict(passed=True,executed_cells=[8,10,14],unchanged_cells=list(range(7))+[11,12],
    numerical_and_semi_exports_preserved=preserved,
    combined_data=str(PATH.relative_to(ROOT)),full_analytical_source=str(NEW.relative_to(ROOT)),
    coverage=merged['coverage'],method_identities=merged['method_identities'],
    refreshed_exports=[str((exports/(stem+ext)).relative_to(ROOT)) for stem in sorted(changed_stems) for ext in ('.png','.pdf')])
(HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Verified corrected analytical data, unchanged semi-analytical data and numerical cells/exports.')
