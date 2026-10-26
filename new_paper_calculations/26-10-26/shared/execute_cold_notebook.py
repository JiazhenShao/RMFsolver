"""Execute only appended cold cells, preserving the original numerical figures.

In-process IPython avoids opening a Jupyter kernel port on restricted machines.
"""
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.pylabtools import select_figure_formats
from IPython.utils.capture import capture_output
import matplotlib

matplotlib.use('module://matplotlib_inline.backend_inline')
from matplotlib_inline.backend_inline import flush_figures

ROOT = Path(__file__).resolve().parents[3]
TAG = 'cold-adiabatic-analytical-contours'


def main():
    path = ROOT/'26-10-26.ipynb'
    notebook = nbformat.read(path,as_version=4)
    shell = InteractiveShell.instance()
    select_figure_formats(shell, {'png'})
    shell.execution_count = 1 + max((c.get('execution_count') or 0 for c in notebook.cells
                                    if TAG not in c.metadata.get('tags',[])), default=0)
    # Load the existing verified numerical grids without rerunning their plots.
    with capture_output(stdout=True,stderr=True,display=True):
        initialized = shell.run_cell(notebook.cells[1].source,store_history=False)
    if initialized.error_before_exec or initialized.error_in_exec:
        raise initialized.error_before_exec or initialized.error_in_exec
    for number,cell in enumerate(notebook.cells):
        if cell.cell_type != 'code' or TAG not in cell.metadata.get('tags',[]):
            continue
        print('Executing appended cell',number,flush=True)
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result = shell.run_cell(cell.source,store_history=True)
            flush_figures()
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout,captured.stderr)
            raise result.error_before_exec or result.error_in_exec
        outputs=[]
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
        cell.outputs=outputs
        cell.execution_count=shell.execution_count-1
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    print('Saved executed cold figures in',path)


if __name__=='__main__':
    main()
