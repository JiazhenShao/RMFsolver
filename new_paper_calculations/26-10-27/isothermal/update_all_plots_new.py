"""Replace the isothermal-contour section of All_Plots_new.ipynb with the 26-10-27 results.

The old section (archived 26-09-16 / 26-10-01 snapshots) is located by content, the new
cells (tagged TAG) are executed solver-free in an IPython shell, and the notebook is
written back only if nobody saved it in the meantime. Each figure cell writes its PNG/PDF
to local_analysis_results/All_Plots_new/ and the PDF the manuscript includes to
Latex_writting/Hydrodynamical combustion/Plots/. Rerunning replaces the tagged cells.

    PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:. \
    ./bin/python3 -B new_paper_calculations/26-10-27/isothermal/update_all_plots_new.py
"""
import inspect
import os
from pathlib import Path
import re
import sys

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.pylabtools import select_figure_formats
from IPython.utils.capture import capture_output
import matplotlib

matplotlib.use("module://matplotlib_inline.backend_inline")
from matplotlib_inline.backend_inline import flush_figures  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOTEBOOK = ROOT / "All_Plots_new.ipynb"
TAG = "isothermal-contours-26-10-27"
OLD_FIRST = "# Numerical isothermal contour\n"
NEXT_SECTION = "# GW signal section"
sys.path[:0] = [str(HERE)]
import _iso_plot  # noqa: E402

LATEX_PLOTS = "Latex_writting/Hydrodynamical combustion/Plots"

MARKDOWN = r"""# Isothermal contours (2026-10-27 production)

Numerical speeds from `solve_front_isothermal_slow(initial_guess_mode="auto")` and analytical speeds from `analytic_velocity_isothermal_slow`, `RMFsolver` [v4.9.6]: massless quarks, QMC-RMF3 pure neutron matter, $\alpha_s=0.4$, revised Madsen rate, non-relativistic equations.
Above 10 MeV the points lie on a polar mesh between the two gray boundaries; below 10 MeV on 40 logarithmic temperature rows.
No point is evaluated inside a gray region, failed points stay holes, and contours are refined for display only.
The payloads live in `new_paper_calculations/26-10-27/isothermal/`; these cells call no solver.
Each figure cell writes PNG/PDF to `local_analysis_results/All_Plots_new/` and the PDF included by `Latex_writting/Main.tex` to `Latex_writting/Hydrodynamical combustion/Plots/`, so recompiling the manuscript picks up any change made here."""

LOAD = r'''from pathlib import Path
import sys
import numpy as np

ISO_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "RMFsolver/phase_velocity.py").is_file())
ISO_WORKFLOW = ISO_ROOT / "new_paper_calculations/26-10-27/isothermal"
ISO_SCIPY_RUNTIME = ISO_ROOT / "local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64"
if "scipy" not in sys.modules and ISO_SCIPY_RUNTIME.is_dir():
    sys.path.insert(0, str(ISO_SCIPY_RUNTIME))   # the venv SciPy wheel does not load on macOS 27
for iso_path in (ISO_WORKFLOW, ISO_WORKFLOW / "highT"):
    if str(iso_path) not in sys.path:
        sys.path.insert(0, str(iso_path))
import _iso_common as iso_common
import _iso_plot as iso_plot
import _iso_xi as iso_xi
import build_boundary_curves as iso_boundary_curves

iso_payloads = {f"{method}_{panel}": iso_common.load(ISO_WORKFLOW / f"{panel}/isothermal-{panel}-{method}.npy")
                for method in ("numerical", "analytic") for panel in ("highT", "lowT")}
for panel in ("highT", "lowT"):
    numerical, analytic = iso_payloads[f"numerical_{panel}"], iso_payloads[f"analytic_{panel}"]
    assert numerical["mesh"]["fingerprint"] == analytic["mesh"]["fingerprint"], panel
    assert numerical["identity"]["sources"] == analytic["identity"]["sources"], panel
    for name, payload in (("numerical", numerical), ("analytic", analytic)):
        statuses = payload["status"].astype(str)
        print(f"{panel} {name}: { {s: int(np.sum(statuses == s)) for s in np.unique(statuses)} }")
iso_boundaries = iso_boundary_curves.load(iso_payloads["numerical_highT"]["mesh"]["boundaries"])
iso_xi_cache_lowT = iso_xi.build_cache(iso_payloads["analytic_lowT"])'''


def tuning_block(prefix, style, keys, extra):
    lines = ["# Tunable plotting parameters"]
    lines += [f"{prefix}{key.upper()} = {style[key]!r}" for key in keys]
    return "\n".join(lines + extra)


def used_log_keys():
    """Style keys read by the log-figure drawing code, plus the figure-level ones."""
    source = inspect.getsource(_iso_plot.draw_log_reference) + inspect.getsource(_iso_plot._label_log_contours)
    keys = set(re.findall(r"""style(?:\[|\.get\()["'](\w+)["']""", source))
    keys |= {"figure_size", "figure_dpi", "export_dpi", "constrained_layout", "background_color",
             "font_family", "math_font_set", "use_tex"}
    return [key for key in _iso_plot.LOG_REFERENCE_STYLE if key in keys]


LINEAR_FIGURE = tuning_block("ISO_LIN_", _iso_plot.LINEAR_REFERENCE_STYLE, _iso_plot.LINEAR_REFERENCE_STYLE, [
    "ISO_LIN_HIGH_T_CLIP_MEV = 10.0   # polar mesh above, row mesh below",
    "ISO_LIN_RADIAL_UPSAMPLE = 8   # display-only PCHIP refinement along rays in -ln(1-f)",
    "ISO_LIN_LOW_T_UPSAMPLE = 8    # display-only PCHIP refinement along rows in logit(f)",
    'ISO_LIN_OUTPUT_PNG = "local_analysis_results/All_Plots_new/isothermal-numerical-contour.png"',
    'ISO_LIN_OUTPUT_PDF = "local_analysis_results/All_Plots_new/isothermal-numerical-contour.pdf"',
    f'ISO_LIN_OUTPUT_LATEX_PDF = "{LATEX_PLOTS}/isothermal-numerical-contour.pdf"   # Fig. speed-isothermal-linear',
]) + r'''

import matplotlib as mpl
import matplotlib.pyplot as plt

iso_lin_style = {name[len("ISO_LIN_"):].lower(): value for name, value in list(globals().items())
                 if name.startswith("ISO_LIN_")}
iso_lin_figure, iso_lin_axis, iso_lin_rc = iso_plot.new_axes(iso_lin_style)
with mpl.rc_context(iso_lin_rc):
    iso_plot.draw_background(iso_lin_axis, iso_boundaries, iso_lin_style)
    iso_plot.draw_contours(iso_lin_axis, iso_payloads["numerical_highT"], T_range=(ISO_LIN_HIGH_T_CLIP_MEV, np.inf),
                           style=iso_lin_style, radial_upsample=ISO_LIN_RADIAL_UPSAMPLE)
    iso_plot.draw_row_contours(iso_lin_axis, iso_payloads["numerical_lowT"], ISO_LIN_VELOCITY_CONTOUR_LEVELS_M_S,
                               ISO_LIN_VELOCITY_CONTOUR_COLORS, ISO_LIN_VELOCITY_CONTOUR_LINE_STYLE,
                               ISO_LIN_VELOCITY_CONTOUR_LINE_WIDTH, ISO_LIN_CONTOUR_ZORDER,
                               T_range=(ISO_LIN_Y_LIMITS_MEV[0], ISO_LIN_HIGH_T_CLIP_MEV), x_limits=ISO_LIN_X_LIMITS,
                               radial_upsample=ISO_LIN_LOW_T_UPSAMPLE)
    iso_plot.finish_axes(iso_lin_axis, iso_lin_style)
    for iso_output, iso_dpi in ((ISO_LIN_OUTPUT_PNG, ISO_LIN_EXPORT_DPI), (ISO_LIN_OUTPUT_PDF, None),
                                (ISO_LIN_OUTPUT_LATEX_PDF, None)):
        (ISO_ROOT / iso_output).parent.mkdir(parents=True, exist_ok=True)
        iso_lin_figure.savefig(ISO_ROOT / iso_output, dpi=iso_dpi or "figure", facecolor=ISO_LIN_BACKGROUND_COLOR)
        print(f"Saved {iso_output}")
    plt.show()'''

LOG_MARKDOWN = r"""# Isothermal contours: logarithmic temperature, analytical versus numerical

Row mesh only (0.01–10 MeV): numerical solid, analytical dashed, the same half-decade levels labelled on the numerical lines.
The analytical speed is rescaled exactly from the saved intermediates at `ISO_LOG_XI`; the manuscript value is $\xi=-0.85$, and the contour fit in `wiki/methods/isothermal-xi-fit.md` gives $\xi=-0.56$ for this panel."""

LOG_FIGURE = tuning_block("ISO_LOG_", _iso_plot.LOG_REFERENCE_STYLE, used_log_keys(), [
    "ISO_LOG_XI = -0.85   # manuscript value; the contour fit for 0.01-10 MeV gives -0.56",
    "ISO_LOG_UPSAMPLE = 8    # display-only PCHIP refinement along rows in logit(f)",
    'ISO_LOG_OUTPUT_PNG = "local_analysis_results/All_Plots_new/isothermal-analytic-numerical-logT.png"',
    'ISO_LOG_OUTPUT_PDF = "local_analysis_results/All_Plots_new/isothermal-analytic-numerical-logT.pdf"',
    f'ISO_LOG_OUTPUT_LATEX_PDF = "{LATEX_PLOTS}/isothermal-analytic-numerical-logT.pdf"   # Fig. speed-isothermal-log',
]) + r'''

import matplotlib.pyplot as plt

iso_log_style = {name[len("ISO_LOG_"):].lower(): value for name, value in list(globals().items())
                 if name.startswith("ISO_LOG_")}
iso_log_lines = {}
iso_analytic_lowT = dict(iso_payloads["analytic_lowT"], speed_m_s=iso_xi.retune(iso_xi_cache_lowT, ISO_LOG_XI))
for iso_name, iso_payload in (("numerical", iso_payloads["numerical_lowT"]), ("analytic", iso_analytic_lowT)):
    iso_x, iso_T, iso_logv = iso_plot.contour_arrays(iso_payload, ISO_LOG_UPSAMPLE, "logit")
    iso_log_lines[iso_name] = iso_plot.contour_lines(iso_x, iso_T, iso_logv, ISO_LOG_CONTOUR_LEVELS, log_T=True)

with plt.rc_context({"font.family": ISO_LOG_FONT_FAMILY, "mathtext.fontset": ISO_LOG_MATH_FONT_SET,
                     "text.usetex": ISO_LOG_USE_TEX}):
    iso_log_figure, iso_log_axis = plt.subplots(figsize=ISO_LOG_FIGURE_SIZE, dpi=ISO_LOG_FIGURE_DPI,
                                                constrained_layout=ISO_LOG_CONSTRAINED_LAYOUT,
                                                facecolor=ISO_LOG_BACKGROUND_COLOR)
    iso_log_axis.set_facecolor(ISO_LOG_BACKGROUND_COLOR)
    iso_plot.draw_log_reference(iso_log_axis, iso_boundaries, iso_log_lines["numerical"], iso_log_lines["analytic"],
                                ISO_LOG_XI, iso_log_style)
    for iso_output, iso_dpi in ((ISO_LOG_OUTPUT_PNG, ISO_LOG_EXPORT_DPI), (ISO_LOG_OUTPUT_PDF, None),
                                (ISO_LOG_OUTPUT_LATEX_PDF, None)):
        (ISO_ROOT / iso_output).parent.mkdir(parents=True, exist_ok=True)
        iso_log_figure.savefig(ISO_ROOT / iso_output, dpi=iso_dpi or "figure", facecolor=ISO_LOG_BACKGROUND_COLOR)
        print(f"Saved {iso_output}")
    plt.show()'''


def section_bounds(cells):
    """[start, stop) of the isothermal section: tagged cells, else the old archived section."""
    tagged = [i for i, c in enumerate(cells) if TAG in c.metadata.get("tags", [])]
    if tagged:
        return tagged[0], tagged[-1] + 1
    start = next(i for i, c in enumerate(cells) if c.source.startswith(OLD_FIRST))
    stop = next(i for i, c in enumerate(cells) if i > start and c.source.startswith(NEXT_SECTION))
    assert stop - start == 6, (start, stop)   # 2 markdown + 4 code cells of the archived section
    return start, stop


def execute(cells):
    shell = InteractiveShell.instance()
    select_figure_formats(shell, {"png"})
    shell.run_cell(f"import os; os.chdir({str(ROOT)!r})", store_history=False)
    for cell in cells:
        if cell.cell_type != "code":
            continue
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
            flush_figures()
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout, captured.stderr)
            raise result.error_before_exec or result.error_in_exec
        cell.outputs = ([nbformat.v4.new_output("stream", name="stdout", text=captured.stdout)]
                        if captured.stdout else [])
        cell.outputs += [nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata)
                         for item in captured.outputs]


def main():
    before = os.stat(NOTEBOOK).st_mtime_ns
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    start, stop = section_bounds(notebook.cells)
    new = [nbformat.v4.new_markdown_cell(MARKDOWN), nbformat.v4.new_code_cell(LOAD),
           nbformat.v4.new_code_cell(LINEAR_FIGURE), nbformat.v4.new_markdown_cell(LOG_MARKDOWN),
           nbformat.v4.new_code_cell(LOG_FIGURE)]
    for cell in new:
        cell.metadata["tags"] = [TAG]
    execute(new)
    if os.stat(NOTEBOOK).st_mtime_ns != before:
        raise RuntimeError(f"{NOTEBOOK} was saved by someone else meanwhile; nothing written, rerun")
    notebook.cells[start:stop] = new
    nbformat.write(notebook, NOTEBOOK)
    print(f"replaced cells {start}-{stop - 1} of {NOTEBOOK.name} with {len(new)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
