"""Build and execute 26-10-27-isothermal.ipynb (solver-free; loads saved payloads).

Reruns replace only cells tagged TAG, so later low-T cells can be appended
beside them. Requires highT payloads and highT/boundary-curves.npz.
"""
from pathlib import Path
import shutil
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
NOTEBOOK = ROOT / "26-10-27-isothermal.ipynb"
TAG = "isothermal-highT"
sys.path[:0] = [str(HERE)]
import _iso_plot  # noqa: E402

MARKDOWN = r"""# Isothermal front speed: high temperature (10–100 MeV)

Numerical `solve_front_isothermal_slow(initial_guess_mode="auto")` and analytical `analytic_velocity_isothermal_slow` on a polar mesh. The rays start at $n_B=n_0$, $T=0$, and the points lie strictly between the two static boundaries:
- **Lower gray region:** pure neutron matter is stable against $\mu_K=0$ uds quark matter at the same $P,T$, so the speed is zero.
- **Upper gray region:** neutron matter converts directly to udd quark matter ($a(0^+)=1$), so the speed diverges.

No point inside a gray region is evaluated. The analytical speed is $v=c\sqrt{2I_2/[\lambda_n^2(1-a)(1+\xi a)]}$ with $I_2$, $\lambda_n$ and $a$ saved per point, so the comparison cell retunes $\xi$ instantly without new solves.
The band below 10 MeV comes from the low-temperature phase. Workflow: `new_paper_calculations/26-10-27/isothermal/`."""

LOAD = r'''from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "RMFsolver/phase_velocity.py").is_file())
WORKFLOW = ROOT / "new_paper_calculations/26-10-27/isothermal"
SCIPY_RUNTIME = ROOT / "local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64"
if "scipy" not in sys.modules and SCIPY_RUNTIME.is_dir():
    sys.path.insert(0, str(SCIPY_RUNTIME))   # the venv SciPy wheel does not load on macOS 27
sys.path[:0] = [str(WORKFLOW), str(WORKFLOW / "highT")]
import _iso_common as iso_common
import _iso_plot as iso_plot
import _iso_xi as iso_xi
import build_boundary_curves

numerical_highT = iso_common.load(WORKFLOW / "highT/isothermal-highT-numerical.npy")
analytic_highT = iso_common.load(WORKFLOW / "highT/isothermal-highT-analytic.npy")
assert numerical_highT["mesh"]["fingerprint"] == analytic_highT["mesh"]["fingerprint"]
assert numerical_highT["identity"]["sources"] == analytic_highT["identity"]["sources"]
boundary_curves = build_boundary_curves.load(numerical_highT["mesh"]["boundaries"])
xi_cache_highT = iso_xi.build_cache(analytic_highT)
for name, payload in (("numerical", numerical_highT), ("analytic", analytic_highT)):
    statuses = payload["status"].astype(str)
    counts = {s: int(np.sum(statuses == s)) for s in np.unique(statuses)}
    print(f"{name}: {counts}")
for level, entry in iso_plot.coverage(numerical_highT, iso_common.LEVELS_HIGH).items():
    missing = {}
    for reason in entry["missing"].values():
        missing[reason] = missing.get(reason, 0) + 1
    print(f"  numerical {level:6g} m/s: bracketed on {len(entry['bracketed'])} rays; missing {missing}")'''


def style_block(extra):
    lines = ["# Tunable plotting parameters"]
    for key, value in _iso_plot.LINEAR_REFERENCE_STYLE.items():
        lines.append(f"{key.upper()} = {value!r}")
    lines += extra
    return "\n".join(lines)


STYLE_FROM_GLOBALS = ("style = dict(iso_plot.LINEAR_REFERENCE_STYLE)\n"
                      "style.update({key.lower(): value for key, value in list(globals().items())\n"
                      "              if key.isupper() and key.lower() in style})")

NUMERICAL_FIGURE = style_block([
    "HIGH_T_CLIP_MEV = 10.0",
    "RADIAL_UPSAMPLE = 8   # display-only PCHIP refinement along rays in -ln(1-f)",
    'OUTPUT_STEM = "local_analysis_results/26-10-27/isothermal-highT-numerical"',
]) + "\n\n" + STYLE_FROM_GLOBALS + r'''

figure, axis, rc = iso_plot.new_axes(style)
with mpl.rc_context(rc):
    iso_plot.draw_background(axis, boundary_curves, style)
    iso_plot.draw_contours(axis, numerical_highT, T_range=(HIGH_T_CLIP_MEV, np.inf), style=style,
                           radial_upsample=RADIAL_UPSAMPLE)
    iso_plot.finish_axes(axis, style)
    iso_plot.export(figure, ROOT / OUTPUT_STEM, style)
    plt.show()'''

COMPARISON_FIGURE = style_block([
    "XI = -0.85",
    "HIGH_T_CLIP_MEV = 10.0",
    "RADIAL_UPSAMPLE = 8   # display-only PCHIP refinement along rays in -ln(1-f)",
    "TITLE = 'Isothermal $u(0^-)$ [$\\\\mathrm{m\\\\,s^{-1}}$]'",
    "ANALYTICAL_LINE_STYLE = '--'",
    "NUMERICAL_LINE_STYLE = '-'",
    "LEGEND_LOCATION = 'upper left'",   # empty udd-favored strip above ~95 MeV
    "LEGEND_FONT_SIZE = 16",
    "LEGEND_HANDLE_COLOR = '0.15'",
    'OUTPUT_STEM = "local_analysis_results/26-10-27/isothermal-highT-comparison"',
]) + "\n\n" + STYLE_FROM_GLOBALS + r'''
analytic_retuned = dict(analytic_highT, speed_m_s=iso_xi.retune(xi_cache_highT, XI))

figure, axis, rc = iso_plot.new_axes(style)
with mpl.rc_context(rc):
    iso_plot.draw_background(axis, boundary_curves, style)
    iso_plot.draw_contours(axis, numerical_highT, T_range=(HIGH_T_CLIP_MEV, np.inf), style=style,
                           line_style=NUMERICAL_LINE_STYLE, radial_upsample=RADIAL_UPSAMPLE)
    iso_plot.draw_contours(axis, analytic_retuned, T_range=(HIGH_T_CLIP_MEV, np.inf), style=style,
                           line_style=ANALYTICAL_LINE_STYLE, label=False,
                           radial_upsample=RADIAL_UPSAMPLE)
    iso_plot.method_legend(axis, [("Numerical", NUMERICAL_LINE_STYLE),
                                  (rf"Analytical, $\xi={XI:g}$", ANALYTICAL_LINE_STYLE)],
                           font_size=LEGEND_FONT_SIZE, location=LEGEND_LOCATION, color=LEGEND_HANDLE_COLOR,
                           line_width=VELOCITY_CONTOUR_LINE_WIDTH)
    iso_plot.finish_axes(axis, style, title=TITLE)
    iso_plot.export(figure, ROOT / OUTPUT_STEM, style)
    plt.show()'''

OLD_VS_NEW_FIGURE = style_block([
    "HIGH_T_CLIP_MEV = 10.0",
    "RADIAL_UPSAMPLE = 8   # display-only PCHIP refinement along rays in -ln(1-f)",
    "TITLE = 'Previous vs current $u(0^-)$ [$\\\\mathrm{m\\\\,s^{-1}}$]'",
    "PREVIOUS_LINE_STYLE = ':'",
    "PREVIOUS_LINE_WIDTH = 2.2",
    'PREVIOUS_ARCHIVE = "local_analysis_results/All_Plots_new/isothermal_contour_data/isothermal-numerical-contour.npz"',
    "LEGEND_LOCATION = 'upper left'",   # empty udd-favored strip above ~95 MeV
    "LEGEND_FONT_SIZE = 16",
    "LEGEND_HANDLE_COLOR = '0.15'",
    'OUTPUT_STEM = "local_analysis_results/26-10-27/isothermal-highT-old-vs-new"',
]) + "\n\n" + STYLE_FROM_GLOBALS + r'''
previous = np.load(ROOT / PREVIOUS_ARCHIVE)
previous_levels = previous["levels_m_s"].tolist()

figure, axis, rc = iso_plot.new_axes(style)
with mpl.rc_context(rc):
    iso_plot.draw_background(axis, boundary_curves, style)
    for level, color in zip(VELOCITY_CONTOUR_LEVELS_M_S, VELOCITY_CONTOUR_COLORS):
        curve = previous[f"curve_vertices_{previous_levels.index(level)}"]
        curve = curve[curve[:, 1] >= HIGH_T_CLIP_MEV]
        axis.plot(curve[:, 0], curve[:, 1], PREVIOUS_LINE_STYLE, color=color,
                  linewidth=PREVIOUS_LINE_WIDTH, zorder=CONTOUR_ZORDER)
    iso_plot.draw_contours(axis, numerical_highT, T_range=(HIGH_T_CLIP_MEV, np.inf), style=style,
                           radial_upsample=RADIAL_UPSAMPLE)
    iso_plot.method_legend(axis, [("Current: nonrel., $\\alpha_s=0.4$", "-"),
                                  ("Previous: rel., $\\alpha_s=0.3$", PREVIOUS_LINE_STYLE)],
                           font_size=LEGEND_FONT_SIZE, location=LEGEND_LOCATION, color=LEGEND_HANDLE_COLOR)
    iso_plot.finish_axes(axis, style, title=TITLE)
    iso_plot.export(figure, ROOT / OUTPUT_STEM, style)
    plt.show()'''


def main():
    if NOTEBOOK.exists():
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        backup = NOTEBOOK.with_name(NOTEBOOK.stem + ".before-highT-rebuild.ipynb")
        shutil.copy2(NOTEBOOK, backup)
        notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get("tags", [])]
    else:
        notebook = nbformat.v4.new_notebook()
    new = [nbformat.v4.new_markdown_cell(MARKDOWN)] + [
        nbformat.v4.new_code_cell(source) for source in (LOAD, NUMERICAL_FIGURE, COMPARISON_FIGURE,
                                                          OLD_VS_NEW_FIGURE)]
    for cell in new:
        cell.metadata["tags"] = [TAG]
    notebook.cells = new + notebook.cells
    shell = InteractiveShell.instance()
    select_figure_formats(shell, {"png"})
    shell.run_cell(f"import os; os.chdir({str(ROOT)!r})", store_history=False)
    for cell in new:
        if cell.cell_type != "code":
            continue
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
            flush_figures()
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout, captured.stderr)
            raise result.error_before_exec or result.error_in_exec
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata))
        cell.outputs = outputs
        cell.execution_count = shell.execution_count - 1
        print(captured.stdout.strip())
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print("wrote", NOTEBOOK)


if __name__ == "__main__":
    main()
