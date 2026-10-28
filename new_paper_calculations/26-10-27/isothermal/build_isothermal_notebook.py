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
    print(f"  numerical {level:6g} m/s: bracketed on {len(entry['bracketed'])} rays; missing {missing}")

# Low-temperature row mesh (0.01-10 MeV), drawn below 10 MeV when present.
low_T_paths = {m: WORKFLOW / f"lowT/isothermal-lowT-{m}.npy" for m in ("numerical", "analytic")}
numerical_lowT, analytic_lowT = (iso_common.load(p) if p.is_file() else None for p in low_T_paths.values())
xi_cache_lowT = iso_xi.build_cache(analytic_lowT) if analytic_lowT is not None else None
for name, payload in (("numerical low T", numerical_lowT), ("analytic low T", analytic_lowT)):
    if payload is not None:
        statuses = payload["status"].astype(str)
        print(f"{name}: { {s: int(np.sum(statuses == s)) for s in np.unique(statuses)} }")'''


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
    "HIGH_T_CLIP_MEV = 10.0   # polar mesh above, row mesh below",
    "RADIAL_UPSAMPLE = 8   # display-only PCHIP refinement along rays in -ln(1-f)",
    "LOW_T_UPSAMPLE = 8    # display-only PCHIP refinement along rows in logit(f)",
    'OUTPUT_STEM = "local_analysis_results/26-10-27/isothermal-numerical-linear"',
]) + "\n\n" + STYLE_FROM_GLOBALS + r'''

figure, axis, rc = iso_plot.new_axes(style)
with mpl.rc_context(rc):
    iso_plot.draw_background(axis, boundary_curves, style)
    iso_plot.draw_contours(axis, numerical_highT, T_range=(HIGH_T_CLIP_MEV, np.inf), style=style,
                           radial_upsample=RADIAL_UPSAMPLE)
    if numerical_lowT is not None:
        iso_plot.draw_row_contours(axis, numerical_lowT, VELOCITY_CONTOUR_LEVELS_M_S, VELOCITY_CONTOUR_COLORS,
                                   VELOCITY_CONTOUR_LINE_STYLE, VELOCITY_CONTOUR_LINE_WIDTH, CONTOUR_ZORDER,
                                   T_range=(Y_LIMITS_MEV[0], HIGH_T_CLIP_MEV), x_limits=X_LIMITS,
                                   radial_upsample=LOW_T_UPSAMPLE)
    iso_plot.finish_axes(axis, style)
    iso_plot.export(figure, ROOT / OUTPUT_STEM, style)
    plt.show()'''

COMPARISON_FIGURE = style_block([
    "XI = -0.85",
    "HIGH_T_CLIP_MEV = 10.0",
    "RADIAL_UPSAMPLE = 8   # display-only PCHIP refinement along rays in -ln(1-f)",
    "TITLE = 'Isothermal speed [$\\\\mathrm{m\\\\,s^{-1}}$]'",
    "ANALYTICAL_LINE_STYLE = '--'",
    "NUMERICAL_LINE_STYLE = '-'",
    "LEGEND_LOCATION = 'upper left'",   # empty udd-favored strip above ~95 MeV
    "LEGEND_FONT_SIZE = 16",
    "LEGEND_HANDLE_COLOR = '0.15'",
    "LOW_T_UPSAMPLE = 8    # display-only PCHIP refinement along rows in logit(f)",
    'OUTPUT_STEM = "local_analysis_results/26-10-27/isothermal-comparison-linear"',
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
    if numerical_lowT is not None:
        analytic_lowT_retuned = dict(analytic_lowT, speed_m_s=iso_xi.retune(xi_cache_lowT, XI))
        for payload, line_style in ((numerical_lowT, NUMERICAL_LINE_STYLE), (analytic_lowT_retuned, ANALYTICAL_LINE_STYLE)):
            iso_plot.draw_row_contours(axis, payload, VELOCITY_CONTOUR_LEVELS_M_S, VELOCITY_CONTOUR_COLORS,
                                       line_style, VELOCITY_CONTOUR_LINE_WIDTH, CONTOUR_ZORDER,
                                       T_range=(Y_LIMITS_MEV[0], HIGH_T_CLIP_MEV), x_limits=X_LIMITS,
                                       radial_upsample=LOW_T_UPSAMPLE)
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
    "TITLE = 'Previous vs current speed [$\\\\mathrm{m\\\\,s^{-1}}$]'",
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


LOW_T_MARKDOWN = r"""## Low temperature (0.01–10 MeV), logarithmic axis

Row mesh: on each of the 40 logarithmic rows, $n_B=x_{\rm in}(T)+f\,[x_{\rm out}(T)-x_{\rm in}(T)]$ strictly between that row's coexistence and udd roots, with $f$ from $10^{-5}$ to $1-10^{-4}$. Contours are computed in $(n_B,\log_{10}T)$, numerical solid and analytical dashed. The styling follows `26-10-02.ipynb` cell 8; the upper limit is now 10 MeV."""

LOG_FIGURE = "\n".join(["# Tunable plotting parameters"] + [
    f"{key.upper()} = {value!r}" for key, value in _iso_plot.LOG_REFERENCE_STYLE.items()] + [
    "XI = -0.85",
    "LOW_T_UPSAMPLE = 8    # display-only PCHIP refinement along rows in logit(f)",
    'OUTPUT_DIRECTORY = "local_analysis_results/26-10-27"',
]) + r'''

log_style = dict(iso_plot.LOG_REFERENCE_STYLE)
log_style.update({key.lower(): value for key, value in list(globals().items())
                  if key.isupper() and key.lower() in log_style})
analytic_lowT_retuned = dict(analytic_lowT, speed_m_s=iso_xi.retune(xi_cache_lowT, XI))
lines = {}
for name, payload in (("numerical", numerical_lowT), ("analytic", analytic_lowT_retuned)):
    x, T, logv = iso_plot.contour_arrays(payload, LOW_T_UPSAMPLE, "logit")
    lines[name] = iso_plot.contour_lines(x, T, logv, CONTOUR_LEVELS, log_T=True)

with plt.rc_context({"font.family": FONT_FAMILY, "mathtext.fontset": MATH_FONT_SET, "text.usetex": USE_TEX}):
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, constrained_layout=CONSTRAINED_LAYOUT,
                                facecolor=BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)
    iso_plot.draw_log_reference(axis, boundary_curves, lines["numerical"], lines["analytic"], XI, log_style)
    for extension in ("png", "pdf"):
        figure.savefig(ROOT / OUTPUT_DIRECTORY / f"{EXPORT_STEM}.{extension}", dpi=EXPORT_DPI,
                       facecolor=BACKGROUND_COLOR)
    plt.show()'''


XI_FIT_MARKDOWN = r"""## Tuning $\xi$ to the numerical speeds

`xi_fit/fit_xi.py` rescales the saved analytical intermediates exactly, $u(0^-;\xi)=\sqrt{2I_2}/\big(\lambda_n\sqrt{[1-a(0^+)][1+\xi a(0^+)]}\big)$, with no solver calls. It fits one $\xi$ per panel by minimizing the RMS displacement of the plotted contours along rays (high $T$) or rows (low $T$), in units of $n_0$, as for the adiabatic fit. A log-speed fit over all points and a separate anchor fit for every contour level are reported alongside. Near coexistence $a(0^+)\to0$, so $\xi$ barely moves those contours."""

XI_FIT_TABLE = r"""import json

XI_FIT_FILE = WORKFLOW / "xi_fit/xi_fit.json"
xi_fit = json.loads(XI_FIT_FILE.read_text())
for name, fit in xi_fit["fits"].items():
    print(f"{name:6s} contour-fit xi = {fit['contour']['xi']:+.3f}   log-speed-fit xi = {fit['log_speed']['xi']:+.3f}")
for tag, levels in xi_fit["level_anchors"].items():
    print(f"{tag} level anchors: " + ", ".join(f"{k}: {v['xi']:+.2f}" for k, v in levels.items()))
for label, entry in xi_fit["at_candidates"].items():
    print(f"xi = {entry['xi']:+.3f} ({label})")
    for tag in ("highT", "lowT"):
        e = entry[tag]
        print(f"   {tag}: contour offset RMS {e['rms_offset_n0']:.3f} n0, max {e['max_abs_offset_n0']:.3f} n0; "
              "analytic/numerical speed p5/50/95 " + ", ".join(f"{r:.2f}" for r in e["speed_ratio_percentiles_5_50_95"]))
XI_HIGH_T_FIT = round(xi_fit["fits"]["highT"]["contour"]["xi"], 2)
XI_LOW_T_FIT = round(xi_fit["fits"]["lowT"]["contour"]["xi"], 2)"""


def _replace_once(source, pairs):
    for old, new in pairs:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


XI_FIT_LINEAR = _replace_once(COMPARISON_FIGURE, (
    ("XI = -0.85", "XI = XI_HIGH_T_FIT   # contour fit on the polar mesh (T >= 10 MeV)\n"
                   "XI_LOW_T = XI_LOW_T_FIT   # contour fit on the row mesh (T < 10 MeV)"),
    ("iso_xi.retune(xi_cache_lowT, XI)", "iso_xi.retune(xi_cache_lowT, XI_LOW_T)"),
    ('rf"Analytical, $\\xi={XI:g}$"', 'rf"Analytical, $\\xi={XI:+g}$" "\\n" rf"(${XI_LOW_T:+g}$ below 10 MeV)"'),
    ("isothermal-comparison-linear\"", "isothermal-comparison-linear-xi-fit\""),
))

XI_FIT_LOG = _replace_once(LOG_FIGURE, (
    ("XI = -0.85", "XI = XI_LOW_T_FIT   # contour fit on the row mesh"),
    ("EXPORT_STEM = 'isothermal-lowT-logT'", "EXPORT_STEM = 'isothermal-lowT-logT-xi-fit'"),
))

XI_ANCHOR_FIGURE = r"""# Tunable plotting parameters
FIGURE_SIZE = (7.2, 5.6)
FIGURE_DPI = 160
EXPORT_DPI = 220
BACKGROUND_COLOR = 'white'
MATH_FONT_SET = 'cm'
HIGH_T_COLOR = '#d62728'
LOW_T_COLOR = '#1f77b4'
REFERENCE_COLOR = '0.35'
HIGH_T_MARKER = 'o'
LOW_T_MARKER = 's'
MARKER_SIZE = 9
LINE_WIDTH = 1.8
FIT_LINE_STYLE = '--'
REFERENCE_LINE_STYLE = ':'
X_SCALE = 'log'
X_LIMITS = (30.0, 2.0e6)
Y_LIMITS = (-1.0, 2.4)
TITLE = r'Contour-anchored $\xi$ per speed level'
TITLE_FONT_SIZE = 20
X_LABEL = r'Contour level $u(0^-)$ [m s$^{-1}$]'
Y_LABEL = r'$\xi$'
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
LEGEND_LOCATION = 'upper right'
TICK_DIRECTION = 'in'
EXPORT_STEM = 'local_analysis_results/26-10-27/isothermal-xi-anchors'

with plt.rc_context({"mathtext.fontset": MATH_FONT_SET}):
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, constrained_layout=True,
                                facecolor=BACKGROUND_COLOR)
    for tag, label, color, marker, fit in (("highT", "8–100 MeV", HIGH_T_COLOR, HIGH_T_MARKER, XI_HIGH_T_FIT),
                                           ("lowT", "0.01–10 MeV", LOW_T_COLOR, LOW_T_MARKER, XI_LOW_T_FIT)):
        anchors = xi_fit["level_anchors"][tag]
        axis.plot([float(k) for k in anchors], [v["xi"] for v in anchors.values()], marker=marker,
                  color=color, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=f"{label} anchors")
        axis.axhline(fit, color=color, linestyle=FIT_LINE_STYLE, linewidth=LINE_WIDTH,
                     label=rf"{label} best single $\xi={fit:+g}$")
    axis.axhline(iso_common.XI, color=REFERENCE_COLOR, linestyle=REFERENCE_LINE_STYLE, linewidth=LINE_WIDTH,
                 label=rf"previous $\xi={iso_common.XI:g}$")
    axis.set(xscale=X_SCALE, xlim=X_LIMITS, ylim=Y_LIMITS)
    axis.set_title(TITLE, fontsize=TITLE_FONT_SIZE)
    axis.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    axis.tick_params(which="both", direction=TICK_DIRECTION, top=True, right=True, labelsize=TICK_LABEL_FONT_SIZE)
    axis.legend(fontsize=LEGEND_FONT_SIZE, loc=LEGEND_LOCATION, framealpha=0.94)
    for extension in ("png", "pdf"):
        figure.savefig(ROOT / f"{EXPORT_STEM}.{extension}", dpi=EXPORT_DPI, facecolor=BACKGROUND_COLOR)
    plt.show()"""


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
    if all((HERE / f"lowT/isothermal-lowT-{m}.npy").is_file() for m in ("numerical", "analytic")):
        new += [nbformat.v4.new_markdown_cell(LOW_T_MARKDOWN), nbformat.v4.new_code_cell(LOG_FIGURE)]
        if (HERE / "xi_fit/xi_fit.json").is_file():
            new += [nbformat.v4.new_markdown_cell(XI_FIT_MARKDOWN)] + [
                nbformat.v4.new_code_cell(source) for source in (XI_FIT_TABLE, XI_FIT_LINEAR, XI_FIT_LOG,
                                                                  XI_ANCHOR_FIGURE)]
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
