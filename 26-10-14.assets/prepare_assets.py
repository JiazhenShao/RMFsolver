"""Assemble meeting figures from saved results; no EOS or BVP solves."""
from pathlib import Path
import csv
import hashlib
import importlib
import json
import os
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent
SOURCES = {
    "rates-absolute.png": "ref-list/massless-weak-rate/results/exact-rates.png",
    "rates-relative-1200.png": "ref-list/massless-weak-rate/results/approximation-errors-muB-1200.png",
    "rates-relative-900.png": "ref-list/massless-weak-rate/results/approximation-errors-muB-900.png",
    "isothermal-speed-ratios.png": "ref-list/massless-weak-rate/results/speed-sensitivity.png",
    "adiabatic-speed-map.png": "ref-list/massless-weak-rate/results/adiabatic-speed-map.png",
    "adiabatic-trajectory.png": "ref-list/massless-weak-rate/results/adiabatic-trajectory-diagnostics.png",
    "adiabatic-diffusion-weighting.png": "ref-list/massless-weak-rate/results/adiabatic-diffusion-weighting.png",
    "isothermal-log-contours.png": "local_analysis_results/26-10-14/isothermal-contours-mfp-tail.png",
    "isothermal-core-mfp.png": "local_analysis_results/26-10-14/isothermal-core-mfp-f1e-4.png",
    "diffusive-flux-shapes.png": "local_analysis_results/26-09-29/xi-comparison/diffusive-flux-shapes.png",
    "constant-momentum-profile.png": "local_analysis_results/26-10-14/constant-momentum-profile.png",
    "aiso-accuracy.png": "local_analysis_results/26-10-14/aiso-accuracy.png",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = {}
for name, source in SOURCES.items():
    original, destination = ROOT / source, ASSETS / name
    shutil.copy2(original, destination)
    assert digest(original) == digest(destination)
    manifest[name] = {"source": source, "sha256": digest(destination)}

# Rescore the already saved analytical predictions, using the original metric.
scan_directory = ROOT / "local_analysis_results/26-10-12/xi-optimum"
sys.path.insert(0, str(scan_directory))
analysis = importlib.import_module("analyze")
old_directory = Path.cwd()
os.chdir(scan_directory)
try:
    data = analysis.table()
finally:
    os.chdir(old_directory)
xi_values = np.round(np.arange(-0.99, -0.295, 0.005), 3)
bands = ((50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000))
scores = {}
for method in analysis.METHODS:
    values = []
    for xi in xi_values:
        error = np.log(analysis.rescale(data[f"{method}_u0"], data[f"{method}_a"], xi) / data["u_num"])
        values.append(np.sqrt(np.mean([
            np.mean(error[(data["u_num"] >= lo) & (data["u_num"] < hi) & np.isfinite(error)]**2)
            for lo, hi in bands
        ])))
    scores[method] = np.asarray(values)
joint = (scores["analytic"] + scores["semi_analytic"]) / 2
best_index = int(joint.argmin())
current_index = int(np.argmin(abs(xi_values + 0.85)))
basin = xi_values[joint <= 1.01 * joint[best_index]]
assert xi_values[best_index] == -0.865
assert np.allclose([basin.min(), basin.max()], [-0.89, -0.845])

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/meeting-20261014-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tunable plotting parameters
FIGURE_SIZE = (10, 4.4)
EXPORT_DPI = 180
FONT_SIZE = 12
LINE_WIDTH = 2.2
CURRENT_COLOR = "#0072B2"
BEST_COLOR = "#D55E00"
CURVE_COLORS = {"analytic": "#0072B2", "semi_analytic": "#D55E00", "joint": "#202020"}
ZOOM_X_LIMITS = (-0.925, -0.78)

with plt.rc_context({"font.size": FONT_SIZE}):
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, layout="constrained")
    for method, label in (("analytic", "Full analytical"), ("semi_analytic", "Semi-analytical")):
        axes[0].plot(xi_values, scores[method], color=CURVE_COLORS[method],
                     linewidth=LINE_WIDTH, label=label)
    axes[0].plot(xi_values, joint, color=CURVE_COLORS["joint"],
                 linewidth=LINE_WIDTH, label="Joint score")
    axes[0].set(title="Saved 50-2000 m/s benchmarks", xlabel=r"$\xi$", ylabel="Band-balanced log-RMS score")
    axes[0].legend(frameon=False)
    axes[1].plot(xi_values, joint, color=CURVE_COLORS["joint"], linewidth=LINE_WIDTH)
    axes[1].axvspan(basin.min(), basin.max(), alpha=0.12, color=BEST_COLOR,
                    label="Within 1% of minimum")
    axes[1].axvline(xi_values[best_index], color=BEST_COLOR, linestyle=":", linewidth=LINE_WIDTH,
                    label=r"Best: $\xi=-0.865$")
    axes[1].axvline(-0.85, color=CURRENT_COLOR, linestyle="--", linewidth=LINE_WIDTH,
                    label=r"Current: $\xi=-0.85$")
    axes[1].set(title="Joint-score minimum", xlabel=r"$\xi$", xlim=ZOOM_X_LIMITS,
                 ylim=(float(joint.min()) * 0.995, float(joint.min()) * 1.13))
    axes[1].legend(frameon=False, fontsize=10, loc="upper center")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(ASSETS / "xi-optimum.png", dpi=EXPORT_DPI)
    plt.close(figure)

with (ASSETS / "xi-scan.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("xi", "full_band_rms", "semi_band_rms", "joint_score"))
    writer.writerows(zip(xi_values, scores["analytic"], scores["semi_analytic"], joint))
manifest["xi-optimum.png"] = {
    "source": "local_analysis_results/26-10-12/xi-optimum/analytical_xi-0.8500.jsonl and benchmark checkpoints via analyze.table()",
    "sha256": digest(ASSETS / "xi-optimum.png"),
    "best_xi": float(xi_values[best_index]),
    "best_score": float(joint[best_index]),
    "current_score": float(joint[current_index]),
    "current_relative_score_increase": float(joint[current_index] / joint[best_index] - 1),
    "one_percent_basin": [float(basin.min()), float(basin.max())],
}
(ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

# Extract the numbers used by the accompanying notes from the saved CSVs.
with (ROOT / "local_analysis_results/26-10-15/massless-phase-space/phase-space_all.csv").open() as stream:
    benchmark_rows = list(csv.DictReader(stream))
print("Local rate errors at muB=1200 MeV, T=1 MeV:")
for row in benchmark_rows:
    if float(row["mu_B_MeV"]) == 1200 and float(row["T_MeV"]) == 1:
        exact = float(row["rate_exact_net_MeV4"])
        print(row["a"], *[round(100*(float(row[key])/exact-1), 5) for key in
              ("rate_Madsen_Eq23_MeV4", "rate_Eq23_muB_over_3_MeV4", "rate_derived_T0_T2_MeV4")])
with (ROOT / "ref-list/massless-weak-rate/results/speed-sensitivity.csv").open() as stream:
    endpoint_rows = [row for row in csv.DictReader(stream) if float(row["a0plus"]) == 1]
print("Isothermal endpoint speed ratios:", [(row["T"], row["eq23_speed_ratio"], row["sommerfeld_T2_speed_ratio"]) for row in endpoint_rows])
print("Xi calibration:", manifest["xi-optimum.png"])
print(f"Prepared {len(manifest)} images with source provenance.")
