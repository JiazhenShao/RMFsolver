"""Fit the analytical xi to the numerical isothermal speeds from the saved caches (no solver calls).

Per panel (high T: polar mesh, levels 50-5000 m/s; low T: row mesh, levels 100-1e6 m/s)
and for both panels together:
- contour fit: one xi minimizing the RMS plot-normalized displacement of the plotted
  contours, (n_B/n_0 - 1)/4.5 along rays or rows (the visual criterion of the adiabatic fit);
- log-speed fit: one xi minimizing the RMS of ln(v_analytic/v_numerical) over all points;
- level anchors: the contour fit for each plotted level on its own.
Offsets are reported in n_0 units (normalized offset x 4.5). Writes xi_fit.json.

    PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:. \
    ./bin/python3 -B new_paper_calculations/26-10-27/isothermal/xi_fit/fit_xi.py
"""
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

import _iso_common as common  # noqa: E402
import _iso_plot as plot  # noqa: E402
import _iso_xi as xi_cache  # noqa: E402

BOUNDS = (-0.99, 10.0)
REFERENCE_XI = common.XI
PANELS = dict(highT=plot.LINEAR_REFERENCE_STYLE["velocity_contour_levels_m_s"],
              lowT=plot.LOG_REFERENCE_STYLE["contour_levels"])


def load_panel(tag):
    analytic = common.load(HERE.parent / f"{tag}/isothermal-{tag}-analytic.npy")
    numerical = common.load(HERE.parent / f"{tag}/isothermal-{tag}-numerical.npy")
    return xi_cache.build_cache(analytic), numerical


def _minimize(misfit):
    result = minimize_scalar(misfit, bounds=BOUNDS, method="bounded", options=dict(xatol=1e-6))
    xi = float(result.x)
    return xi, bool(min(xi - BOUNDS[0], BOUNDS[1] - xi) < 1e-3)


def contour_misfit(panels, levels_by_panel):
    def misfit(xi):
        offsets = np.concatenate([xi_cache.contour_offsets(cache, numerical, xi, levels_by_panel[tag]).ravel()
                                  for tag, (cache, numerical) in panels.items()])
        return float(np.sqrt(np.nanmean(offsets**2)))
    return misfit


def log_speed_misfit(panels):
    def misfit(xi):
        residual = np.concatenate([(np.log(xi_cache.retune(cache, xi)) - np.log(numerical["speed_m_s"])).ravel()
                                   for cache, numerical in panels.values()])
        return float(np.sqrt(np.nanmean(residual**2)))
    return misfit


def summary(panels, levels_by_panel, xi):
    """Contour offsets [n_0] and analytic/numerical speed ratios at one xi."""
    out = {}
    for tag, (cache, numerical) in panels.items():
        offsets = 4.5 * xi_cache.contour_offsets(cache, numerical, xi, levels_by_panel[tag])
        ratio = xi_cache.retune(cache, xi) / numerical["speed_m_s"]
        ratio = ratio[np.isfinite(ratio)]
        out[tag] = dict(
            rms_offset_n0=float(np.sqrt(np.nanmean(offsets**2))),
            max_abs_offset_n0=float(np.nanmax(np.abs(offsets))),
            per_level_rms_offset_n0={f"{level:g}": float(np.sqrt(np.nanmean(row**2)))
                                     for level, row in zip(levels_by_panel[tag], offsets)},
            per_level_mean_offset_n0={f"{level:g}": float(np.nanmean(row))
                                      for level, row in zip(levels_by_panel[tag], offsets)},
            speed_ratio_percentiles_5_50_95=[float(v) for v in np.percentile(ratio, [5, 50, 95])])
    return out


def main():
    started = time.perf_counter()
    panels = {tag: load_panel(tag) for tag in PANELS}
    fits = {}
    for name, members in (("highT", ("highT",)), ("lowT", ("lowT",)), ("both", ("highT", "lowT"))):
        subset = {tag: panels[tag] for tag in members}
        contour_xi, contour_at_bound = _minimize(contour_misfit(subset, PANELS))
        speed_xi, speed_at_bound = _minimize(log_speed_misfit(subset))
        fits[name] = dict(
            contour=dict(xi=contour_xi, at_bound=contour_at_bound, rms_normalized=contour_misfit(subset, PANELS)(contour_xi)),
            log_speed=dict(xi=speed_xi, at_bound=speed_at_bound, rms_log=log_speed_misfit(subset)(speed_xi)))
    anchors = {}
    for tag, levels in PANELS.items():
        anchors[tag] = {}
        for level in levels:
            xi, at_bound = _minimize(contour_misfit({tag: panels[tag]}, {tag: (level,)}))
            anchors[tag][f"{level:g}"] = dict(xi=xi, at_bound=at_bound)
    candidates = {"reference": REFERENCE_XI, **{f"{name}_contour": fit["contour"]["xi"] for name, fit in fits.items()}}
    out = dict(bounds=BOUNDS, levels={tag: list(levels) for tag, levels in PANELS.items()}, fits=fits,
               level_anchors=anchors,
               at_candidates={label: dict(xi=xi, **summary(panels, PANELS, xi)) for label, xi in candidates.items()},
               provenance={tag: dict(cache=cache["provenance"], numerical_identity_sources=numerical["identity"]["sources"])
                           for tag, (cache, numerical) in panels.items()},
               runtime_s=time.perf_counter() - started)
    (HERE / "xi_fit.json").write_text(json.dumps(out, indent=1, default=str))
    for name, fit in fits.items():
        print(f"{name:6s} contour xi = {fit['contour']['xi']:+.3f}   log-speed xi = {fit['log_speed']['xi']:+.3f}"
              + ("   (a fit hit the bound)" if fit["contour"]["at_bound"] or fit["log_speed"]["at_bound"] else ""))
    for tag, levels in anchors.items():
        print(f"{tag} level anchors: " + ", ".join(f"{k}: {v['xi']:+.2f}" for k, v in levels.items()))
    for label, entry in out["at_candidates"].items():
        print(f"xi = {entry['xi']:+.3f} ({label}): " + "; ".join(
            f"{tag} RMS {entry[tag]['rms_offset_n0']:.3f} n0, max {entry[tag]['max_abs_offset_n0']:.3f} n0, "
            f"ratio p5/50/95 {', '.join(f'{r:.2f}' for r in entry[tag]['speed_ratio_percentiles_5_50_95'])}"
            for tag in PANELS))
    print(f"{out['runtime_s']:.1f} s -> {HERE / 'xi_fit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
