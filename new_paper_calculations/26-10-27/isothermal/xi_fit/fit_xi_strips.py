"""Fit xi in temperature strips from the saved caches (no solver calls); writes xi_strips.json.

The contour fit of fit_xi.py, restricted to the plotted-contour crossings whose numerical crossing
temperature lies in a strip: half-decade strips on the row mesh (0.01-10 MeV) and 10 MeV strips on the
polar mesh (10-95 MeV), each panel with its own plotted levels. Offsets are tabulated once per panel on a
xi grid, so every strip, every single level and every bootstrap resample of the rays/rows crossing a strip
is a cheap reduction; the grid minimum is refined by a parabola. Error bars are the 16-84 % range of the
best xi over bootstrap resamples of those rays/rows.

    PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:. \
    ./bin/python3 -B new_paper_calculations/26-10-27/isothermal/xi_fit/fit_xi_strips.py
"""
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np  # noqa: E402

import _iso_common as common  # noqa: E402
import _iso_xi as xi_cache  # noqa: E402
from fit_xi import PANELS as PANEL_LEVELS, load_panel  # noqa: E402

XI_GRID = np.round(np.arange(-0.99, 4.0 + 1e-9, 0.01), 2)
STRIPS = dict(lowT=[(float(10.0 ** e), float(10.0 ** (e + 0.5))) for e in np.arange(-2.0, 1.0, 0.5)],
              highT=[(float(T), float(T + 10.0)) for T in range(10, 80, 10)] + [(80.0, 95.0)])
RESAMPLES = 400
SEED = 20261101
N0_PER_UNIT = common.POLAR["x_scale"]   # plot-normalized offset -> n_0


def tabulate(cache, numerical, levels, xi_grid=XI_GRID):
    """Contour offsets on the xi grid, shape (grid, level, ray or row), and the numerical crossing
    temperature of each level on each ray or row."""
    offsets = np.array([xi_cache.contour_offsets(cache, numerical, xi, levels) for xi in xi_grid])
    T = np.asarray(cache["T"], dtype=float)
    T_cross = np.array([xi_cache._ray_crossings(numerical["speed_m_s"], T, cache["fractions"], level)
                        for level in levels])
    return dict(offsets=offsets, T_cross=T_cross, levels=np.asarray(levels, dtype=float),
                xi_grid=np.asarray(xi_grid, dtype=float))


def _minimum(misfit, xi_grid):
    """Grid argmin refined by a parabola through its neighbours; True if it sits on the grid edge."""
    k = int(np.nanargmin(misfit))
    if k in (0, len(xi_grid) - 1) or not np.all(np.isfinite(misfit[k - 1:k + 2])):
        return float(xi_grid[k]), k in (0, len(xi_grid) - 1)
    left, middle, right = misfit[k - 1:k + 2]
    curvature = left - 2.0 * middle + right
    shift = 0.5 * (left - right) / curvature if curvature > 0 else 0.0
    return float(xi_grid[k] + shift * (xi_grid[1] - xi_grid[0])), False


def strip_fit(table, T_range, level=None, resamples=0, rng=None, reference_xi=()):
    """Best xi on the crossings with T_range[0] <= T <= T_range[1] (one level, or all levels), the RMS
    contour offset there [n_0] at it and at each reference xi, and the 16/84 % best xi over bootstrap
    resamples of the crossing rays/rows. None if the strip holds no crossing."""
    T_cross, xi_grid = table["T_cross"], table["xi_grid"]
    inside = (T_cross >= T_range[0]) & (T_cross <= T_range[1])
    if level is not None:
        inside &= table["levels"][:, None] == level
    if not inside.any():
        return None
    squared = np.where(inside, table["offsets"] ** 2, np.nan)      # (grid, level, ray)
    total, count = np.nansum(squared, axis=1), np.sum(np.isfinite(squared), axis=1)   # (grid, ray)
    with np.errstate(invalid="ignore", divide="ignore"):
        misfit = np.sqrt(total.sum(axis=1) / count.sum(axis=1))
    xi, at_bound = _minimum(misfit, xi_grid)
    rms = lambda value: float(N0_PER_UNIT * np.interp(value, xi_grid, misfit))
    fit = dict(T_range=[float(v) for v in T_range], T_median=float(np.median(T_cross[inside])),
               n_crossings=int(inside.sum()), n_rays=int(inside.any(axis=0).sum()), xi=xi, at_bound=at_bound,
               rms_n0=rms(xi), rms_n0_at={f"{value:+.2f}": rms(value) for value in reference_xi})
    if resamples:
        rays = np.flatnonzero(inside.any(axis=0))
        weights = rng.multinomial(rays.size, np.full(rays.size, 1.0 / rays.size), size=resamples)
        with np.errstate(invalid="ignore", divide="ignore"):
            curves = np.sqrt((weights @ total[:, rays].T) / (weights @ count[:, rays].T))   # (resample, grid)
        samples = [_minimum(curve, xi_grid)[0] for curve in curves if np.isfinite(curve).any()]
        fit.update(xi_16=float(np.percentile(samples, 16)), xi_84=float(np.percentile(samples, 84)))
    return fit


def main():
    started = time.perf_counter()
    panel_fits = json.loads((HERE / "xi_fit.json").read_text())["fits"]
    rng = np.random.default_rng(SEED)
    out = dict(xi_grid=[float(XI_GRID[0]), float(XI_GRID[-1]), float(XI_GRID[1] - XI_GRID[0])],
               resamples=RESAMPLES, seed=SEED, levels={tag: list(levels) for tag, levels in PANEL_LEVELS.items()},
               panel_xi={tag: panel_fits[tag]["contour"]["xi"] for tag in PANEL_LEVELS}, strips={})
    for tag, levels in PANEL_LEVELS.items():
        cache, numerical = load_panel(tag)
        table = tabulate(cache, numerical, levels)
        reference = (round(out["panel_xi"][tag], 2), common.XI)
        whole = strip_fit(table, (-np.inf, np.inf))
        print(f"{tag}: whole-panel xi {whole['xi']:+.3f} (fit_xi.py: {out['panel_xi'][tag]:+.3f})")
        out["strips"][tag] = []
        for T_range in STRIPS[tag]:
            fit = strip_fit(table, T_range, resamples=RESAMPLES, rng=rng, reference_xi=reference)
            if fit is None:
                continue
            fit["per_level"] = {f"{level:g}": per for level in levels
                                if (per := strip_fit(table, T_range, level, RESAMPLES, rng, reference)) is not None}
            out["strips"][tag].append(fit)
            print(f"  {T_range[0]:7.3g}-{T_range[1]:<6.3g} MeV  xi {fit['xi']:+.2f} [{fit['xi_16']:+.2f}, "
                  f"{fit['xi_84']:+.2f}]{' (bound)' if fit['at_bound'] else ''}  {fit['n_crossings']:3d} crossings "
                  f"on {fit['n_rays']:2d} rays/rows  RMS {fit['rms_n0']:.3f} n0 "
                  + " ".join(f"(xi {k}: {v:.3f})" for k, v in fit["rms_n0_at"].items())
                  + "\n      per level: " + ", ".join(f"{k}: {v['xi']:+.2f}" for k, v in fit["per_level"].items()))
    out["runtime_s"] = time.perf_counter() - started
    (HERE / "xi_strips.json").write_text(json.dumps(out, indent=1))
    print(f"{out['runtime_s']:.1f} s -> {HERE / 'xi_strips.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
