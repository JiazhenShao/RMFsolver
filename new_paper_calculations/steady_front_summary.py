#!/usr/bin/env python3
"""Steady-front shooting summary plots with analytic aQstar diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import RMFsolver.constants as const
import RMFsolver.phase_velocity as phase_velocity


FS = 1.5
BASE = 10
TITLE_FS = BASE * FS
LABEL_FS = BASE * FS
TICK_FS = BASE * FS
ANNOT_FS = 8 * FS
LEGEND_FS = 8 * FS
CBAR_FS = BASE * FS

DEFAULT_RESULTS_ROOT = Path("/Users/janshao/NSM_related/new_paper_calculations/26-03-31")
DEFAULT_CLUSTER_RUN = DEFAULT_RESULTS_ROOT / "steady_front_run_20260405_155749"
MU_K_BRANCH_TOL = 1.0e-3
DEFAULT_TARGET_X_MAX_NM = 50.0
TARGET_X_MAX_TOL_NM = 20.0


def _run_x_max_nm(run_dir: Path) -> float:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return np.nan
    config = json.loads(config_path.read_text())
    return float(config["x_max"]) * const.MeV_fm / 1.0e6


def _has_local_npz(run_dir: Path) -> bool:
    return any(run_dir.glob("trial_*.npz"))


def _find_default_cluster_run(results_root: Path, target_x_max_nm: float = DEFAULT_TARGET_X_MAX_NM) -> Path:
    run_dirs = sorted(path for path in results_root.glob("steady_front_run_*") if path.is_dir())
    if not run_dirs:
        raise RuntimeError(f"No steady_front_run_* folders found under {results_root}")

    short_domain_runs = [
        run_dir
        for run_dir in run_dirs
        if _has_local_npz(run_dir)
        and np.isfinite(_run_x_max_nm(run_dir))
        and abs(_run_x_max_nm(run_dir) - target_x_max_nm) <= TARGET_X_MAX_TOL_NM
    ]
    if short_domain_runs:
        return short_domain_runs[0]

    usable_runs = [run_dir for run_dir in run_dirs if _has_local_npz(run_dir)]
    if usable_runs:
        return usable_runs[0]

    return run_dirs[0]


def _find_latest_cluster_run(results_root: Path, require_npz: bool = True) -> Path:
    run_dirs = sorted(path for path in results_root.glob("steady_front_run_*") if path.is_dir())
    if require_npz:
        run_dirs = [run_dir for run_dir in run_dirs if _has_local_npz(run_dir)]
    if not run_dirs:
        raise RuntimeError(f"No steady_front_run_* folders found under {results_root}")
    return run_dirs[-1]


def _load_run_rows(run_dir: Path) -> tuple[dict, list[dict]]:
    config = json.loads((run_dir / "run_config.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "trial_summary.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return config, rows


def _branch_label_from_muK(muK_max: float) -> str:
    return "muK-poor" if abs(muK_max) < MU_K_BRANCH_TOL else "muK-rich"


def _analytic_aQstar_prediction(jB: float, nB_Q: float, nB_Qstar: float, DQ: float, eta: float, gamma: float) -> tuple[float, float]:
    if not np.isfinite([jB, nB_Q, nB_Qstar, DQ, eta, gamma]).all():
        return np.nan, np.nan
    if nB_Q <= 0.0 or nB_Qstar <= 0.0 or DQ <= 0.0:
        return np.nan, np.nan

    u_Q = jB / nB_Q
    disc = u_Q * u_Q + 4.0 * DQ * gamma * eta
    if disc < 0.0:
        return np.nan, np.nan

    lambda_val = (-u_Q + np.sqrt(disc)) / (2.0 * DQ)
    denom = DQ * lambda_val + jB / nB_Qstar
    if abs(denom) <= 1.0e-300:
        return lambda_val, np.nan

    aQstar_approx = (jB / nB_Q) / denom
    return lambda_val, aQstar_approx


def _enrich_rows(config: dict, rows: list[dict], run_dir: Path) -> list[dict]:
    enriched = []
    n0_plot = 0.16 * const.MeV_fm**3
    T_run = float(config["T"])
    B14_run = float(config["B_one_forth"])
    ms_run = float(config.get("ms", 0.0))

    for row in rows:
        matches = sorted(run_dir.glob(f"trial_{int(row['trial_id']):02d}__*.npz"))
        if not matches:
            continue

        npz_path = matches[0]
        with np.load(npz_path) as data:
            row = dict(row)
            row["npz_path_local"] = str(npz_path)
            row["merit"] = float(abs(row["a_end"]) + abs(row["q_end"]))
            row["muK_max"] = float(np.max(np.abs(data["muK"])))
            row["nB_span"] = float(np.max(data["nB"]) - np.min(data["nB"]))
            row["u_span"] = float(np.max(data["u"]) - np.min(data["u"]))
            row["aN"] = float(data["meta_a_N"]) if "meta_a_N" in data else float(row.get("a_N", np.nan))
            thresh = 0.01 * abs(float(data["a"][0]))
            idx = np.where(np.abs(data["a"]) <= thresh)[0]
            row["x_1pct"] = float(data["x"][idx[0]]) if len(idx) else np.nan
            row["jB_scaled"] = float(row["jB"] / n0_plot * const.m_per_sec)

            muB_Q = float(data["muB_Q"])
            muB_Qstar = float(data["muB_Qstar"])
            muK_Qstar = float(data["muK_Qstar"])
            jB = float(data["jB"])

            nB_Q_recomputed = float(phase_velocity.nB_QM(muB_Q, 0.0, B14_run, T_run, ms=ms_run))
            nB_Qstar_recomputed = float(phase_velocity.nB_QM(muB_Qstar, muK_Qstar, B14_run, T_run, ms=ms_run))
            nB_profile = np.asarray(data["nB"], dtype=float) if "nB" in data else np.array([], dtype=float)
            nB_Q = float(data["nB_Q"]) if "nB_Q" in data else nB_Q_recomputed
            nB_Qstar = float(nB_profile[0]) if nB_profile.size else nB_Qstar_recomputed

            micro = {
                "DQ": float(data["DQ"]) if "DQ" in data else np.nan,
                "eta": float(data["eta"]) if "eta" in data else np.nan,
                "gamma": float(data["gamma"]) if "gamma" in data else np.nan,
                "tau": float(data["tau"]) if "tau" in data else np.nan,
            }

            lambda_val, aQstar_approx = _analytic_aQstar_prediction(
                jB=jB,
                nB_Q=nB_Q,
                nB_Qstar=nB_Qstar,
                DQ=float(micro["DQ"]),
                eta=float(micro["eta"]),
                gamma=float(micro["gamma"]),
            )

            aQstar_actual = float(data["aQstar"])
            denom = max(abs(aQstar_actual), 1.0e-15)
            row["branch_label"] = _branch_label_from_muK(row["muK_max"])
            row["nB_Q"] = nB_Q
            row["nB_Q_recomputed"] = nB_Q_recomputed
            row["nB_Qstar"] = nB_Qstar
            row["nB_Qstar_recomputed"] = nB_Qstar_recomputed
            row["muB_Q"] = muB_Q
            row["muB_Qstar"] = muB_Qstar
            row["muK_Qstar"] = muK_Qstar
            row["DQ_diag"] = float(micro["DQ"])
            row["eta_diag"] = float(micro["eta"])
            row["gamma_diag"] = float(micro["gamma"])
            row["tau_diag"] = float(micro["tau"])
            row["lambda_diag"] = float(lambda_val)
            row["aQstar_approx"] = float(aQstar_approx)
            row["aQstar_delta"] = float(aQstar_approx - aQstar_actual)
            row["aQstar_abs_diff"] = float(abs(aQstar_approx - aQstar_actual))
            row["aQstar_rel_diff"] = float(abs(aQstar_approx - aQstar_actual) / denom)
            enriched.append(row)

    return sorted(enriched, key=lambda r: r["merit"])


def _make_color_norm(values: np.ndarray) -> LogNorm:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def add_nonoverlap_labels(ax, xs, ys, labels, radius_pts=12.0, min_sep_pts=16.0, max_radius_pts=34.0):
    fig = ax.figure
    fig.canvas.draw()
    pts_px = ax.transData.transform(np.column_stack([xs, ys]))
    if len(pts_px) == 0:
        return

    angles = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
    radius_px = radius_pts * fig.dpi / 72.0
    min_sep_px = min_sep_pts * fig.dpi / 72.0
    max_radius_px = max_radius_pts * fig.dpi / 72.0
    pos_px = pts_px + np.column_stack([radius_px * np.cos(angles), radius_px * np.sin(angles)])

    for _ in range(500):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                delta = pos_px[i] - pos_px[j]
                dist = np.hypot(delta[0], delta[1])
                if dist < min_sep_px:
                    if dist < 1.0e-9:
                        delta = np.array([1.0, 0.0])
                        dist = 1.0
                    shift = 0.5 * (min_sep_px - dist) * delta / dist
                    pos_px[i] += shift
                    pos_px[j] -= shift
                    moved = True

            delta0 = pos_px[i] - pts_px[i]
            dist0 = np.hypot(delta0[0], delta0[1])
            if dist0 < radius_px:
                if dist0 < 1.0e-9:
                    delta0 = np.array([1.0, 0.0])
                    dist0 = 1.0
                pos_px[i] += (radius_px - dist0) * delta0 / dist0
                moved = True
            elif dist0 > max_radius_px:
                pos_px[i] -= 0.18 * (dist0 - max_radius_px) * delta0 / dist0
                moved = True
        if not moved:
            break

    for xy_px, label_px, label in zip(pts_px, pos_px, labels):
        offset_pts = (label_px - xy_px) * 72.0 / fig.dpi
        ax.annotate(
            str(label),
            xy=ax.transData.inverted().transform(xy_px),
            xytext=tuple(offset_pts),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=ANNOT_FS,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45", alpha=0.7),
            zorder=5,
        )


def _print_run_summary(run_dir: Path, config: dict, all_rows: list[dict], plotted_rows: list[dict], success_only: bool):
    n_saved = len(all_rows)
    n_success_saved = sum(bool(r.get("success")) for r in all_rows)
    n_muK_poor = sum(_branch_label_from_muK(r["muK_max"]) == "muK-poor" for r in plotted_rows)
    n_muK_rich = len(plotted_rows) - n_muK_poor

    print(f"Selected run: {run_dir.name}")
    print(
        f"T = {config['T']} MeV, nB_N = {config['nB_factor']} n0, "
        f"B^(1/4) = {config['B_one_forth']} MeV"
    )
    print(
        f"x_max = {config['x_max']:.3e}, n_eval = {config['n_eval']}, "
        f"root_tol = {config['root_tol']}, max_nfev = {config['max_nfev']}"
    )
    print(f"x_max ~= {_run_x_max_nm(run_dir):.3f} nm")
    print(f"Trials with saved profiles: {n_saved}")
    print(f"Successful saved-profile trials: {n_success_saved}/{n_saved}")
    if success_only:
        print(f"Plotting successful trials only: {len(plotted_rows)}")
    else:
        print(f"Plotting saved-profile trials: {len(plotted_rows)}")
    print(f"muK-poor plotted trials: {n_muK_poor}, muK-rich plotted trials: {n_muK_rich}")
    print("Top trials by |a_end| + |q_end|:")
    for row in plotted_rows[:8]:
        branch = "poor" if row["branch_label"] == "muK-poor" else "rich"
        print(
            f"  trial {row['trial_id']:2d}: merit={row['merit']:.3e}, jB={row['jB']:.6f}, "
            f"aQstar={row['aQstar']:.6f}, muKmax={row['muK_max']:.3e}, branch={branch}, "
            f"a_end={row['a_end']:.3e}, q_end={row['q_end']:.3e}"
        )

    print("")
    print("Analytic aQstar diagnostic:")
    for row in plotted_rows:
        print(
            f"  trial {row['trial_id']:2d}: {row['branch_label']}, "
            f"jB/n0={row['jB_scaled']:.3f} m/s, "
            f"aQstar={row['aQstar']:.6f}, "
            f"aQstar_approx={row['aQstar_approx']:.6f}, "
            f"rel.diff={row['aQstar_rel_diff']:.3e}, "
            f"lambda={row['lambda_diag']:.3e}"
        )

    rel_diffs = np.array([row["aQstar_rel_diff"] for row in plotted_rows], dtype=float)
    if len(rel_diffs):
        print("")
        print(
            "Relative-difference summary: "
            f"median={np.nanmedian(rel_diffs):.3e}, "
            f"mean={np.nanmean(rel_diffs):.3e}, "
            f"max={np.nanmax(rel_diffs):.3e}"
        )


def plot_shooting_summary(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    run_dir: Path | None = None,
    output_path: Path | None = None,
    success_only: bool = False,
    show: bool = True,
):
    results_root = Path(results_root)
    if run_dir is None:
        preferred_run = DEFAULT_CLUSTER_RUN
        if preferred_run.exists():
            run_dir = preferred_run
        else:
            run_dir = _find_latest_cluster_run(results_root, require_npz=True)
    else:
        run_dir = Path(run_dir)
    config, rows = _load_run_rows(run_dir)
    all_rows = _enrich_rows(config, rows, run_dir)
    if not all_rows:
        raise RuntimeError(f"No trial .npz files found under {run_dir}")
    rows_sorted = [row for row in all_rows if bool(row.get("success"))] if success_only else list(all_rows)
    if not rows_sorted:
        raise RuntimeError(f"No plotted trials remain under {run_dir}")
    row_by_id = {row["trial_id"]: row for row in rows_sorted}

    _print_run_summary(run_dir, config, all_rows, rows_sorted, success_only=success_only)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    muK_color = np.array([max(row["muK_max"], 1.0e-8) for row in rows_sorted], dtype=float)
    color_norm = _make_color_norm(muK_color)
    color_map = plt.get_cmap("viridis")
    n0_plot = 0.16 * const.MeV_fm**3

    for row in rows_sorted:
        color = color_map(color_norm(max(row["muK_max"], 1.0e-8)))
        axes[0].plot(
            [row["jB_guess"] / n0_plot * const.m_per_sec, row["jB_scaled"]],
            [row["aQstar_guess"], row["aQstar"]],
            color=color,
            alpha=0.45,
            lw=1.2,
            zorder=1,
        )
        axes[0].scatter(
            row["jB_guess"] / n0_plot * const.m_per_sec,
            row["aQstar_guess"],
            marker="x",
            s=52,
            c=[color],
            linewidths=1.35,
            alpha=0.95,
            zorder=2,
        )
        axes[0].scatter(
            row["jB_scaled"],
            row["aQstar"],
            c=[max(row["muK_max"], 1.0e-8)],
            cmap="viridis",
            norm=color_norm,
            s=78,
            marker="o",
            edgecolors="r" if not row["success"] else "k",
            linewidths=0.9 if row["success"] else 1.4,
            zorder=3,
        )

    analytic_styles = {
        "muK-rich": {"color": "tab:orange", "label": "analytic approx (muK-rich)"},
        "muK-poor": {"color": "tab:blue", "label": "analytic approx (muK-poor)"},
    }
    analytic_handles = []
    branch_order = ["muK-rich", "muK-poor"]
    for branch in branch_order:
        branch_rows = [row for row in rows_sorted if row["branch_label"] == branch]
        if not branch_rows:
            continue
        branch_rows = sorted(branch_rows, key=lambda row: (row["jB_scaled"], row["trial_id"]))
        style = analytic_styles[branch]
        xs = np.array([row["jB_scaled"] for row in branch_rows], dtype=float)
        ys = np.array([row["aQstar_approx"] for row in branch_rows], dtype=float)
        axes[0].plot(
            xs,
            ys,
            linestyle="--",
            lw=2.0,
            color=style["color"],
            alpha=0.9,
            zorder=3.5,
        )
        axes[0].scatter(
            xs,
            ys,
            marker="D",
            s=52,
            facecolors="white",
            edgecolors=style["color"],
            linewidths=1.5,
            zorder=4,
        )
        analytic_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="--",
                marker="D",
                markerfacecolor="white",
                markersize=6.5,
                label=style["label"],
            )
        )

    sc = axes[1].scatter(
        [row["jB_scaled"] for row in rows_sorted],
        [row["merit"] for row in rows_sorted],
        c=muK_color,
        cmap="viridis",
        norm=color_norm,
        s=85,
        edgecolors="k",
        linewidths=0.6,
    )

    axes[0].set_xlabel(r"final $j_B / n_0$ [m/s]", fontsize=LABEL_FS)
    axes[0].set_ylabel(r"final $a_{Q\star}$", fontsize=LABEL_FS)
    title_suffix = " (successful trials)" if success_only else ""
    axes[0].set_title(f"Initial guess to final shooting solution{title_suffix}", fontsize=TITLE_FS)
    axes[1].set_xlabel(r"final $j_B / n_0$ [m/s]", fontsize=LABEL_FS)
    axes[1].set_ylabel(r"$|a_{Q}| + |q_{Q}|$", fontsize=LABEL_FS)
    axes[1].set_yscale("log")
    axes[1].set_title(f"Residual quality{title_suffix}", fontsize=TITLE_FS)
    cb = fig.colorbar(sc, ax=axes, shrink=0.95)
    cb.set_label(r"$\mu_K^Q$ [MeV]", fontsize=CBAR_FS)

    add_nonoverlap_labels(
        axes[0],
        np.array([row["jB_scaled"] for row in rows_sorted], dtype=float),
        np.array([row["aQstar"] for row in rows_sorted], dtype=float),
        [row["trial_id"] for row in rows_sorted],
    )
    add_nonoverlap_labels(
        axes[1],
        np.array([row["jB_scaled"] for row in rows_sorted], dtype=float),
        np.array([row["merit"] for row in rows_sorted], dtype=float),
        [row["trial_id"] for row in rows_sorted],
        radius_pts=10.0,
        min_sep_pts=15.0,
        max_radius_pts=28.0,
    )

    legend_handles = [
        Line2D([0], [0], marker="x", color="0.45", linestyle="None", markersize=7, markeredgewidth=1.35, label="initial guess"),
        Line2D([0], [0], marker="o", color="0.35", markerfacecolor="white", linestyle="None", markersize=7, label="final point"),
        Line2D([0], [0], color="0.5", lw=1.2, label="shooting path"),
    ]
    legend_handles.extend(analytic_handles)
    axes[0].legend(handles=legend_handles, fontsize=LEGEND_FS, loc="upper left")

    for ax in axes:
        ax.tick_params(axis="both", labelsize=TICK_FS)
    cb.ax.tick_params(labelsize=TICK_FS)

    output_path = Path(output_path) if output_path is not None else results_root / "shooting_summary.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    return {
        "results_root": results_root,
        "run_dir": run_dir,
        "config": config,
        "all_rows": all_rows,
        "rows": rows_sorted,
        "row_by_id": row_by_id,
        "figure": fig,
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot steady-front shooting summary with analytic aQstar diagnostics")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--success-only", action="store_true", help="Plot only successful trials")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it")
    args = parser.parse_args()

    plot_shooting_summary(
        results_root=args.results_root,
        run_dir=args.run_dir,
        output_path=args.output,
        success_only=args.success_only,
        show=args.show,
    )


if __name__ == "__main__":
    main()
