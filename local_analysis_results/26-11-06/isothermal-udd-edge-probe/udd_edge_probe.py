"""Probe solve_front_isothermal_slow next to the upper (udd) gray boundary of the
high-T isothermal contour figure, on the polar rays of the shipped mesh.

For each point: production settings, the default (tail-shaped) initial guess, tight
tolerances, and a 3x longer domain. Prints speed, a(0+), validation and
v*sqrt(1-a(0+)), which must tend to a constant if v ~ (1-a)^(-1/2) at the boundary.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/janshao/NSM_related")
WORKFLOW = ROOT / "new_paper_calculations/26-10-27/isothermal"
sys.path.insert(0, str(WORKFLOW))
import _iso_common as common  # noqa: E402

OUT = Path(__file__).with_name("udd_edge_probe.json")   # run from the repo root with the SciPy runtime on PYTHONPATH
RAYS = {80: (0.5, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 1 - 1e-6, 1 - 1e-7),
        70: (0.9, 0.99, 0.999, 0.9999, 0.99999, 1 - 1e-6),
        30: (0.9, 0.99, 0.999, 0.9999, 0.99999)}
TIGHT = dict(tol_bvp=1e-6, n_mesh=1200, tail_eps=1e-10, path_rtol=1e-7, max_nodes=200000)
VARIANTS = {"production": {}, "default_guess": dict(initial_guess_mode="default"),
            "tight": TIGHT, "long_domain": dict(kappa_factor=3.0)}


def solve(task):
    from RMFsolver import phase_velocity as pv
    import _iso_evaluate as evaluate
    ray, f, x, T, variant = task
    settings = dict(common.NUMERICAL, **VARIANTS[variant])
    started = time.perf_counter()
    try:
        r = pv.solve_front_isothermal_slow(T, x * common.N0_MEV3, common.PHYSICS["B_one_forth"], **settings)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        r = dict(success=False, message=str(exc))
    reason = evaluate.numerical_acceptance(r, settings["tol_bvp"]) if r.get("success") else r.get("message")
    a = float(r.get("a_0plus", np.nan))
    v = float(r.get("u_0minus", np.nan)) * common.C_M_S
    return dict(ray=ray, f=f, x=x, T=T, variant=variant, success=bool(r.get("success")), reason=reason,
                message=r.get("message", ""), method=r.get("solution_method"), guess=r.get("initial_guess_used"),
                v=v, a=a, v_sqrt=v * np.sqrt(1 - a) if 0 < a < 1 else np.nan,
                rms=float(r.get("bvp_max_rms_residual", np.nan)), nodes=r.get("bvp_nodes"),
                cons=float(max(r.get("conservation_errors", {"x": np.nan}).values())),
                bres=float(np.max(np.abs(r.get("boundary_residuals", [np.nan])))),
                resid=float(max(r.get(k, np.nan) for k in ("constitutive_residual_norm", "reaction_residual_norm",
                                                            "diffusion_integral_residual_norm"))),
                wall=time.perf_counter() - started)


def main():
    mesh = common.load(WORKFLOW / "highT/isothermal-highT-numerical.npy")["mesh"]
    tasks = []
    for ray, fractions in RAYS.items():
        b = mesh["boundaries"][ray]
        for f in fractions:
            x, T = common.to_physical(b["rho_inner"] + f * (b["rho_outer"] - b["rho_inner"]), b["theta"])
            tasks += [(ray, f, float(x), float(T), variant) for variant in VARIANTS]
    rows = common.map_parallel(solve, tasks, int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    for row in rows:
        print(f"ray {row['ray']} 1-f={1 - row['f']:.0e} x={row['x']:.4f} T={row['T']:.5f} {row['variant']:>13} "
              f"ok={row['success']!s:5} v={row['v']:.6g} 1-a={1 - row['a']:.3e} v*sqrt(1-a)={row['v_sqrt']:.4f} "
              f"{row['method']}/{row['guess']} rms={row['rms']:.1e} resid={row['resid']:.1e} "
              f"cons={row['cons']:.1e} bres={row['bres']:.1e} {row['wall']:.0f}s {row['reason'] or ''}", flush=True)


if __name__ == "__main__":
    main()
