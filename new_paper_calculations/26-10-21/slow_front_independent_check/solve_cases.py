import pickle, sys, time, warnings
import numpy as np
warnings.simplefilter("ignore")
import RMFsolver.constants as const
import RMFsolver.phase_velocity as pv

N0 = 0.16 * const.MeV_fm**3
CASES = {
    "iso_T10_3n0_auto": ("isothermal", dict(T=10.0, nB_0minus=3 * N0, B_one_forth=180.0)),
    "iso_T20_3.5n0_a0.3": ("isothermal", dict(T=20.0, nB_0minus=3.5 * N0, B_one_forth=180.0, a_0plus=0.3)),
    "adb_T10_3n0_a0.55": ("adiabatic", dict(T_0minus=10.0, nB_0minus=3 * N0, B_one_forth=180.0, a_0plus=0.55)),
    "adb_T10_3n0_T0plus1": ("adiabatic", dict(T_0minus=10.0, nB_0minus=3 * N0, B_one_forth=180.0, T_0plus=1.0)),
}
out = {}
for name, (closure, kwargs) in CASES.items():
    t0 = time.time()
    fn = getattr(pv, f"solve_front_{closure}_slow")
    res = fn(**kwargs, return_profile=True)
    res["_closure"], res["_kwargs"], res["_seconds"] = closure, kwargs, time.time() - t0
    out[name] = res
    keys = ("success", "status", "jB", "u_0minus", "a_0plus", "T_0plus", "T_inf", "closure_error_max",
            "constitutive_residual_norm", "reaction_residual_norm", "diffusion_integral_residual_norm",
            "bvp_max_rms_residual", "path_nodes", "bvp_nodes")
    print(name, f"{res['_seconds']:.1f}s", {k: res.get(k) for k in keys}, res.get("conservation_errors"),
          res.get("message"), flush=True)
pickle.dump(out, open(sys.argv[1], "wb"))
print("saved")
