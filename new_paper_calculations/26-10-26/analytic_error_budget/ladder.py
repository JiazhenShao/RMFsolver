"""Read-only ladder semi -> corrected full analytic, one approximation at a time (T0+=0, xi=-0.85)."""
import sys, warnings, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
warnings.simplefilter("ignore")
import RMFsolver.constants as const
import RMFsolver.phase_velocity as pv
from ablation_lib import speed, closed_I2, path_I2_variants, N0, B, XI
c = const.c_km * 1000
STATES = [(8.860992751744538, 3.0268624251597), (27.83373842080244, 2.96875),
          (70.6735943750199, 1.434488038438507), (45.610814207, 3.25), (81.347189695, 2.027839731),
          (91.70140516, 1.756937712), (10.15113868209788, 4.500181704610503), (102.106059811, 1.41642637)]
print(f"{'T0':>6} {'n/n0':>5} {'a_semi':>6} {'a_an':>6} | {'BVP':>7} {'semi':>7} | x LandauD  x cubic  x ellipse  x endpoint | {'analytic':>8} an/semi an/BVP")
for T0, f in STATES:
    nB0 = f * N0
    q = pv.analytic_velocity_adiabatic_slow(T0, nB0, B, T_0plus=0.0, xi=XI)
    s = pv.semi_analytic_velocity_adiabatic_slow(T0, nB0, B, T_0plus=0.0, xi=XI)
    b = pv.solve_front_adiabatic_slow(T0, nB0, B, T_0plus=0.0)
    ub = b["u_0minus"] * c if b.get("success") else np.nan
    path = pv._slow_front_path("adiabatic", q["P_0minus"], q["h_0minus"] / nB0, T0, B, 0.0, None, 0.0, 1e-6)
    I2_full, I2_L, I2_LC = path_I2_variants(path, s["muB_inf"])
    a_s, lam = s["a_0plus"], s["lambda_n"]
    u0 = s["u_0minus"]
    u1 = speed(I2_L, lam, a_s)
    u2 = speed(I2_LC, lam, a_s)
    u3 = speed(closed_I2(s["muB_inf"], a_s, a_s), lam, a_s)
    u4 = q["u_0minus"]
    print(f"{T0:6.1f} {f:5.2f} {a_s:6.3f} {q['a_0plus']:6.3f} | {ub:7.1f} {u0*c:7.1f} | {u1/u0:8.3f} {u2/u1:8.3f} {u3/u2:9.3f} {u4/u3:10.3f} | {u4*c:8.1f} {u4/u0:7.3f} {u4*c/ub:6.3f}", flush=True)
