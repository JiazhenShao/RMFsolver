"""Ratio of every method's u(0-) to the matching slow (non-relativistic) BVP."""
import re
rows = {}
for log in ("run_state1.log", "run_states23.log"):
    for line in open(log):
        m = re.match(r"(T0=\S+ nB=\S+ (?:iso a0\+=\S+|adb T0\+=\S+)) \| (.+?)\s+(?:ok=\S+ )?u=([0-9.e+-]+)", line)
        if m:
            rows.setdefault(m.group(1), {})[m.group(2).strip()] = float(m.group(3))
cq = {  # consistent-quadratic speeds from consistent_quadratic.py
    "T0=10 nB=3n0 adb T0+=5": 5.622761e-07, "T0=10 nB=3n0 adb T0+=1": 6.639602e-07,
    "T0=20 nB=3.5n0 adb T0+=5": 1.626469e-06, "T0=5 nB=3n0 adb T0+=1": 6.390734e-07}
for case, d in rows.items():
    if case in cq:
        d["diag: consistent-quadratic endpoint"] = cq[case]
    bvp = d["NEW slow BVP"]
    print(case, f"(slow BVP u={bvp:.4e} = {bvp*299792458:.1f} m/s)")
    for k, v in d.items():
        if k != "NEW slow BVP":
            print(f"    {k:45s} {100*(v/bvp-1):+8.3f}%")
