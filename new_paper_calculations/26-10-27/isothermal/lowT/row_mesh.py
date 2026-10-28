"""Low-temperature mesh: Cartesian in (radial fraction, log10 T).

Each row T sits between its own coexistence root x_in(T) and udd root
x_out(T); points are x = x_in + f (x_out - x_in), strictly inside, so no gray
point is evaluated. The rows are the 40 logarithmic fixed-T rows (0.01-10 MeV)
whose exact roots highT/build_boundary_curves.py already solved and cached.
The inner decades of f resolve the slow contours, which at 0.01 MeV lie within
~5e-4 of the window above coexistence.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "highT")]
import build_boundary_curves  # noqa: E402

FRACTIONS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.006, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15,
             0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90,
             0.95, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999)
T_ROWS = np.geomspace(0.01, 10.0, 40)   # the cached boundary rows


def row_roots(T_rows=T_ROWS):
    """Exact (x_inner, x_outer) of each cached boundary row."""
    data = build_boundary_curves.saved()
    roots = []
    for name in ("inner", "outer"):
        x, T = data[name + "_x"], data[name + "_T"]
        index = [int(np.flatnonzero(np.isclose(T, t, rtol=1e-12, atol=0))[0]) for t in T_rows]
        roots.append(x[index])
    return roots


def build_mesh(fractions=FRACTIONS, T_rows=T_ROWS):
    x_inner, x_outer = row_roots(T_rows)
    fractions = np.asarray(fractions, dtype=float)
    x = x_inner[:, None] + fractions[None, :] * (x_outer - x_inner)[:, None]
    T = np.broadcast_to(np.asarray(T_rows, dtype=float)[:, None], x.shape).copy()
    fingerprint = hashlib.sha256(np.concatenate([T_rows, fractions, x_inner, x_outer]).tobytes()).hexdigest()
    return dict(kind="rows", T_rows=np.asarray(T_rows, dtype=float), x_inner=x_inner, x_outer=x_outer,
                fractions=fractions, x=x, T=T, point_status=np.full(x.shape, "pending", dtype=object),
                fingerprint=fingerprint)
