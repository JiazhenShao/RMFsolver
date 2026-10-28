"""Dense gray-region boundary curves for the isothermal figures (static EOS only).

40 logarithmic fixed-T rows (0.01-10 MeV), 80 Chebyshev rays and the 41 mesh
rays as exact vertices. Writes highT/boundary-curves.npz; reruns reuse it
unless the solver or boundary sources changed.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402

import _iso_boundaries as boundaries  # noqa: E402
import polar_mesh  # noqa: E402

OUTPUT = HERE / "boundary-curves.npz"


def main(workers=4):
    key = repr(sorted(polar_mesh._boundary_key(np.array([])).items()))
    if OUTPUT.is_file() and str(np.load(OUTPUT)["key"]) == key:
        print(f"up to date: {OUTPUT}")
        return
    mesh = polar_mesh.build_mesh(workers=workers)
    curves = boundaries.boundary_curves(n_rows=40, n_rays=81, workers=workers,
                                        extra_rays=mesh["boundaries"])
    np.savez(OUTPUT, key=key, inner_x=curves["inner"][0], inner_T=curves["inner"][1],
             outer_x=curves["outer"][0], outer_T=curves["outer"][1])
    print(f"wrote {OUTPUT}: inner {len(curves['inner'][0])} and outer {len(curves['outer'][0])} points")


def load():
    data = np.load(OUTPUT)
    return dict(inner=(data["inner_x"], data["inner_T"]), outer=(data["outer_x"], data["outer_T"]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
