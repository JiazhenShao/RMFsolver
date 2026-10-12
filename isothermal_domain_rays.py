"""Polar-ray tracer for the isothermal interface domain in (T_0minus, nB_0minus).

Two boundaries are traced in one pass, because both are read off the same
expensive upstream nuclear solve:

``delta_muB = 0``
    Equilibrated (uds, muK = 0) isothermal coexistence.  At common P and T the
    phase with the lower baryon chemical potential is favored, so this is the
    stable-neutron-matter gate already implemented in
    ``analytic_velocity_isothermal``.

``a_0plus = 1``
    The interface composition at its ceiling.  a_0plus is the local fraction
    nK(0+)/nB(0+), so its largest possible value is 1: the quark matter at the
    interface is at most as K-rich as the pure neutron matter it came from.
    The interface state is solved from the pressure match and that composition
    constraint, with muB(0+) free, and the boundary is where the state stops
    undercutting the upstream neutron matter in Gibbs energy per baryon:

        muB(0+) + a_0plus * muK(0+) = muB(0-).

The two boundaries are the two velocity limits of the isothermal front.  At
a_0plus -> 0 the Gibbs gap reduces identically to delta_muB, so the first
boundary is u(0-) = 0; the closed-form speed carries a 1/(1 - a_0plus) factor,
so the second is u(0-) -> infinity.  The front therefore exists strictly
between them, on the quark-favored side of delta_muB = 0.

Do not confuse a_0plus with the ``A_boundary`` of the energy-conserving
solvers.  That quantity belongs to the non-trivial-temperature problem, where
T(0+) != T(0-) and a conversion-layer trajectory selects the composition.  The
isothermal problem has only a_0plus, whose maximum is 1.

The mesh is the dimensionless elliptical polar coordinate of the 26-08-18
cluster ray grid: both curves are monotone decreasing in nB, hence star shaped
about an origin at the low-density, cold corner, so every ray meets each curve
exactly once and no 2D scan is needed.
"""

from __future__ import annotations

import os

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from scipy.optimize import brentq

from RMFsolver import RMFparameter as para
from RMFsolver import phase_velocity
from RMFsolver.constants import MeV_fm


# Calibration and window, matching new_paper_calculations/26-08-18.
N0_MEV3 = 0.16 * MeV_fm**3
B14_MEV = 189.1566
DENSITY_MIN_OVER_N0 = 1.0
DENSITY_MAX_OVER_N0 = 5.5
TEMPERATURE_MIN_MEV = 1.0e-2
TEMPERATURE_MAX_MEV = 120.0
NM_TYPE = "PNM"
MS_MEV = 0.0
UPB = 5000

# Warm-start window for the branch-validated muB inversion.  A cold call scans
# 48 points over [950, 2200] MeV and costs ~38 s; a narrow seeded window with a
# handful of points reproduces it to the last digit in ~8.5 s.  Cost is roughly
# 5 s + 0.7 s per scan point, so the window must be centred well: consecutive
# coarse probes on one ray move muB by ~80 MeV, which a fixed +-40 MeV window
# misses every time.  The centre is therefore extrapolated along the ray from
# the two nearest solved radii and the half width scales with the local slope.
MUB_COLD_LO_MEV = 950.0
MUB_COLD_HI_MEV = 2200.0
MUB_COLD_SCAN_POINTS = 48
MUB_WARM_SCAN_POINTS = 6
MUB_WARM_MIN_HALF_WIDTH_MEV = 45.0
MUB_WARM_MAX_HALF_WIDTH_MEV = 260.0
MUB_WARM_SLOPE_SAFETY = 0.6
MUB_SEED_HALF_WIDTH_MEV = 200.0

BOUNDARY_KINDS = ("delta_muB_equals_zero", "a_0plus_equals_one")

# The interface-state solver guards 0 < a_0plus < 1 strictly, so the ceiling is
# approached rather than hit.  The offset is far below any physical sensitivity.
A_0PLUS_CEILING = 1.0 - 1.0e-9


class RayTransform:
    """Dimensionless elliptical coordinates used only for mesh construction.

    ``temperature_min_MeV`` clamps the mapped temperature from below so that the
    endpoint-inclusive theta = 0 ray traces the cold isotherm rather than
    T = 0, which the RMF microphysics guard rejects.  The clamp only binds at
    radii far inside both boundaries.
    """

    def __init__(
        self,
        density_origin_over_n0: float = DENSITY_MIN_OVER_N0,
        density_scale_over_n0: float = DENSITY_MAX_OVER_N0 - DENSITY_MIN_OVER_N0,
        temperature_scale_MeV: float = TEMPERATURE_MAX_MEV,
        temperature_min_MeV: float = TEMPERATURE_MIN_MEV,
    ):
        self.density_origin_over_n0 = float(density_origin_over_n0)
        self.density_scale_over_n0 = float(density_scale_over_n0)
        self.temperature_scale_MeV = float(temperature_scale_MeV)
        self.temperature_min_MeV = float(temperature_min_MeV)
        if self.density_scale_over_n0 <= 0.0 or self.temperature_scale_MeV <= 0.0:
            raise ValueError("coordinate scales must be positive")

    def to_physical(self, rho, theta_rad):
        radius = np.asarray(rho, dtype=float)
        theta = np.asarray(theta_rad, dtype=float)
        density = (
            self.density_origin_over_n0
            + self.density_scale_over_n0 * radius * np.cos(theta)
        )
        temperature = np.maximum(
            self.temperature_scale_MeV * radius * np.sin(theta),
            self.temperature_min_MeV,
        )
        return density, temperature

    def to_hidden(self, nB_over_n0, temperature_MeV):
        density = np.asarray(nB_over_n0, dtype=float)
        temperature = np.asarray(temperature_MeV, dtype=float)
        X = (density - self.density_origin_over_n0) / self.density_scale_over_n0
        Y = temperature / self.temperature_scale_MeV
        return np.hypot(X, Y), np.arctan2(Y, X)

    def rho_upper_bound(self, theta_rad: float) -> float:
        """Largest radius on this ray still inside the physical window."""
        cosine = float(np.cos(theta_rad))
        sine = float(np.sin(theta_rad))
        limits = []
        if cosine > 1.0e-12:
            limits.append(
                (DENSITY_MAX_OVER_N0 - self.density_origin_over_n0)
                / (self.density_scale_over_n0 * cosine)
            )
        if sine > 1.0e-12:
            limits.append(TEMPERATURE_MAX_MEV / (self.temperature_scale_MeV * sine))
        if not limits:
            raise ValueError("ray does not intersect the physical window")
        return float(min(limits))

    def metadata(self) -> dict[str, Any]:
        return {
            "density_origin_over_n0": self.density_origin_over_n0,
            "density_scale_over_n0": self.density_scale_over_n0,
            "temperature_scale_MeV": self.temperature_scale_MeV,
            "temperature_min_MeV": self.temperature_min_MeV,
            "rho_definition": (
                "sqrt(((nB_over_n0-density_origin_over_n0)/"
                "density_scale_over_n0)^2+(T_MeV/temperature_scale_MeV)^2)"
            ),
            "theta_definition": (
                "atan2(T_MeV/temperature_scale_MeV,"
                "(nB_over_n0-density_origin_over_n0)/density_scale_over_n0)"
            ),
        }


def chebyshev_closed_axis(lower: float, upper: float, count: int) -> np.ndarray:
    """Endpoint-inclusive axis clustered toward both ends."""
    if int(count) < 2:
        raise ValueError("closed Chebyshev axis requires at least two points")
    parameter = np.linspace(0.0, 1.0, int(count))
    return float(lower) + (float(upper) - float(lower)) * 0.5 * (
        1.0 - np.cos(np.pi * parameter)
    )


def interface_residuals(
    nB_over_n0: float,
    temperature_MeV: float,
    *,
    B14: float = B14_MEV,
    n0: float = N0_MEV3,
    param=para.paraQMCRMF3,
    nm_type: str = NM_TYPE,
    ms: float = MS_MEV,
    upB: int = UPB,
    muB_window: tuple[float, float, int] | None = None,
) -> dict[str, Any]:
    """Return both boundary residuals from one upstream nuclear solve."""
    nB_0minus = float(nB_over_n0) * float(n0)
    temperature_MeV = float(temperature_MeV)

    inversion_windows = []
    if muB_window is not None:
        lo, hi, points = float(muB_window[0]), float(muB_window[1]), int(muB_window[2])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            inversion_windows.append(
                {
                    "muB_lo": max(lo, 1.0),
                    "muB_hi": hi,
                    "scan_points": points,
                }
            )
    inversion_windows.append(
        {
            "muB_lo": MUB_COLD_LO_MEV,
            "muB_hi": MUB_COLD_HI_MEV,
            "scan_points": MUB_COLD_SCAN_POINTS,
            "auto_expand": True,
        }
    )

    muB_0minus = np.nan
    last_error = ""
    for window in inversion_windows:
        try:
            muB_0minus = float(
                phase_velocity.muB_from_nB_physical(
                    nB_0minus,
                    temperature_MeV,
                    param=param,
                    NM_type=nm_type,
                    **window,
                )
            )
            break
        except Exception as exc:  # noqa: BLE001 - warm window may simply miss
            last_error = str(exc)
    if not np.isfinite(muB_0minus):
        raise RuntimeError(f"muB inversion failed: {last_error}")

    nuclear_state = phase_velocity._analytic_nuclear_state(
        muB_0minus,
        temperature_MeV,
        param=param,
        NM_type=nm_type,
    )
    P_0minus = float(nuclear_state["P_0minus"])

    # Boundary 1: equilibrated uds coexistence at common P and T.
    muB_0plus_equilibrated = float(
        phase_velocity._solve_muB_inf_at_muK0_for_given_Pi(
            P_0minus, 0.0, float(B14), temperature_MeV, ms=float(ms), upB=int(upB)
        )
    )
    delta_muB = muB_0plus_equilibrated - muB_0minus

    # Boundary 2: interface state with the composition pinned at its ceiling.
    # Both muB(0+) and muK(0+) are solved from the pressure match and
    # nK(0+)/nB(0+) = a_0plus, so the state exists on both sides of coexistence.
    # The residual is the Gibbs energy per baryon, g = muB + a muK, measured
    # against the upstream neutron matter; at a_0plus -> 0 it reduces
    # identically to delta_muB.
    muB_0plus = np.nan
    muK_0plus = np.nan
    nK_0plus = np.nan
    nB_0plus = np.nan
    residual_a_0plus = np.nan
    try:
        muB_0plus, muK_0plus = (
            float(value)
            for value in phase_velocity._solve_interface_0plus_from_local_a_and_Pi(
                A_0PLUS_CEILING,
                P_0minus,
                0.0,
                float(B14),
                temperature_MeV,
                ms=float(ms),
                upB=int(upB),
            )
        )
        nK_0plus = float(
            phase_velocity.nK_QM(
                muB_0plus, muK_0plus, float(B14), temperature_MeV,
                ms=float(ms), upB=int(upB),
            )
        )
        nB_0plus = float(
            phase_velocity.nB_QM(
                muB_0plus, muK_0plus, float(B14), temperature_MeV,
                ms=float(ms), upB=int(upB),
            )
        )
        residual_a_0plus = (
            muB_0plus + A_0PLUS_CEILING * muK_0plus - muB_0minus
        )
    except Exception:  # noqa: BLE001 - reported as a non-finite residual
        pass

    return {
        "nB_over_n0": float(nB_over_n0),
        "T_0minus_MeV": temperature_MeV,
        "muB_0minus_MeV": muB_0minus,
        "P_0minus_MeV4": P_0minus,
        "nB_0minus_MeV3": nB_0minus,
        "muB_0plus_equilibrated_MeV": muB_0plus_equilibrated,
        "delta_muB_MeV": float(delta_muB),
        "a_0plus": float(A_0PLUS_CEILING),
        "muB_0plus_MeV": float(muB_0plus),
        "muK_0plus_MeV": float(muK_0plus),
        "nK_0plus_MeV3": float(nK_0plus),
        "nB_0plus_MeV3": float(nB_0plus),
        "nB_0plus_over_n0": float(nB_0plus / float(n0)),
        "compression_ratio": float(nB_0plus / nB_0minus),
        "residual_a_0plus": float(residual_a_0plus),
    }


def _muB_inversion_window(
    solved: list[tuple[float, float]],
    rho: float,
) -> tuple[float, float, int] | None:
    """Predict a narrow muB window for radius ``rho`` from solved neighbours.

    With two or more solved radii the centre is linearly extrapolated along the
    ray, which tracks muB(rho) closely enough that a slope-scaled half width
    brackets the answer on the first try.
    """
    if not solved:
        return None
    ordered = sorted(solved, key=lambda point: abs(point[0] - rho))
    centre, half = ordered[0][1], MUB_SEED_HALF_WIDTH_MEV
    if len(ordered) >= 2:
        (rho_a, muB_a), (rho_b, muB_b) = ordered[0], ordered[1]
        if abs(rho_b - rho_a) > 1.0e-12:
            slope = (muB_b - muB_a) / (rho_b - rho_a)
            centre = muB_a + slope * (rho - rho_a)
            half = max(
                MUB_WARM_MIN_HALF_WIDTH_MEV,
                MUB_WARM_SLOPE_SAFETY * abs(slope) * abs(rho - rho_a),
            )
    half = min(half, MUB_WARM_MAX_HALF_WIDTH_MEV)
    return (centre - half, centre + half, MUB_WARM_SCAN_POINTS)


def _solve_ray(task: dict[str, Any]) -> dict[str, Any]:
    """Trace both boundaries along one ray.  Module level so it pickles."""
    import warnings

    from scipy.integrate import IntegrationWarning

    warnings.filterwarnings("ignore", category=IntegrationWarning)
    theta = float(task["theta_rad"])
    transform = RayTransform(**task["transform"])
    B14 = float(task["B14"])
    n0 = float(task["n0"])
    probe_count = int(task["probe_count"])
    rho_tol = float(task["rho_tol"])

    rho_upper = transform.rho_upper_bound(theta)
    rho_lower = float(task["rho_lower"])
    cache: dict[float, dict[str, Any]] = {}

    def evaluate(rho: float) -> dict[str, Any]:
        key = float(np.round(rho, 12))
        if key in cache:
            return cache[key]
        density, temperature = (
            float(value) for value in transform.to_physical(key, theta)
        )
        solved = [
            (row["rho"], row["muB_0minus_MeV"])
            for row in cache.values()
            if np.isfinite(row.get("muB_0minus_MeV", np.nan))
        ]
        row = interface_residuals(
            density,
            temperature,
            B14=B14,
            n0=n0,
            muB_window=_muB_inversion_window(solved, key),
        )
        row["rho"] = key
        row["theta_rad"] = theta
        cache[key] = row
        return row

    probes = np.linspace(rho_lower, rho_upper, probe_count)
    coarse_rows = [evaluate(float(value)) for value in probes]

    def bracket_and_solve(field: str, probe_radii: np.ndarray) -> dict[str, Any]:
        radii = np.asarray(probe_radii, dtype=float)
        values = np.asarray([evaluate(float(value))[field] for value in radii], dtype=float)
        brackets = [
            (float(radii[i]), float(radii[i + 1]))
            for i in range(radii.size - 1)
            if np.all(np.isfinite((values[i], values[i + 1])))
            and values[i] * values[i + 1] <= 0.0
        ]
        if len(brackets) != 1:
            return {
                "status": "no_unique_bracket",
                "theta_rad": theta,
                "bracket_count": len(brackets),
                "finite_probe_count": int(np.sum(np.isfinite(values))),
            }
        lower, upper = brackets[0]
        rho_root = float(
            brentq(
                lambda value: evaluate(float(value))[field],
                lower,
                upper,
                xtol=rho_tol,
                rtol=1.0e-12,
            )
        )
        record = dict(evaluate(rho_root))
        record["status"] = "root"
        record["residual_field"] = field
        record["bracket_lower_rho"] = lower
        record["bracket_upper_rho"] = upper
        return record

    # Both residuals are defined across the whole ray and each changes sign
    # once, so the same coarse scan brackets both.
    delta_record = bracket_and_solve("delta_muB_MeV", probes)
    a_record = bracket_and_solve("residual_a_0plus", probes)

    return {
        "theta_rad": theta,
        "theta_deg": float(np.degrees(theta)),
        "rho_lower": rho_lower,
        "rho_upper": rho_upper,
        "delta_muB_equals_zero": delta_record,
        "a_0plus_equals_one": a_record,
        "evaluation_count": len(cache),
        "probe_rows": [dict(row) for row in cache.values()],
    }


def solve_domain(
    *,
    theta_count: int = 12,
    probe_count: int = 10,
    rho_lower: float = 0.02,
    rho_tol: float = 1.0e-4,
    B14: float = B14_MEV,
    n0: float = N0_MEV3,
    max_workers: int = 8,
    transform: RayTransform | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Trace both boundaries on ``theta_count`` endpoint-inclusive rays."""
    transform = transform or RayTransform()
    theta_axis = chebyshev_closed_axis(0.0, 0.5 * np.pi, theta_count)
    transform_kwargs = {
        "density_origin_over_n0": transform.density_origin_over_n0,
        "density_scale_over_n0": transform.density_scale_over_n0,
        "temperature_scale_MeV": transform.temperature_scale_MeV,
        "temperature_min_MeV": transform.temperature_min_MeV,
    }
    tasks = [
        {
            "theta_rad": float(theta),
            "transform": transform_kwargs,
            "B14": float(B14),
            "n0": float(n0),
            "probe_count": int(probe_count),
            "rho_lower": float(rho_lower),
            "rho_tol": float(rho_tol),
        }
        for theta in theta_axis
    ]

    rays: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
        for index, ray in enumerate(executor.map(_solve_ray, tasks), start=1):
            rays.append(ray)
            if progress:
                statuses = " ".join(
                    f"{kind.split('_')[0]}={ray[kind]['status']}"
                    for kind in BOUNDARY_KINDS
                )
                print(
                    f"  ray {index:2d}/{len(tasks)}  "
                    f"theta={ray['theta_deg']:5.1f} deg  "
                    f"evals={ray['evaluation_count']:3d}  {statuses}",
                    flush=True,
                )

    rays.sort(key=lambda ray: ray["theta_rad"])
    return {
        "rays": rays,
        "theta_axis_rad": theta_axis,
        "transform": transform.metadata(),
        "calibration": {
            "B14_MeV": float(B14),
            "n0_MeV3": float(n0),
            "NM_type": NM_TYPE,
            "ms_MeV": MS_MEV,
            "upB": UPB,
            "parameter_set": "paraQMCRMF3",
        },
        "window": {
            "density_min_over_n0": DENSITY_MIN_OVER_N0,
            "density_max_over_n0": DENSITY_MAX_OVER_N0,
            "temperature_min_MeV": TEMPERATURE_MIN_MEV,
            "temperature_max_MeV": TEMPERATURE_MAX_MEV,
        },
    }


def save_payload(payload: dict[str, Any], path) -> None:
    """Write a traced domain to a pickled ``.npy`` payload, as the cluster does."""
    np.save(str(path), payload, allow_pickle=True)


def load_payload(path) -> dict[str, Any]:
    """Read a payload written by :func:`save_payload`."""
    return np.load(str(path), allow_pickle=True).item()


def boundary_curve(payload: dict[str, Any], kind: str) -> dict[str, np.ndarray]:
    """Collect one traced boundary into plot-ready arrays, ordered in theta."""
    if kind not in BOUNDARY_KINDS:
        raise ValueError(f"kind must be one of {BOUNDARY_KINDS}")
    rows = [
        ray[kind]
        for ray in payload["rays"]
        if ray[kind].get("status") == "root"
    ]
    if not rows:
        return {name: np.asarray([]) for name in ("nB_over_n0", "T_0minus_MeV")}
    return {
        "theta_deg": np.asarray([np.degrees(row["theta_rad"]) for row in rows]),
        "nB_over_n0": np.asarray([row["nB_over_n0"] for row in rows]),
        "T_0minus_MeV": np.asarray([row["T_0minus_MeV"] for row in rows]),
        "muB_0minus_MeV": np.asarray([row["muB_0minus_MeV"] for row in rows]),
        "muK_0plus_MeV": np.asarray([row["muK_0plus_MeV"] for row in rows]),
        "nB_0plus_over_n0": np.asarray([row["nB_0plus_over_n0"] for row in rows]),
        "compression_ratio": np.asarray([row["compression_ratio"] for row in rows]),
        "delta_muB_MeV": np.asarray([row["delta_muB_MeV"] for row in rows]),
        "residual_a_0plus": np.asarray([row["residual_a_0plus"] for row in rows]),
    }


# Neutron-side equilibrated coexistence densities published in
# wiki/physics/isothermal-analytic-front-speed.md for B14 = 189.1566 MeV.
# T = 0 is not an admissible state of the microphysics, so the cold entry is
# checked at TEMPERATURE_MIN_MEV.
WIKI_COEXISTENCE_REFERENCE = (
    (0.0, 3.0000),
    (20.0, 2.8335),
    (40.0, 2.2923),
    (60.0, 1.3247),
)


def _validation_task(task: dict[str, Any]) -> dict[str, Any]:
    import warnings

    from scipy.integrate import IntegrationWarning

    warnings.filterwarnings("ignore", category=IntegrationWarning)
    temperature = float(task["temperature_MeV"])
    reference = float(task["reference_nB_over_n0"])
    try:
        solved = coexistence_at_fixed_temperature(
            max(temperature, TEMPERATURE_MIN_MEV),
            B14=float(task["B14"]),
            n0=float(task["n0"]),
            density_bounds=task["density_bounds"],
            probe_count=int(task["probe_count"]),
        )
        return {
            "status": "root",
            "temperature_MeV": temperature,
            "reference_nB_over_n0": reference,
            "solved_nB_over_n0": float(solved),
            "absolute_error": float(solved - reference),
        }
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {
            "status": "failed",
            "temperature_MeV": temperature,
            "reference_nB_over_n0": reference,
            "solved_nB_over_n0": np.nan,
            "absolute_error": np.nan,
            "message": str(exc),
        }


def validate_against_wiki(
    *,
    B14: float = B14_MEV,
    n0: float = N0_MEV3,
    density_half_width: float = 0.45,
    probe_count: int = 8,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Re-solve the published coexistence densities on fixed-T paths.

    This is an independent check of the ray trace: a different path mode, its
    own brackets, seeded only by the published density.
    """
    tasks = [
        {
            "temperature_MeV": temperature,
            "reference_nB_over_n0": reference,
            "B14": float(B14),
            "n0": float(n0),
            "density_bounds": (
                max(reference - density_half_width, 0.3),
                reference + density_half_width,
            ),
            "probe_count": int(probe_count),
        }
        for temperature, reference in WIKI_COEXISTENCE_REFERENCE
    ]
    with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
        return list(executor.map(_validation_task, tasks))


def coexistence_at_fixed_temperature(
    temperature_MeV: float,
    *,
    B14: float = B14_MEV,
    n0: float = N0_MEV3,
    density_bounds: tuple[float, float] = (0.5, 5.5),
    probe_count: int = 12,
    field: str = "delta_muB_MeV",
) -> float:
    """Independent fixed-T check of one boundary, for wiki-table validation."""
    lower, upper = density_bounds
    probes = np.linspace(lower, upper, int(probe_count))
    solved: list[tuple[float, float]] = []

    def residual(density: float) -> float:
        row = interface_residuals(
            float(density), float(temperature_MeV),
            B14=B14, n0=n0,
            muB_window=_muB_inversion_window(solved, float(density)),
        )
        solved.append((float(density), float(row["muB_0minus_MeV"])))
        return float(row[field])

    values = [residual(float(value)) for value in probes]
    for index in range(len(values) - 1):
        if (
            np.all(np.isfinite((values[index], values[index + 1])))
            and values[index] * values[index + 1] <= 0.0
        ):
            return float(
                brentq(
                    residual,
                    float(probes[index]),
                    float(probes[index + 1]),
                    xtol=1.0e-5,
                    rtol=1.0e-12,
                )
            )
    raise RuntimeError(
        f"no sign change of {field} at T = {temperature_MeV} MeV"
    )
