"""Exact cold massless weak phase space for 0 <= mu_s <= mu_u <= mu_d.

Run with the repository Python interpreter to reproduce the symbolic derivation
and checks.  ``cold_phase_space`` itself needs only the Python standard library.
The weak prefactor is NOT included: Gamma0 = 9*GF**2*sin(theta_C)**2
*cos(theta_C)**2/(8*pi**5) * cold_phase_space(mu_u, mu_d, mu_s).
"""

from functools import lru_cache
from math import fsum, isfinite


def cold_phase_space(mu_u, mu_d, mu_s):
    """Return I0 (energy**8), with cancellation controlled near equilibrium."""
    if not all(isfinite(value) for value in (mu_u, mu_d, mu_s)):
        raise ValueError("Chemical potentials must be finite")
    if not 0 <= mu_s <= mu_u <= mu_d:
        raise ValueError("Require 0 <= mu_s <= mu_u <= mu_d")
    delta = mu_d - mu_s
    if delta == 0 or mu_u == 0:
        return 0.0

    if delta <= mu_u:
        # Expand around equal potentials before floating-point evaluation.
        # v+w=1; every term retains the required cubic imbalance scaling.
        v = (mu_d - mu_u) / delta
        w = (mu_u - mu_s) / delta
        ratio = delta / mu_u
        coefficients = (
            128 / 45,
            8 / 9 * (v**4 - v**3*w - 9*v**2*w**2 - 17*v*w**3 - 7*w**4),
            -8 / 45 * (3*v**5 + 10*v**4*w - 50*v*w**4 - 27*w**5),
            4 / 45 * (v**6 + 12*v**5*w + 30*v**4*w**2
                      + 30*v**2*w**4 - 12*v*w**5 - 15*w**6),
            8 / 315 * (5*v**7 + 14*v**6*w - 35*v**4*w**3
                       + 35*v**3*w**4 + 5*w**7),
            -1 / 315 * (15*v**8 + 80*v**7*w + 168*v**6*w**2
                        + 168*v**5*w**3 + 70*v**4*w**4 + 168*v**3*w**5
                        + 168*v**2*w**6 + 88*v*w**7 + 21*w**8),
        )
        return mu_u**5 * delta**3 * fsum(
            coefficient * ratio**power
            for power, coefficient in enumerate(coefficients)
        )

    u, d, s = mu_u, mu_d, mu_s
    return fsum((
        u**4 * (210*d**4 - 112*d**3*u - 84*d**2*u**2 - 32*d*u**3 + 65*u**4),
        s**4 * (168*d**3*s - 280*d**3*u - 168*d**2*s**2 + 504*d**2*s*u
                - 420*d**2*u**2 + 80*d*s**3 - 336*d*s**2*u + 504*d*s*u**2
                - 280*d*u**3 - 18*s**4 + 64*s**3*u - 196*s**2*u**2
                + 168*s*u**3 + 210*u**4),
    )) / 315


def derive_symbolically():
    """Integrate the ordered regions and return exact SymPy polynomials."""
    import sympy as sp

    x, y, en, m, u, d, s = sp.symbols("x y en m u d s")
    kernel = sp.Rational(32, 3)*en**2*m**3 - 16*en*m**4 + sp.Rational(32, 5)*m**5
    integrated_energy = sp.integrate(kernel, (en, u+y, x+d))
    lower_integrand = sp.integrate(integrated_energy.subs(m, x), (y, s, x+d-u))
    upper_regions = (
        sp.integrate(sp.integrate(integrated_energy.subs(m, y), (y, s, x)), (x, s, u))
        + sp.integrate(sp.integrate(integrated_energy.subs(m, x), (y, x, x+d-u)), (x, s, u))
    )
    small = sp.factor(sp.integrate(lower_integrand, (x, u+s-d, s)) + upper_regions)
    large = sp.factor(sp.integrate(lower_integrand, (x, 0, s)) + upper_regions)
    correction = (u+s-d)**6 * (15*d**2 + 10*d*(s+u) + 3*(s+u)**2) / 315
    assert sp.expand(large-small-correction) == 0
    return {"symbols": (u, d, s), "small": small, "large": large,
            "correction": correction, "lower_integrand": sp.factor(lower_integrand)}


@lru_cache(maxsize=1)
def _gradient_evaluators():
    import sympy as sp

    results = derive_symbolically()
    u, d, s = results["symbols"]
    v, w = sp.symbols("v w")
    small_gradient = tuple(
        sp.expand(sp.diff(results["small"], variable).subs({d: u+v, s: u-w}))
        for variable in (u, d, s)
    )
    large_gradient = tuple(sp.diff(results["large"], variable) for variable in (u, d, s))
    return (sp.lambdify((u, v, w), small_gradient, modules="math"),
            sp.lambdify((u, d, s), large_gradient, modules="math"))


def cold_phase_space_gradient(mu_u, mu_d, mu_s):
    """Return (dI0/dmu_u, dI0/dmu_d, dI0/dmu_s); lazily requires SymPy."""
    if not all(isfinite(value) for value in (mu_u, mu_d, mu_s)):
        raise ValueError("Chemical potentials must be finite")
    if not 0 <= mu_s <= mu_u <= mu_d:
        raise ValueError("Require 0 <= mu_s <= mu_u <= mu_d")
    if mu_d == mu_s or mu_u == 0:
        return (0.0, 0.0, 0.0)
    small_gradient, large_gradient = _gradient_evaluators()
    if mu_d-mu_s <= mu_u:
        return small_gradient(mu_u, mu_d-mu_u, mu_u-mu_s)
    return large_gradient(mu_u, mu_d, mu_s)


def verify():
    """Check exact branch matching, stable evaluation and neutral benchmarks."""
    import sympy as sp

    results = derive_symbolically()
    u, d, s = results["symbols"]
    small, large = results["small"], results["large"]
    matching = [sp.factor(sp.diff(large-small, d, order).subs(d, u+s))
                for order in range(7)]
    assert matching == [0, 0, 0, 0, 0, 0, 64*(s+u)**2]
    endpoint = u**4*(210*d**4-112*d**3*u-84*d**2*u**2-32*d*u**3+65*u**4)/315
    assert sp.expand(large.subs(s, 0)-endpoint) == 0

    # Exact rational inputs test both branches, the boundary, and d > 2*u.
    cases = [(sp.Rational(1), sp.Rational(6, 5), sp.Rational(4, 5)),
             (sp.Rational(1), sp.Rational(5, 4), sp.Rational(1, 4)),
             (sp.Rational(1), sp.Rational(5, 4), sp.Rational(1, 8)),
             (sp.Rational(1), sp.Rational(3), sp.Rational(1, 2))]
    for uu, dd, ss in cases:
        expression = small if dd-ss <= uu else large
        exact = float(expression.subs({u: uu, d: dd, s: ss}))
        computed = cold_phase_space(float(uu), float(dd), float(ss))
        assert abs(computed/exact-1) < 3e-13, (uu, dd, ss, computed, exact)

    assert "cold_phase_space_gradient" in globals(), "Exact gradient evaluator is missing"
    for uu, dd, ss in cases + [(sp.Rational(1), sp.Rational(100000001, 100000000),
                               sp.Rational(99999999, 100000000))]:
        expression = small if dd-ss <= uu else large
        computed = cold_phase_space_gradient(float(uu), float(dd), float(ss))
        # Compare at the same binary inputs, avoiding decimal-input rounding noise.
        substitutions = {u: sp.Rational(float(uu)), d: sp.Rational(float(dd)),
                         s: sp.Rational(float(ss))}
        exact = [float(sp.diff(expression, variable).subs(substitutions))
                 for variable in (u, d, s)]
        for actual, expected in zip(computed, exact):
            assert abs(actual/expected-1) < 3e-12, (uu, dd, ss, computed, exact)

    benchmarks = []
    for a in (0.001, 0.146, 0.5, 0.9, 0.99, 1.0):
        dd, ss = (1+a)**(1/3), (1-a)**(1/3)
        ratio = 45*cold_phase_space(1.0, dd, ss)/(128*(dd-ss)**3)
        benchmarks.append((a, ratio))
    assert abs(benchmarks[-1][1]-0.1096583030099788) < 1e-14
    near = 1e-8
    dd, ss = 1+near, 1-near
    near_ratio = 45*cold_phase_space(1.0, dd, ss)/(128*(dd-ss)**3)
    assert abs(near_ratio-1) < 1e-7
    for bad in ((1.0, 0.9, 0.8), (1.0, 1.2, -0.1), (1.0, float("nan"), 0.8)):
        try:
            cold_phase_space(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid inputs accepted: {bad}")
    print("I0, delta <= u:", small)
    print("I0, delta >= u:", large)
    print("Branch correction:", results["correction"])
    print("Boundary derivative differences (orders 0..6):", matching)
    print("Neutral exact / Eq. (23) cold benchmarks:")
    for a, ratio in benchmarks:
        print(f"  a={a:g}: {ratio:.12f}")
    print("All symbolic and numerical checks passed.")


if __name__ == "__main__":
    verify()
