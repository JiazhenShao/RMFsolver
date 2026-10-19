"""Fixed-chemical-potential thermal coefficients for 0 <= s < u < d.

Coefficients multiply the phase-space integral before the common weak prefactor.
This is a derivation/validation helper, not a production rate implementation.
"""

from math import pi


ZETA_3 = 1.2020569031595942854


def phase_space_hessian_trace(mu_u, mu_d, mu_s):
    """Return sum_i d^2 I0/d mu_i^2, keeping four thresholds independent."""
    if not (0 <= mu_s <= mu_u <= mu_d and mu_u > 0):
        raise ValueError("Requires 0 <= mu_s <= mu_u <= mu_d and mu_u > 0")
    x = (mu_d - mu_u) / mu_u
    y = (mu_u - mu_s) / mu_u
    polynomial = (
        -9 * x**6
        + x**5 * (10 - 32 * y)
        + x**4 * (40 - 40 * y**2)
        + x**3 * (40 + 60 * y - 40 * y**3)
        + x**2 * (40 + 40 * y + 60 * y**2 - 40 * y**4)
        + x * (128 - 240 * y + 280 * y**2 - 100 * y**3 + 40 * y**4 - 36 * y**5)
        + y * (128 - 280 * y + 280 * y**2 - 120 * y**3 + 30 * y**4 - 13 * y**5)
    )
    if x + y > 1:
        d = mu_d / mu_u
        s = mu_s / mu_u
        polynomial += (x + y - 1)**4 * (
            9 * d**2 + 4 * d * s + 2 * s**2 + 4 * d + 4 * s + 2
        )
    return (8 / 15) * mu_u**6 * polynomial


def phase_space_t2(mu_u, mu_d, mu_s):
    """Coefficient multiplying T**2 in the forward phase-space integral."""
    return pi**2 / 6 * phase_space_hessian_trace(mu_u, mu_d, mu_s)


def phase_space_t3(mu_u, mu_d, mu_s):
    """Cubic coefficient from coincident incoming/outgoing up thresholds."""
    phase_space_hessian_trace(mu_u, mu_d, mu_s)
    if not mu_s < mu_u < mu_d:
        raise ValueError("Cubic formula requires strict mu_s < mu_u < mu_d")
    return -(64 / 3) * ZETA_3 * mu_u**2 * (mu_d - mu_s) * (
        mu_d**2 + mu_d * mu_s + mu_s**2
    )


def derive_symbolically():
    """Reproduce polynomial boundary integrals using exact rational algebra."""
    import sympy as sp

    u, d, s, m, e = sp.symbols("u d s m e", real=True)
    w = lambda energy, minimum: 32 * minimum**2 * (energy - minimum)**2
    integ = lambda expr, var, low, high: sp.integrate(expr, (var, low, high))
    q = u + s - d
    upper = integ(w(u + d, m) * (d - s), m, 0, s) + integ(
        w(u + d, m) * (d - m), m, s, u
    )
    lower_positive = integ(w(u + s, m) * (u - q), m, 0, q) + integ(
        w(u + s, m) * (u - m), m, q, s
    )
    lower_nonpositive = integ(w(u + s, m) * (u - m), m, 0, s)
    incoming_up_edge = integ(w(e, u) * (e - 2 * u), e, 2 * u, u + d)
    incoming_down_positive = integ(
        w(d + m, m) * (d + m - u - s), m, q, s
    ) + integ(w(d + m, m) * (d - u), m, s, u)
    incoming_down_nonpositive = integ(
        w(d + m, m) * (d + m - u - s), m, 0, s
    ) + integ(w(d + m, m) * (d - u), m, s, u)
    outgoing_up_edge = integ(w(u + m, m) * (u - m), m, s, u)
    outgoing_strange_edge = integ(w(e, s) * (u - s), e, u + s, d + s) + integ(
        w(e, s) * (u + d - e), e, d + s, u + d
    )
    common = 2 * upper + incoming_up_edge + outgoing_up_edge - outgoing_strange_edge
    j_positive = sp.factor(common + 2 * lower_positive - incoming_down_positive)
    j_nonpositive = sp.factor(common + 2 * lower_nonpositive - incoming_down_nonpositive)
    correction = sp.factor(j_positive - j_nonpositive)
    expected = -sp.Rational(8, 15) * (d - s - u)**4 * (
        9 * d**2 + 4 * d * s + 2 * s**2 + 4 * d * u + 4 * s * u + 2 * u**2
    )
    assert sp.expand(correction - expected) == 0
    x, y = sp.symbols("x y")
    normalized = sp.Poly(
        sp.expand(sp.Rational(15, 8) * j_positive.subs({u: 1, d: 1 + x, s: 1 - y})),
        x,
    )
    print("J for q <= 0:", j_nonpositive)
    print("J(q > 0) - J(q <= 0):", correction)
    print("15 J(q > 0)/(8 u^6), x=(d-u)/u, y=(u-s)/u:", normalized.as_expr())
    eps = sp.symbols("eps")
    print("Equal-flavor limit:", sp.series(j_positive.subs({d: u + eps, s: u - eps}), eps, 0, 2))
    print("Strange-zero limit:", sp.factor(j_nonpositive.subs(s, 0)))
    return j_positive, j_nonpositive


if __name__ == "__main__":
    derive_symbolically()
    for asymmetry in (0.001, 0.146, 0.5, 0.9, 0.99, 1.0):
        u = 1.0
        d = (1 + asymmetry)**(1 / 3)
        s = (1 - asymmetry)**(1 / 3)
        print(asymmetry, phase_space_hessian_trace(u, d, s), phase_space_t2(u, d, s), phase_space_t3(u, d, s))
