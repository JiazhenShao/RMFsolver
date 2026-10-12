"""Cheap contract tests for the optional isothermal layer-resolved seed."""

import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from RMFsolver import phase_velocity as pv


STATE = {
    "lambda_inf": 0.1, "jK_0plus": 1.0, "nK_0plus": 2.0,
    "u_0plus": 1.0 / 12.0, "D_K_0plus": 1.0, "x_end": -np.log(0.01) / 0.1,
    "nB_inf": 10.0, "nB_0plus": 12.0, "D_K_inf": 2.0,
    "jB": 1.0, "tail_coefficient": 0.3, "nK_inf": 0.0,
}


def test_layer_seed_matches_physical_core_profile_and_endpoint_flux():
    s, nK, jK = pv._isothermal_layer_resolved_initial_guess(STATE, 0.99, 8, 1.0)
    slope = (STATE["jK_0plus"] / STATE["nK_0plus"] - STATE["u_0plus"]) / STATE["D_K_0plus"]
    x_core = 0.5 / (slope - STATE["lambda_inf"])
    x = np.r_[0.0, np.geomspace(x_core * 1e-3, STATE["x_end"], 7)]
    fraction = np.exp(-STATE["lambda_inf"] * x) / np.sqrt(1.0 + x / x_core)
    nB = STATE["nB_inf"] + (STATE["nB_0plus"] - STATE["nB_inf"]) * fraction**2
    D = STATE["D_K_inf"] + (STATE["D_K_0plus"] - STATE["D_K_inf"]) * fraction**2
    expected_jK = nK * (STATE["jB"] / nB + D * (STATE["lambda_inf"] + 0.5 / (x_core + x)))
    np.testing.assert_allclose(s, -np.expm1(-STATE["lambda_inf"] * x))
    np.testing.assert_allclose(nK, STATE["nK_0plus"] * fraction)
    np.testing.assert_allclose(jK[1:-1], expected_jK[1:-1])
    assert np.all(np.diff(s) > 0)
    assert s[0] == 0.0
    assert jK[0] == STATE["jK_0plus"]
    assert jK[-1] == STATE["tail_coefficient"] * nK[-1]


def test_layer_seed_rejects_unsupported_or_degenerate_inputs():
    for change in ({"nK_inf": 1e-6}, {"lambda_inf": np.nan},
                   {"D_K_0plus": -1.0}, {"x_end": 1e-10},
                   {"nK_0plus": 0.0},
                   {"u_0plus": 0.0, "jK_0plus": 0.2}):
        try:
            pv._isothermal_layer_resolved_initial_guess(dict(STATE, **change), 0.99, 8, 1.0)
        except ValueError:
            continue
        raise AssertionError(f"invalid layer seed accepted: {change}")


def test_public_mode_is_last_optional_argument_and_invalid_mode_rejected_early():
    parameters = list(inspect.signature(pv.solve_front_isothermal).parameters.values())
    assert parameters[-1].name == "initial_guess_mode"
    assert parameters[-1].default == "default"
    try:
        pv.solve_front_isothermal(10.0, 1.0, 180.0, a_0plus=0.5, initial_guess_mode="wrong")
    except ValueError as exc:
        assert "initial_guess_mode" in str(exc)
    else:
        raise AssertionError("invalid mode accepted")


def _run_mocked(mode="default", continuation=None, ms=0.0, nK_inf=0.0,
                bvp_raises=False):
    captured = {}

    def thermo(muB, muK, bag, temperature, jB, **kwargs):
        if muK == 0.0:
            return {"nK": nK_inf, "nB": 10.0, "u": jB / 10.0}
        return {"nK": 2.0, "nB": 12.0, "u": jB / 12.0}

    def local(nK, temperature, pi, jB, bag, **kwargs):
        return {"nK": nK, "nB": 10.0, "u": jB / 10.0, "muB": 1000.0, "muK": 0.1}

    def fake_bvp(fun, bc, x, y, **kwargs):
        captured["s"] = np.array(x)
        captured["y"] = np.array(y)
        if bvp_raises:
            raise RuntimeError("mock BVP exception")
        return SimpleNamespace(x=x, y=y, p=kwargs["p"], success=False,
                               status=1, message="mock failure", niter=1,
                               rms_residuals=np.array([0.2, 0.4]))

    with patch.multiple(pv,
                        _isothermal_upstream_nuclear_state=lambda *a: {"P_0minus": 1.0, "h_0minus": 1.0, "a_0minus": 1.0, "e_0minus": 1.0, "nK_0minus": 1.0, "proton_fraction_0minus": 0.0},
                        _solve_muB_inf_at_muK0_for_given_Pi=lambda *a, **kw: 1000.0,
                        _solve_interface_0plus_from_local_a_and_Pi=lambda *a, **kw: (1000.0, 0.1),
                        _quark_thermo_state=thermo,
                        _microphysics_from_quark_state_energy=lambda *a: {"invD": 1.0},
                        _exact_kaon_transport_rate=lambda *a, **kw: {"Gamma_K": 1e-4},
                        _solve_local_quark_state_from_nK_T_and_Pi=local,
                        solve_bvp=fake_bvp):
        result = pv.solve_front_isothermal(10.0, 10.0, 180.0, a_0plus=0.5,
                                           jB_guess=1.0, n_mesh=8, max_nodes=100,
                                           tail_eps=0.01, ms=ms,
                                           continuation_guess=continuation,
                                           initial_guess_mode=mode)
    return result, captured


def test_default_mode_preserves_linear_mesh_and_old_seed():
    result, captured = _run_mocked()
    np.testing.assert_allclose(captured["s"], np.linspace(0, 0.99, 8))
    expected_nK = 2.0 * np.maximum(1.0 - captured["s"], 0.01)
    np.testing.assert_allclose(captured["y"][0], expected_nK / result["bvp_nK_scale"])
    assert result["initial_guess_mode_requested"] == "default"
    assert result["initial_guess_mode_used"] == "default"
    assert result["initial_guess_fallback_reason"] is None
    assert result["bvp_max_rms_residual"] == 0.4


def test_layer_mode_uses_stretched_mesh_without_changing_failed_status():
    result, captured = _run_mocked("layer_resolved")
    assert not result["success"]
    assert result["initial_guess_mode_used"] == "layer_resolved"
    assert captured["s"][0] == 0
    assert np.all(np.diff(captured["s"]) > 0)
    assert not np.allclose(captured["s"], np.linspace(0, 0.99, 8))
    assert result["bvp_max_rms_residual"] == 0.4


def test_layer_mode_falls_back_with_reason():
    for kwargs in ({"ms": 1.0}, {"nK_inf": 0.1}):
        result, captured = _run_mocked("layer_resolved", **kwargs)
        assert result["initial_guess_mode_used"] == "default"
        assert result["initial_guess_fallback_reason"]
        np.testing.assert_allclose(captured["s"], np.linspace(0, 0.99, 8))


def test_continuation_takes_precedence_over_layer_mode():
    continuation = {"jB": 1.0, "s": [0, 0.31, 0.5], "nK": [2.0, 1.5, 1.0],
                    "jK": [1.0, 0.75, 0.5], "nK_inf": 0.0, "jK_inf": 0.0}
    result, captured = _run_mocked("layer_resolved", continuation=continuation)
    assert result["initial_guess_mode_used"] == "continuation"
    assert result["initial_guess_fallback_reason"] is None
    assert len(captured["s"]) > 8


def test_solver_exception_reports_mode_and_unavailable_rms():
    result, _ = _run_mocked("layer_resolved", bvp_raises=True)
    assert result["success"] is False
    assert result["initial_guess_mode_requested"] == "layer_resolved"
    assert result["initial_guess_mode_used"] == "layer_resolved"
    assert np.isnan(result["bvp_max_rms_residual"])


class IsothermalLayerInitializerTests(unittest.TestCase):
    def test_contracts(self):
        for function in (
            test_layer_seed_matches_physical_core_profile_and_endpoint_flux,
            test_layer_seed_rejects_unsupported_or_degenerate_inputs,
            test_public_mode_is_last_optional_argument_and_invalid_mode_rejected_early,
            test_default_mode_preserves_linear_mesh_and_old_seed,
            test_layer_mode_uses_stretched_mesh_without_changing_failed_status,
            test_layer_mode_falls_back_with_reason,
            test_continuation_takes_precedence_over_layer_mode,
            test_solver_exception_reports_mode_and_unavailable_rms,
        ):
            with self.subTest(case=function.__name__):
                function()
