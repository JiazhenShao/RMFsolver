"""Check fitted xi at the numerical eigenvalue, without solving another BVP."""
import sys
from time import perf_counter

from analyze_xi import C, OUTPUT, ROOT, load_samples
import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT))
from RMFsolver import phase_velocity as pv


if __name__ == '__main__':
    analytical, numerical, frame, profiles = load_samples()
    selected = frame[(frame.theta_index == 15) & frame.radial_fraction.isin(
        [.02, .1, .2, .4, .6, .78, .84])].copy()
    selected = pd.concat([selected, frame[(frame.theta_index == 15) &
                                         (frame.radial_fraction > .878)]])
    rows = []
    for _, point in selected.iterrows():
        started = perf_counter()
        profile = profiles[point.key]
        upstream = pv._validated_pnm_state_from_nB(
            point.nB_0minus_over_n0 * .16 * pv.const.MeV_fm**3, point.T_0minus_MeV)
        upstream['T_0minus'] = point.T_0minus_MeV
        speed = point.numerical_u_m_s / C
        endpoint = pv._solve_analytic_inf_endpoint_for_u_0minus(
            speed, upstream, 189.1565957288247,
            initial_guess=(profile['muB'][-1], profile['T_profile'][-1]))
        for method, closure in [('analytic', 'closed_form'), ('semi_analytic', 'numerical_I2')]:
            formula = pv._analytic_velocity_formula_from_endpoint(
                endpoint, upstream, -.5, T_0plus=.1, velocity_closure=closure)
            a = formula['a_0plus']
            ratio = np.sqrt(formula['u_0minus_formula_squared']) / speed
            xi = (ratio**2 * (1 - .5*a) - 1) / a
            check = pv._analytic_velocity_formula_from_endpoint(
                endpoint, upstream, xi, T_0plus=.1, velocity_closure=closure)
            ratio_check = np.sqrt(check['u_0minus_formula_squared']) / speed
            np.testing.assert_allclose(ratio_check, 1., rtol=1e-10)
            row = dict(key=point.key, method=method, numerical_u_m_s=point.numerical_u_m_s,
                       a_0plus=a, xi_fit_at_numerical_endpoint=xi,
                       xi_fit_frozen=point[method+'_xi_fit_frozen'], xi_profile=point.xi_profile,
                       fitted_speed_ratio=ratio_check,
                       profile_xi_speed_ratio=ratio*np.sqrt((1-.5*a)/(1+point.xi_profile*a)),
                       runtime_s=perf_counter()-started)
            rows.append(row)
            print(row, flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT/'endpoint-xi-check.csv', index=False)
