"""Read-only matched-state comparison and profile-derived shape diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
C = 299792458.


def load_samples():
    folder = ROOT / 'new_paper_calculations/26-09-28'
    analytical = np.load(folder / 'T0plus0p1-analytical-comparison.npy', allow_pickle=True).item()
    numerical = np.load(folder / 'inputs/T0plus0p1-uNmax-state-of-art-complete-5km.npy', allow_pickle=True).item()
    sources = {p: np.load(p, allow_pickle=True).item() for p in numerical['profile_source_paths']}
    matched = set(analytical['matched_numerical_grid']['point_keys'].flat)
    rows, profiles = [], {}
    for numeric in numerical['all_successful_records']:
        theta, fraction = numeric['theta_index'], numeric['radial_fraction']
        key = f'{theta}:{fraction:.12f}'
        source = sources[numeric['source_path']]
        if 'numerical_outward_profiles' in source:
            profile = source['numerical_outward_profiles'][numeric['point_index']][.1]
        else:
            profile = source['profiles'][f'theta={theta:02d}|rho={fraction:.8f}']
        profiles[key] = profile
        a = profile['a']
        g = (profile['jK'] - profile['u'] * profile['nK']) / profile['nB']
        # The tail is at a ~ 0; include the exact equilibrium endpoint (0, 0).
        integral = abs(np.trapezoid(np.r_[g, 0.], np.r_[a, 0.]))
        xi_profile = 2 * integral / (g[0] * a[0]) - 1
        row = dict(key=key, theta_index=theta, theta_deg=np.rad2deg(analytical['theta_rad_axis'][theta]),
                   radial_fraction=fraction, nB_0minus_over_n0=numeric['nB_0minus_over_n0'],
                   T_0minus_MeV=numeric['T_0minus_MeV'], numerical_u_m_s=numeric['u_0minus'] * C,
                   numerical_a_0plus=numeric['a_0plus'], numerical_lambda_n=numeric['lambda_n'],
                   numerical_T_inf_MeV=numeric['T_inf_MeV'], xi_profile=xi_profile,
                   profile_nB_relative_span=np.ptp(profile['nB']) / profile['nB'][-1],
                   profile_muB_relative_span=np.ptp(profile['muB']) / profile['muB'][-1],
                   muB_0plus_over_muB_inf=profile['muB'][0] / profile['muB'][-1],
                   profile_node_count=len(a), profile_a_monotonic=bool(np.all(np.diff(a) <= 1e-8)),
                   g_0plus=g[0], g_max_over_g_0plus=max(g) / g[0],
                   matched_plot_grid=key in matched)
        if key in analytical['records']:
            paired = analytical['records'][key]
            np.testing.assert_allclose([paired['point']['nB_0minus_over_n0'], paired['point']['T_0minus_MeV']],
                                       [row['nB_0minus_over_n0'], row['T_0minus_MeV']], rtol=1e-9, atol=1e-9)
            for method in ('analytic', 'semi_analytic'):
                result = paired[method]
                assert result['success']
                ratio = result['u_0minus'] / numeric['u_0minus']
                a_interface = result['a_0plus']
                row[method + '_u_m_s'] = result['u_0minus'] * C
                row[method + '_ratio'] = ratio
                row[method + '_a_0plus'] = a_interface
                row[method + '_lambda_n'] = result['lambda_n']
                row[method + '_xi_fit_frozen'] = (ratio ** 2 * (1 + analytical['xi'] * a_interface) - 1) / a_interface
                row[method + '_with_profile_xi_ratio'] = ratio * np.sqrt(
                    (1 + analytical['xi'] * a_interface) / (1 + xi_profile * a_interface))
                row[method + '_I2_implied'] = (.5 * result['u_0minus'] ** 2 * result['lambda_n'] ** 2
                                               * (1 - a_interface) * (1 + analytical['xi'] * a_interface))
        rows.append(row)
    return analytical, numerical, pd.DataFrame(rows), profiles


def summaries(frame):
    matched = frame[frame['matched_plot_grid']].copy()
    assert len(matched) == 540
    rows = []
    for lower, upper in [(30, 75), (75, 150), (150, 300), (300, 750), (750, 1500), (1500, 3000), (3000, 6000)]:
        subset = matched[matched['numerical_u_m_s'].between(lower, upper, inclusive='left')]
        row = dict(velocity_bin=f'{lower}-{upper}', count=len(subset))
        for column in ['numerical_u_m_s', 'numerical_a_0plus', 'xi_profile', 'analytic_ratio', 'semi_analytic_ratio',
                       'analytic_xi_fit_frozen', 'semi_analytic_xi_fit_frozen', 'profile_muB_relative_span',
                       'analytic_with_profile_xi_ratio', 'semi_analytic_with_profile_xi_ratio']:
            row[column] = float(subset[column].median())
        row['full_xi_outside_physical_range_fraction'] = float((subset['analytic_xi_fit_frozen'] <= -1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def decomposition(frame):
    matched = frame[frame['matched_plot_grid']].copy()
    matched['I2_full_over_semi'] = matched.analytic_I2_implied / matched.semi_analytic_I2_implied
    matched['composition_factor'] = (1-matched.semi_analytic_a_0plus)/(1-matched.analytic_a_0plus)
    matched['shape_denominator_factor'] = (1-.5*matched.semi_analytic_a_0plus)/(1-.5*matched.analytic_a_0plus)
    matched['density_factor'] = (matched.semi_analytic_lambda_n/matched.analytic_lambda_n)**2
    np.testing.assert_allclose(
        (matched.analytic_ratio/matched.semi_analytic_ratio)**2,
        matched.I2_full_over_semi*matched.composition_factor*
        matched.shape_denominator_factor*matched.density_factor, rtol=1e-12)
    return matched


if __name__ == '__main__':
    analytical, numerical, frame, profiles = load_samples()
    frame.to_csv(OUTPUT / 'matched-state-xi.csv', index=False)
    summary = summaries(frame)
    summary.to_csv(OUTPUT / 'velocity-bin-summary.csv', index=False)
    matched = decomposition(frame)
    matched.to_csv(OUTPUT / 'matched-state-decomposition.csv', index=False)
    assert frame.profile_a_monotonic.all()
    corrected_error = abs(matched.semi_analytic_with_profile_xi_ratio-1)
    print('Semi speed error with profile xi (median / 95th / max):',
          corrected_error.quantile([.5, .95, 1.]).to_dict())
    print(summary.to_string(index=False))
    print('Profile count:', len(frame), 'matched plot grid:', int(frame['matched_plot_grid'].sum()))
    print('Profile xi range:', frame['xi_profile'].min(), frame['xi_profile'].max())
    print('Maximum density drift:', frame['profile_nB_relative_span'].max())
    print('Maximum chemical-potential drift:', frame['profile_muB_relative_span'].max())
    print('Semi fit - profile median/max:', (frame['semi_analytic_xi_fit_frozen']-frame['xi_profile']).abs().median(),
          (frame['semi_analytic_xi_fit_frozen']-frame['xi_profile']).abs().max())
