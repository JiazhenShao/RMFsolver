"""Selection of saved contour geometry; deliberately no EOS/solver imports."""
import copy
import numpy as np


def merge_analytical_extension(baseline, extension):
    """Add new saved levels without replacing original targets or source identity."""
    if (not np.array_equal(baseline['temperatures'], extension['temperatures'])
            or baseline['domain_fingerprint'] != extension['domain_fingerprint']
            or baseline['identity']['physics'] != extension['physics']):
        raise ValueError('Incompatible analytical extension: axes, domain, or physics')
    if set(baseline['targets']).intersection(extension['targets']):
        raise ValueError('Analytical extension must not replace baseline records')
    merged = copy.deepcopy(baseline)
    merged['targets'].update(copy.deepcopy(extension['targets']))
    merged['levels'] = sorted(set(map(float, baseline['levels'])) | set(map(float, extension['levels'])))
    merged['extension_provenance'] = copy.deepcopy(extension.get('identity', {}))
    # Original flags refer only to the baseline's scheduled targets.
    merged['work_complete'] = all(r.get('status') != 'pending' for r in merged['targets'].values())
    merged['contours_complete'] = all(r.get('status') in ('success', 'outside_window')
                                      for r in merged['targets'].values())
    return merged


def accepted_target(row, speed_rtol=0.005, require_validated=True):
    if not row or row.get('status') != 'success':
        return None
    root = row.get('root') or {}
    result = root.get('result') or {}
    if ((require_validated and not result.get('validated'))
            or not result.get('success', True)):
        return None
    if not require_validated and (not result.get('success', False)
                                  or ('validated' in result and not result['validated'])):
        return None
    try:
        density, temperature = float(root['nB_0minus_over_n0']), float(root['T_0minus'])
        velocity, level = float(root['velocity_m_s']), float(row['level'])
        requested_temperature = float(row['T_0minus'])
    except (KeyError, TypeError, ValueError):
        return None
    if (not np.all(np.isfinite([density, temperature, velocity, level]))
            or not 1. <= density <= 5.5 or not .01 <= temperature <= 120.
            or temperature != requested_temperature or level <= 0.
            or abs(velocity / level - 1.) > speed_rtol):
        return None
    return np.array([density, temperature])


def analytical_segments(payload, level):
    """Split at every missing/unaccepted scheduled temperature; no extrapolation."""
    temperatures = sorted(set(map(float, payload['temperatures'])))
    rows = {temperature: [] for temperature in temperatures}
    for row in payload['targets'].values():
        if row.get('kind') == 'base' and row['level'] == level and row['T_0minus'] in rows:
            rows[row['T_0minus']].append(row)
    segments, current = [], []
    for matches in rows.values():
        point = accepted_target(matches[0]) if len(matches) == 1 else None
        if point is None:
            if current:
                segments.append(np.asarray(current))
                current = []
        else:
            current.append(point)
    if current:
        segments.append(np.asarray(current))
    return segments


def cold_numerical_points(warm, payload, cold_endpoints, level, split_temperature=20.):
    """Raw accepted roots only. Never turn resampled archive curves into dots."""
    points = []
    if level in warm['levels_m_s']:
        index = list(warm['levels_m_s']).index(level)
        points = list(np.asarray(warm[f'roots_{index}'], dtype=float).reshape(-1, 2))
    for row in payload.get('targets', {}).values():
        if row.get('level') == level:
            point = accepted_target(row)
            if point is not None:
                points.append(point)
    for ray in cold_endpoints.get('rays', {}).values():
        if ray.get('status') != 'success' or ray.get('level') != level:
            continue
        for root in ray.get('roots', []):
            row = dict(status='success', level=level, T_0minus=root['T_0minus'], root=root)
            # This older schema records acceptance in ray.status and
            # result.success, without a separate result.validated field.
            point = accepted_target(row, require_validated=False)
            if point is not None:
                points.append(point)
    unique = {}
    for point in points:
        if (np.all(np.isfinite(point)) and 1. <= point[0] <= 5.5
                and .01 <= point[1] < split_temperature):
            # Same archived solve may appear in more than one checkpoint.
            unique.setdefault(tuple(np.round(point, 10)), point)
    return np.asarray(sorted(unique.values(), key=lambda point: point[1])).reshape(-1, 2)


def warm_curve(curve, split_temperature=20.):
    """Clip existing display geometry, including its intersection with T=20."""
    curve = np.asarray(curve, dtype=float).reshape(-1, 2)
    if len(curve) == 0:
        return curve
    if not np.all(np.isfinite(curve)) or np.any(np.diff(curve[:, 1]) < 0):
        raise ValueError('Saved warm curve must be finite and temperature-ordered')
    warm = curve[curve[:, 1] >= split_temperature].copy()
    if len(warm) and curve[0, 1] < split_temperature < warm[0, 1]:
        crossing = [np.interp(split_temperature, curve[:, 1], curve[:, 0]), split_temperature]
        warm = np.vstack((crossing, warm))
    return warm


def shading_boundaries(geometry, payload, y_limits, x_limits):
    """Use saved physical boundaries only, extending excluded fill to plot edges."""
    result = {}
    for side, field, edge in [('inner', 'lower', x_limits[0]), ('outer', 'upper', x_limits[1])]:
        vertices = np.asarray(geometry[f'{side}_boundary'], dtype=float)
        vertices = vertices[np.argsort(vertices[:, 1])]
        minimum, maximum = vertices[0, 1], vertices[-1, 1]
        for row in payload.get('boundaries', {}).values():
            if row.get('status') != 'success':
                continue
            temperature, density = float(row['T_0minus']), float(row[field])
            if not minimum <= temperature <= maximum or not np.isfinite(density):
                continue
            vertices = vertices[~np.isclose(vertices[:, 1], temperature, rtol=1e-12, atol=1e-12)]
            vertices = np.vstack((vertices, [density, temperature]))
        vertices = vertices[np.argsort(vertices[:, 1])]
        temperatures = np.unique(np.r_[y_limits, vertices[:, 1]])
        temperatures = temperatures[(temperatures >= y_limits[0]) & (temperatures <= y_limits[1])]
        densities = np.clip(np.interp(temperatures, vertices[:, 1], vertices[:, 0]), *x_limits)
        # The outer curve's left edge is the endpoint of the allowed wedge.
        if side == 'outer':
            densities[temperatures > maximum] = x_limits[0]
        result[side] = (temperatures, densities, edge)
    return result
