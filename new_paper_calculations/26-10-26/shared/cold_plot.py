"""Read-only contour geometry helpers; plot styling lives in notebook cells."""
import numpy as np


def segments(payload, method, level):
    """Only connect adjacent resolved rays. Missing crossings split the line."""
    result, current = [], []
    for theta, row in sorted(payload['methods'][method].items()):
        contour = row['contours'][level]
        if contour['status'] == 'resolved':
            point = contour['point']
            current.append((point['nB_0minus_over_n0'], point['T_0minus_MeV']))
        else:
            if len(current) >= 2:
                result.append(np.array(current))
            current = []
    if len(current) >= 2:
        result.append(np.array(current))
    return result


def boundary_segments(payload, method, name):
    result, current = [], []
    for theta, row in sorted(payload['methods'][method].items()):
        boundary = row['boundaries'].get(name)
        if boundary and boundary['resolved']:
            p = boundary['point']
            current.append((p['nB_0minus_over_n0'],p['T_0minus_MeV']))
        else:
            if len(current) >= 2:
                result.append(np.array(current))
            current = []
    if len(current) >= 2:
        result.append(np.array(current))
    return result


def excluded_polygons(payload, method, x_limits, y_limits):
    """Individual model shading, built only from resolved adjacent boundaries."""
    for curve in boundary_segments(payload, method, 'stable_matter'):
        yield np.vstack(([x_limits[0],y_limits[0]],curve))
    for curve in boundary_segments(payload, method, 'composition_ceiling'):
        outer=[]
        for density,temperature in curve[::-1]:
            theta=np.arctan2(temperature/120,(density-1)/4.5)
            rho=min((x_limits[1]-1)/(4.5*max(np.cos(theta),1e-16)),
                    y_limits[1]/(120*max(np.sin(theta),1e-16)))
            outer.append((1+4.5*rho*np.cos(theta),120*rho*np.sin(theta)))
        yield np.vstack((curve,outer))


def label_point(curve, target_temperature, x_limits, y_limits):
    index = int(np.argmin(abs(curve[:,1]-target_temperature)))
    index = min(max(index,1),len(curve)-2) if len(curve)>2 else 0
    before, after = curve[max(0,index-1)], curve[min(len(curve)-1,index+1)]
    delta = (after-before)/np.array([x_limits[1]-x_limits[0],y_limits[1]-y_limits[0]])
    angle = np.degrees(np.arctan2(delta[1],delta[0]))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return curve[index],angle
