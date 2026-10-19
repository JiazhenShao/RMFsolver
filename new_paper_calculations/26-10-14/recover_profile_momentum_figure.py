"""Recover the archived figure's vector coordinates, not a new numerical profile.

Run with /usr/local/bin/python3 (PyMuPDF is installed there).
The notebook itself needs only the resulting JSON, NumPy, and Matplotlib.
"""

import hashlib
import json
from math import isclose
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "jargon/entropy_tqguess_uagrad_pi_qm_contour.pdf"
DESTINATION = ROOT / "local_analysis_results/26-10-14/constant-momentum-profile-data.json"


def points_close(first, second, tolerance=1.e-4):
    return len(first) == len(second) and all(abs(a-b) <= tolerance for a, b in zip(first, second))


def recover():
    document = pymupdf.open(SOURCE)
    assert len(document) == 1
    page = document[0]
    drawings = page.get_drawings()
    frame = next(d["rect"] for d in drawings
                 if d["items"][0][0] == "re"
                 and 0.5 * page.rect.get_area() < d["rect"].get_area() < 0.9 * page.rect.get_area())

    # The original source fixes the y range; the six vertical grid lines are
    # its labeled 1120,...,1220 MeV ticks. Recover the affine PDF/data map.
    x_tick_pdf = sorted(d["rect"].x0 for d in drawings
                        if d.get("color") is not None
                        and points_close(d["color"], [0.69012, 0.69024, 0.69010])
                        and abs(d["rect"].height - frame.height) < 1.e-3
                        and d["rect"].width < 1.e-3)
    assert len(x_tick_pdf) == 6
    x_tick_data = list(range(1120, 1221, 20))
    x_slope = (x_tick_data[-1]-x_tick_data[0]) / (x_tick_pdf[-1]-x_tick_pdf[0])
    x_intercept = x_tick_data[0] - x_slope*x_tick_pdf[0]
    assert max(abs(x_slope*x+x_intercept-y) for x, y in zip(x_tick_pdf, x_tick_data)) < 5.e-6

    def to_data(points):
        return [[x_slope*x + x_intercept, 120.-(y-frame.y0)*160./frame.height] for x, y in points]

    def line_vertices(drawing):
        assert all(item[0] == "l" for item in drawing["items"])
        starts = [item[1] for item in drawing["items"]]
        ends = [item[2] for item in drawing["items"]]
        assert all(points_close(first, second) for first, second in zip(starts[1:], ends[:-1]))
        return to_data([starts[0], *ends])

    def center(drawing):
        rect = drawing["rect"]
        return to_data([[0.5 * (rect.x0 + rect.x1), 0.5 * (rect.y0 + rect.y1)]])[0]

    contours = [d for d in drawings if d["type"] == "s"
                and abs(d["rect"].height - frame.height) < 1.e-3
                and d.get("width") is not None
                and (isclose(d["width"], 2.6, abs_tol=1.e-4) or isclose(d["width"], 1.45, abs_tol=1.e-4))]
    assert len(contours) == 6
    intermediate = [d for d in contours if isclose(d["width"], 1.45, abs_tol=1.e-4)]
    blue = next(d for d in contours if points_close(d["color"], [31/255, 119/255, 180/255]))
    green = next(d for d in contours if points_close(d["color"], [44/255, 160/255, 44/255]))
    samples = [next(d for d in drawings if d["type"] == "fs"
                    and points_close(d.get("fill") or [-1]*3, curve["color"])
                    and d["rect"].width < 10 and d["rect"].height < 10)
               for curve in intermediate]
    profile = next(d for d in drawings if d["type"] == "s"
                   and d.get("width") is not None and isclose(d["width"], 1.6, abs_tol=1.e-4)
                   and d["rect"].height > 100.)

    def endpoint(color):
        matches = [d for d in drawings if d["type"] == "fs"
                   and points_close(d.get("fill") or [-1]*3, color)
                   and d["rect"].width < 15 and d["rect"].height < 15]
        assert len(matches) == 1
        return center(matches[0])

    point_inf = endpoint(green["color"])
    point_0minus = endpoint([1., 215/255, 0.])
    assert abs(point_inf[1]) < 1.e-5
    point_inf[1] = 0.
    # The star's bounding box is asymmetric; the original source fixes muK=0.
    point_0minus[1] = 0.
    result = {
        "provenance": {
            "original_notebook": "26-04-09.ipynb", "original_cell_index": 9,
            "source_pdf": str(SOURCE.relative_to(ROOT)),
            "source_pdf_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "recovery_method": "original PDF vector paths; no raster digitization or new BVP",
            "limitations": "Historical EOS/solver result; raw matching profile was not located. Temperatures retain the precision printed in the original legend. Coordinates have PDF rendering precision, not solver precision.",
        },
        "calibration": {"pdf_x_ticks": x_tick_pdf, "data_x_ticks": x_tick_data,
                        "pdf_frame": list(frame), "data_y_limits": [-40., 120.],
                        "x_slope": x_slope, "x_intercept": x_intercept},
        "nB_0minus_over_n0": 3., "T_0plus_MeV": 0.10, "T_inf_MeV": 17.34,
        "point_0plus": endpoint(blue["color"]),
        "point_inf": point_inf, "point_0minus": point_0minus,
        "sample_x_nm": [200., 400., 600., 800.],
        "sample_points": [center(d) for d in samples],
        "profile_path": line_vertices(profile),
        "contour_0plus": line_vertices(blue), "contour_inf": line_vertices(green),
        "contours_sample": [line_vertices(d) for d in intermediate],
    }
    assert all(points_close(first, second, tolerance=2.e-5)
               for first, second in zip(result["profile_path"], result["sample_points"]))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Recovered six contours, four sampled points, three endpoints: {DESTINATION}")
    print("Endpoint muB [MeV]:", result["point_0plus"][0], result["point_inf"][0])
    print("Relative endpoint change [%]:", 100*(point_inf[0]/result["point_0plus"][0]-1))


if __name__ == "__main__":
    recover()
