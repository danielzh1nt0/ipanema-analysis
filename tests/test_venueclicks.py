import json, os, numpy as np
from ipanema import venueclicks as VC
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_edsberg_base_recovered_from_clicks_alone():
    c = json.load(open(f"{ROOT}/results/labels/SFKBP1109_points_s1.json"))
    names = [n for n in c if len(c[n].get("pairs", [])) >= 4][:8]
    sol = VC.solve({n: c[n] for n in names}, log=None, C0=[53.0, 70.0, -5.0])
    assert np.linalg.norm(np.array(sol["camera"]["C"]) - [53.34, 67.62, -4.8]) < 1.5       # metres; the known base after 56 frames
    assert np.median(list(sol["click_errors_px"].values())) < 6
