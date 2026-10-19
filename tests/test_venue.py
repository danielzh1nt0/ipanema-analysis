"""19 Oct (worker, N0): tests for ipanema/venue.py (new-venue camera base from the painted lines alone; modal_app's
venue_base calls solve_base -> heldout_check -> strip; was untested).

Two kinds of test:
- real lines: masks drawn from the SFK-BP (Edsberg) camera with lines.render_mask, so the true base is known; the true
  base must explain them better than a base 3 m / 2 m off, and the held-out check must prefer it.
- fake fit_pose (fast): the search + refine logic finds a known base, and the cases where nothing can be placed (no
  frame fits under the reference base, no masks at all) end in 'reference kept' instead of a crash."""
import json, os
import numpy as np
import pytest

from ipanema import venue as V, lines as LN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = f"{ROOT}/calibration/panorama/SFKBP1109_lines_solution.json"


def test_candidate_bases_grid():
    c = np.array(V.candidate_bases())
    assert c.shape == (5 * 5 * 4, 3)
    assert c[:, 0].min() == pytest.approx(47) and c[:, 0].max() == pytest.approx(59)      # around the halfway line
    assert (c[:, 1] > 64).all() and c[:, 1].max() == pytest.approx(71.5)                  # behind the near touchline
    assert (c[:, 2] < 0).all() and c[:, 2].min() == pytest.approx(-6.5)                   # up (z down-positive)
    assert len({tuple(r) for r in c.tolist()}) == len(c)


@pytest.mark.skipif(not os.path.exists(SOL), reason="SFK-BP line solution not in the repo")
def test_true_base_explains_its_own_lines_best():
    s = json.load(open(SOL)); cam = s["camera"]; poses = [f["pose"] for f in s["frames"][:6:2]]
    masks = [LN.render_mask(cam, p, 640, 360) for p in poses]
    wrong = list(np.array(cam["C"]) + [3.0, 2.0, 0.0])
    c_true, p_true, n_true = V._cost_of_base(masks, cam["C"], tuple(cam["base_tilt"]), init=poses)
    c_wrong, _, n_wrong = V._cost_of_base(masks, wrong, tuple(cam["base_tilt"]), init=poses)
    assert n_true == n_wrong == 3 and c_true < c_wrong
    for p, q in zip(p_true, poses):                                                          # poses come back where drawn
        assert LN.pose_error(cam, np.asarray(p), np.asarray(q))["median_px"] < 10
    # init with a missing pose: that frame is skipped, not cold-fitted
    _, pp, n = V._cost_of_base(masks, cam["C"], tuple(cam["base_tilt"]), init=[poses[0], None, poses[2]])
    assert n == 2 and pp[1] is None


@pytest.mark.skipif(not os.path.exists(SOL), reason="SFK-BP line solution not in the repo")
def test_heldout_prefers_the_true_base():
    s = json.load(open(SOL)); cam = s["camera"]
    masks = [LN.render_mask(cam, f["pose"], 640, 360) for f in s["frames"][:4]]              # held-out half = frames 1, 3
    wrong = {"C": list(np.array(cam["C"]) + [4.0, 3.0, 1.0]), "base_tilt": cam["base_tilt"]}
    ho = V.heldout_check(masks, cam, wrong, log=None)
    assert ho["new"]["frames"] == 2 and ho["new"]["cost"] < ho["reference"]["cost"]


TRUE_C = np.array([54.0, 68.5, -5.3])


def _fake_fit(mask, camera, keep=8, init=None, **kw):
    """a stand-in for lines.fit_pose: cost grows with the distance from TRUE_C; mask None = no lines (no pose)"""
    if mask is None: return None, {"why": "no lines predicted"}
    d = np.asarray(camera["C"], float) - TRUE_C
    return [0.0, 0.1, 0.0, 1500.0], {"cost": 0.5 + float(d @ d) + 10 * float(np.abs(camera.get("base_tilt", [0, 0])).sum())}


def test_solve_base_finds_the_base(monkeypatch):
    monkeypatch.setattr(V.LN, "fit_pose", _fake_fit)
    ref = {"C": [53.0, 66.0, -4.5], "base_tilt": [0.0, 0.0]}
    cam, poses, rep = V.solve_base([np.zeros(1)] * 6, ref, log=None)
    assert rep["coarse_best"] == pytest.approx([53.0, 68.5, -5.5])      # nearest grid point
    assert np.linalg.norm(np.asarray(cam["C"]) - TRUE_C) < 0.5          # the refine moves towards the truth
    assert rep["refined_cost"] <= rep["coarse_cost"] + 1e-9
    assert rep["frames_used"] == rep["of"] == 6 and all(p is not None for p in poses)
    assert "error" not in rep


def test_solve_base_nothing_placed_keeps_reference(monkeypatch):
    """old code: TypeError (best base stays None) - modal's venue_base crashed after the line model had run"""
    monkeypatch.setattr(V.LN, "fit_pose", _fake_fit)
    ref = {"C": [53.0, 66.0, -4.5], "base_tilt": [0.01, 0.0]}
    cam, poses, rep = V.solve_base([None] * 5, ref, log=None)
    assert cam == {"C": ref["C"], "base_tilt": ref["base_tilt"]} and poses == [None] * 5
    assert rep["frames_used"] == 0 and rep["of"] == 5 and "error" in rep
    ho = V.heldout_check([None] * 5, cam, ref, log=None)
    assert not ho["new"]["cost"] < ho["reference"]["cost"]             # venue_base's test -> REJECTED, reference kept
    json.dumps(rep)


def test_solve_base_no_masks(monkeypatch):
    monkeypatch.setattr(V.LN, "fit_pose", _fake_fit)
    ref = {"C": [53.0, 66.0, -4.5], "base_tilt": [0.0, 0.0]}
    cam, poses, rep = V.solve_base([], ref, log=None)
    assert cam["C"] == ref["C"] and poses == [] and rep["frames_used"] == 0
    sheet = V.strip([], [], cam, poses)                                  # old code: ValueError in np.vstack
    assert sheet.ndim == 3 and sheet.shape[1] == 1280


def test_strip_sheet(tmp_path):
    s = json.load(open(SOL)) if os.path.exists(SOL) else None
    cam = s["camera"] if s else {"C": [53.3, 67.2, -4.7], "base_tilt": [0.0, 0.0]}
    pose = s["frames"][0]["pose"] if s else [-2.27, 0.087, 0.0, 1510.0]
    frames = [np.full((720, 1280, 3), 60, np.uint8)] * 3
    masks = [LN.render_mask(cam, pose, 640, 360)] * 3
    out = tmp_path / "strip.jpg"
    sheet = V.strip(frames, masks, cam, [pose, None, pose], out_path=str(out))
    assert sheet.shape == (720, 1280, 3) and out.exists()               # 3 tiles -> 2 rows of 2, last one padded
    assert sheet[360:, 640:].max() == 0                                  # the padding tile is black
