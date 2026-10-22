"""N0-22b (22 Oct): the follow-cam calibration (mosaic registration) gives EVERY frame a homography - frames where the
registration failed get the edge anchor copied or a straight blend across the gap - and the app then counted every frame
as calibrated: 'calibration: from panorama, 8992/8992 frames registered', calibration_coverage 1.0, frozen 0, and the
full-match QA check 'calibration > 95%' could never fail. Now coverage counts only measured frames (between two anchors
at most ~1 s apart, or within one sample of the first / last); IPANEMA_MOSAIC_COVER=0 = old count."""
import os, json, pickle, tempfile
import cv2, numpy as np, pytest

from ipanema.mosaic import measured_mask, calibrate_via_mosaic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_measured_mask_rules():
    ak = [0, 10, 20, 30, 80, 90]                        # one gap of 50 frames (30 -> 80)
    ok = measured_mask(ak, 100, step=10, max_gap=25)
    assert ok[:31].all()                                # regular sampling: all measured
    assert not ok[31:80].any()                          # inside the long gap: guessed
    assert ok[80:].all()                                # up to one step after the last anchor still counts
    ok = measured_mask([20, 30, 40], 60, step=10, max_gap=25)
    assert not ok[:10].any() and ok[10:51].all() and not ok[51:].any()
    assert measured_mask([0, 20, 40], 41, step=10, max_gap=25).all()   # one missed sample (gap 20) is still fine
    assert not measured_mask([], 5).any()


def _clip(path, grey):
    pano = cv2.imread(f"{ROOT}/results/mosaic_SFKBP1109_c4.jpg"); H0 = pano.shape[0]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    for k in range(300):
        x = int(200 + k * 5); y = max(0, (H0 - 1080) // 2); f = pano[y:y + 1080, x:x + 1920]
        if f.shape[:2] != (1080, 1920): f = cv2.resize(f, (1920, 1080))
        if grey[0] <= k < grey[1]: f = np.full((1080, 1920, 3), 90, np.uint8)     # camera blinded: nothing to register
        vw.write(f)
    vw.release()


def test_real_registration_reports_the_gap():
    p = os.path.join(tempfile.mkdtemp(), "gap.mp4"); _clip(p, (100, 200)); info = {}
    Hs = calibrate_via_mosaic(p, "SFKBP1109_s1200", tempfile.mkdtemp(), code_dir=ROOT, log=lambda *a: None, anchors_out=info)
    assert Hs is not None and len(Hs) == 300            # unchanged: every frame still gets a homography
    assert info["step"] == 10 and not any(100 <= a < 200 for a in info["anchors"])
    assert set(info["direct_keys"]) <= set(info["anchors"]) and info["direct"] + info["chained"] == len(info["anchors"])
    ok = measured_mask(info["anchors"], 300, step=10, max_gap=25)
    assert not ok[101:199].any() and ok[:91].all() and ok[210:].all()


# --- the app's prepare step uses it (stubs, as in test_prepare_routing) ---
from tests.test_prepare_routing import env, _run, N as _N  # noqa: E402,F401
import ipanema.run as R  # noqa: E402
from ipanema import mosaic  # noqa: E402


def _stub(monkeypatch, anchors):
    def _m(video, mid, root, log=print, anchors_out=None, **kw):
        info = anchors_out
        if info is not None: info.update({"step": 1, "anchors": anchors, "direct_keys": anchors, "direct": len(anchors), "chained": 0})
        return {k: np.eye(3) * 2 for k in range(_N)}
    monkeypatch.setattr(mosaic, "calibrate_via_mosaic", _m)


def test_prepare_counts_only_measured(env, monkeypatch):
    S, calls, cache = env
    _stub(monkeypatch, [0, 1, 2])                       # 6 frames at 25 fps, step 1: frames 3-5 lie past the last anchor
    monkeypatch.setattr(R, "_MOSAIC_COVER", True)
    ctx = _run(S)
    assert len(ctx["H"]) == _N
    assert ctx["cal"]["coverage"] == pytest.approx(4 / 6) and ctx["cal"]["frozen"] == 2   # frame 3 is within one step
    from ipanema.mosaic import CAL_VERSION
    assert json.load(open(f"{cache}/calibration_pano_{CAL_VERSION}_anchors.json"))["anchors"] == [0, 1, 2]
    ctx = _run(S)                                       # second run: from the cache + the saved anchors, same answer
    assert ctx["cal"]["coverage"] == pytest.approx(4 / 6)


def test_prepare_old_switch_and_old_cache(env, monkeypatch):
    S, calls, cache = env
    _stub(monkeypatch, [0, 1, 2])
    monkeypatch.setattr(R, "_MOSAIC_COVER", False)
    assert _run(S)["cal"]["coverage"] == 1.0            # old count
    from ipanema.mosaic import CAL_VERSION
    os.remove(f"{cache}/calibration_pano_{CAL_VERSION}_anchors.json")   # a cache made before N0-22b: no anchors known
    monkeypatch.setattr(R, "_MOSAIC_COVER", True)
    ctx = _run(S)
    assert ctx["cal"]["coverage"] == 1.0 and ctx["cal"]["frozen"] == 0


def test_tool_dry_run(tmp_path, monkeypatch):
    """the free-runner tool (tools/n0_22b_mosaic_gaps.py) end to end on two small stand-in clips"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("n0_22b", f"{ROOT}/tools/n0_22b_mosaic_gaps.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
    a = str(tmp_path / "a.mp4"); _clip(a, (100, 200))
    b = str(tmp_path / "b.mp4"); cv2.VideoWriter(b, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080)).release()   # empty: no calibration
    monkeypatch.setenv("LOCAL_CLIP_SFK", a); monkeypatch.setenv("LOCAL_CLIP_AIK", b)
    out = tmp_path / "out"; out.mkdir(); monkeypatch.setattr("sys.argv", ["x", str(out)]); T.main()
    s = json.load(open(out / "summary.json"))
    assert s[0]["frames"] == 300 and s[0]["guessed"] >= 90 and s[0]["gaps"] and s[0]["lines_guessed"]["checked"] >= 9
    assert "error" in s[1]
    assert (out / f"{T.SFK}_sheet.jpg").exists()
