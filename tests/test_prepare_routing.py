"""N0-21c (21 Oct): run.prepare (the app's per-clip step: calibration source, players source, ball guesses, caches)
had no test. Every model / video step is replaced by a stub; the test checks which path prepare takes and which cache it
reads or writes. Two faults found and fixed:
  - a fixed-camera match still ran the mosaic registration (then ignored it) and kept its tracks under the pano / kp
    cache name, so tracks made with another calibration were loaded with their old metres;
  - the side-by-side 'WASB + YOLO' ball check was built from the RF-DETR (+WASB) guesses the app uses, rescaled as if
    they were WASB peaks, and logged under the wrong name."""
import json, os, pickle
import numpy as np
import pytest

import ipanema.run as R
from ipanema.config import Settings
from ipanema import mosaic, linecal, fixedcam, wasb, ballclicks, ballrf

N = 6


@pytest.fixture
def env(tmp_path, monkeypatch):
    calls = {"mosaic": 0, "track": 0, "reposition": 0, "wasb": 0, "yolo": 0, "clicks": 0, "kpcal": 0}
    S = Settings(root=str(tmp_path / "root"), sports_dir=str(tmp_path / "sports"), work=str(tmp_path / "work"))
    for k in ("IPANEMA_BALL", "IPANEMA_PLAYERS", "IPANEMA_BALL_MERGE", "IPANEMA_BALL_FUSE", "IPANEMA_TRACKER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(R.V, "normalise", lambda src, dst: src)
    monkeypatch.setattr(R.V, "info", lambda v: {"width": 1920, "height": 1080, "fps": 25.0, "n": N})
    monkeypatch.setattr(linecal, "find_rows", lambda root, mid: None)
    monkeypatch.setattr(fixedcam, "find", lambda mid, root=None: None)

    def _mosaic(video, mid, root, log=print):
        calls["mosaic"] += 1
        return {k: np.eye(3) * 2 for k in range(N)}
    monkeypatch.setattr(mosaic, "calibrate_via_mosaic", _mosaic)

    def _kpcal(*a, **k):
        calls["kpcal"] += 1
        return {"coverage": 1.0, "frozen": 0, "H": {i: np.eye(3) for i in range(N)}, "L": 105.0, "W": 68.0}
    monkeypatch.setattr(R.C, "calibrate", _kpcal)
    monkeypatch.setattr(R.C, "_pitch_config", lambda d: (None, 105.0, 68.0), raising=False)
    import ipanema.calibration as _cal
    monkeypatch.setattr(_cal, "_pitch_config", lambda d: (None, 105.0, 68.0))

    class TM:
        def __init__(self, d): pass
        def fit(self, *a, **k): return "old-team-model"
    monkeypatch.setattr(R.T, "TeamModel", TM)
    monkeypatch.setattr(R.T, "silence_progress", lambda: None)

    def _track(video, w, H, tm, conf, **k):
        calls["track"] += 1
        return {i: [[1, "A", np.zeros(2), (100.0, 600.0), None, False]] for i in range(N)}, 25.0
    monkeypatch.setattr(R.TR, "track", _track)

    def _repos(per, H):
        calls["reposition"] += 1
        return {k: [[r[0], r[1], "repos", r[3], r[4], r[5]] for r in v] for k, v in per.items()}
    monkeypatch.setattr(R.TR, "reposition", _repos)

    def _wasb(video, root, cache, log=print, train=True, thr=0.05):
        calls["wasb"] += 1
        return {i: [(500.0, 500.0, 0.9)] for i in range(N)}
    monkeypatch.setattr(wasb, "candidates", _wasb)

    def _yolo(*a, **k):
        calls["yolo"] += 1
        return {i: [(10.0, 10.0, 0.5)] for i in range(N)}
    monkeypatch.setattr(R.BL, "candidates", _yolo)

    def _clicks(*a, **k):
        calls["clicks"] += 1
        return {i: [(700.0, 700.0, 0.6)] for i in range(N)}
    monkeypatch.setattr(ballclicks, "candidates", _clicks)
    cache = os.path.join(S.root, "cache", "testmatch")
    os.makedirs(cache, exist_ok=True)
    return S, calls, cache


def _run(S):
    return R.prepare("/x/video.mp4", "testmatch", S, log=lambda *a: None, debug=False)


def _put_rf(cache, ball=True):
    per = {i: [[7, "B", np.zeros(2), (50.0, 500.0), None, False]] for i in range(N)}
    pickle.dump((per, 25.0), open(f"{cache}/tracks_rfdetr_v1.pkl", "wb"))
    pickle.dump("rf-kits", open(f"{cache}/kits_rfdetr_v1.pkl", "wb"))
    if ball: pickle.dump({i: [(500.0, 502.0, 0.4)] for i in range(N)}, open(f"{cache}/ball_cands_{ballrf.VERSION}.pkl", "wb"))


def test_app_default_rf_players_rf_ball_fused_no_fake_alt(env):
    S, calls, cache = env
    _put_rf(cache)
    pickle.dump({i: [(10.0, 10.0, 0.5)] for i in range(N)}, open(f"{cache}/ball_cands.pkl", "wb"))   # an old YOLO cache lying around
    ctx = _run(S)
    assert ctx["tm"] == "rf-kits" and calls["track"] == 0 and calls["reposition"] == 1
    assert ctx["per"][0][0][2] == "repos"                                        # RF tracks get this calibration's metres
    assert calls["mosaic"] == 1                                                  # follow-cam: mosaic registration is the calibration
    assert calls["wasb"] == (1 if ballrf.FUSE_WASB == "1" else 0)
    # fused on the same ball: (RF 1.5*0.4 + WASB 0.9 + bonus 0.8) / 2, capped at 0.99
    assert ctx["cands"][0][0][2] == pytest.approx(min(0.99, (1.5 * 0.4 + 0.9 + 0.8) / 2))
    assert ctx["cands_alt"] is None                                              # fixed: no 'WASB + YOLO' check built from RF guesses


def test_alt_check_kept_for_wasb_only(env, monkeypatch):
    S, calls, cache = env
    monkeypatch.setenv("IPANEMA_BALL", "wasb")
    pickle.dump({i: [(10.0, 10.0, 0.5)] for i in range(N)}, open(f"{cache}/ball_cands.pkl", "wb"))
    ctx = _run(S)
    assert calls["wasb"] == 1 and calls["yolo"] == 0
    assert ctx["cands"][0] == [(500.0, 500.0, 0.9)]
    alt = ctx["cands_alt"][0]
    assert alt[0][:2] == (500.0, 500.0) and alt[0][2] == pytest.approx(0.95)    # WASB peak 0.9 -> 0.5 + 0.5 * 0.9
    assert any(abs(c - 0.3) < 1e-9 for _, _, c in alt)                           # YOLO guess at 0.6x


def test_old_alt_rule_switch(env, monkeypatch):
    S, calls, cache = env
    _put_rf(cache)
    pickle.dump({i: [(10.0, 10.0, 0.5)] for i in range(N)}, open(f"{cache}/ball_cands.pkl", "wb"))
    monkeypatch.setattr(R, "_ALT_WASB_ONLY", False)
    assert _run(S)["cands_alt"] is not None


def test_old_detector_tracks_then_cache(env):
    S, calls, cache = env
    ctx = _run(S)
    assert ctx["tm"] == "old-team-model" and calls["track"] == 1
    from ipanema.mosaic import CAL_VERSION
    assert os.path.exists(f"{cache}/tracks_pano_{CAL_VERSION}.pkl")
    ctx = _run(S)
    assert calls["track"] == 1                                                   # second run: cached, no new detection
    assert calls["mosaic"] == 1                                                  # mosaic registration cached too


def test_ball_backend_choice(env):
    S, calls, cache = env
    os.makedirs(os.path.join(S.root, "models", "ball"), exist_ok=True)
    open(os.path.join(S.root, "models", "ball", "clicks_latest.pt"), "w").write("x")
    ctx = _run(S)                                                                # no RF guesses -> click model (+ WASB)
    assert calls["clicks"] == 1 and calls["wasb"] == 1
    assert ctx["cands"][0][0][2] == pytest.approx(1.5 * 0.6 / 2)   # click guess alone (WASB 200 px away)
    _put_rf(cache)
    n_clicks = calls["clicks"]
    _run(S)                                                                      # RF guesses present -> RF-DETR wins
    assert calls["clicks"] == n_clicks


def _fixed_spec(tmp):
    p = os.path.join(tmp, "testmatch.json")
    pts = [([0, 0], [100, 300]), ([105, 0], [1800, 300]), ([105, 68], [1900, 1000]), ([0, 68], [20, 1000]),
           ([52.5, 0], [950, 300]), ([52.5, 68], [960, 1000]), ([52.5, 34], [955, 600])]
    json.dump({"pitch": {"length": 105, "width": 68}, "image_size": [1920, 1080],
               "points": [{"pitch_m": a, "px": b} for a, b in pts]}, open(p, "w"))
    return p


def test_fixed_camera_no_mosaic_own_cache(env, monkeypatch, tmp_path):
    S, calls, cache = env
    spec = _fixed_spec(str(tmp_path))
    monkeypatch.setattr(fixedcam, "find", lambda mid, root=None: spec)
    from ipanema.mosaic import CAL_VERSION
    # tracks left by an earlier follow-cam / keypoint run of the same clip, with THAT calibration's metres
    stale = {i: [[3, "A", np.array([99.0, 99.0]), (960.0, 600.0), None, False]] for i in range(N)}
    pickle.dump((stale, 25.0), open(f"{cache}/tracks_pano_{CAL_VERSION}.pkl", "wb"))
    pickle.dump((stale, 25.0), open(f"{cache}/tracks_kp.pkl", "wb"))
    ctx = _run(S)
    assert calls["mosaic"] == 0                                                  # fixed camera: no mosaic registration
    assert ctx["cal"].get("fixed") and ctx["L"] == 105.0
    assert calls["track"] == 0 and calls["reposition"] == 1                      # earlier detections reused ...
    assert ctx["per"][0][0][2] == "repos"                                        # ... with this camera's metres
    assert os.path.exists(f"{cache}/tracks_fixed.pkl")
    _run(S)
    assert calls["reposition"] == 2                                              # cached fixed tracks re-positioned every time


def test_fixed_camera_old_names_switch(env, monkeypatch, tmp_path):
    S, calls, cache = env
    spec = _fixed_spec(str(tmp_path))
    monkeypatch.setattr(fixedcam, "find", lambda mid, root=None: spec)
    monkeypatch.setattr(R, "_FIXED_CACHE", False)
    pickle.dump(({i: [] for i in range(N)}, 25.0), open(f"{cache}/tracks_kp.pkl", "wb"))
    _run(S)
    assert calls["track"] == 0 and calls["reposition"] == 0                      # old: kp tracks loaded as they are
