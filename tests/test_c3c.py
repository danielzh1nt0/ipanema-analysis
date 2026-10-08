"""C3c tools: back-projection round trip, near-line pick, raw-frame grab dry run, near-line measurement on a drawn frame."""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from ipanema import lines as LN
import c3c_nearline as NL, c3c_frames as FR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM = {"C": [53.0, 68.0, -5.0], "base_tilt": [0.009, 0.006]}
POSE = [-2.5928, 0.1153, 0.0111, 1258.8]                                    # a real Vallentuna row (C3b c29, t = 4693 s)

def test_backproject_round_trip():
    P = np.array([[10.0, 60.0], [20.0, 64.0], [5.0, 40.0], [30.0, 50.0]])
    uv = LN.project(CAM, POSE, P, 1280, 720); G = NL.backproject(CAM, POSE, uv)
    assert np.allclose(G, P, atol=1e-6)

def test_near_len_and_pick():
    assert FR.near_len_px(CAM, POSE) > 500                                   # this view shows the near touchline
    away = [POSE[0] + np.pi, POSE[1], POSE[2], POSE[3]]
    assert FR.near_len_px(CAM, away) == 0
    rows = [{"t": float(t), "pose": POSE, "why": []} for t in range(0, 400, 5)] + [{"t": 401.0, "pose": POSE, "why": ["lines disagree"]}]
    s = FR.pick(rows, CAM, [[0, 400]], n=5, gap_s=20)
    assert len(s) == 5 and all(5 <= q["t"] <= 395 for q in s) and min(np.diff([q["t"] for q in s])) >= 20

def _grass_with_line(W_paint):
    """a green frame with white paint drawn where a pitch W_paint wide has its near touchline (camera unchanged)"""
    img = np.zeros((720, 1280, 3), np.uint8); img[:] = (40, 140, 40)
    P = np.c_[np.linspace(-5, 40, 400), np.full(400, W_paint)]; q = LN.project(CAM, POSE, P, 1280, 720)
    q = q[np.isfinite(q).all(1)].round().astype(np.int32)
    cv2.polylines(img, [q.reshape(-1, 1, 2)], False, (245, 245, 245), 3)
    return img

def test_measure_finds_the_painted_line():
    r = NL.measure(_grass_with_line(64.0), CAM, POSE); assert r and abs(r["y_med"] - 64.0) < 0.2
    r = NL.measure(_grass_with_line(62.0), CAM, POSE); assert r and abs(r["y_med"] - 62.0) < 0.2

def test_grab_dry_run(tmp_path):
    v = str(tmp_path / "v.mp4"); w = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*"mp4v"), 1, (64, 36))
    for i in range(30): w.write(np.full((36, 64, 3), 8 * i, np.uint8))
    w.release(); d = tmp_path / "f"; d.mkdir()
    json.dump({"src_key": "x/video.mp4", "camera": CAM, "rows": [{"id": "n00", "t": 3.0, "pose": POSE}, {"id": "n01", "t": 20.0, "pose": POSE}]}, open(d / "sample.json", "w"))
    os.environ["VIDEO"] = v
    try: assert FR.grab(str(d)) == 2
    finally: del os.environ["VIDEO"]
    assert cv2.imread(str(d / "n01.jpg")).shape == (720, 1280, 3)

def test_committed_samples_are_trusted_rows():
    for m in ("vall", "sfk"):
        p = f"{ROOT}/results/qa/c3c/frames_{m}/sample.json"
        if os.path.exists(p):
            S = json.load(open(p)); assert len(S["rows"]) >= 10 and all(FR.near_len_px(S["camera"], q["pose"]) >= 500 for q in S["rows"])

def test_fit_prefers_the_painted_pitch_width():
    """paint a 106 x 65 pitch (near touchline sky-blue and wide, like Vallentuna's in shade); with poses re-fitted, the
    65 m hypothesis must explain the near line better than 64 m and the far lines at least as well"""
    import c3c_fit as FIT
    img = np.zeros((720, 1280, 3), np.uint8); img[:] = (40, 140, 40)
    S = LN.projected_segments(CAM, np.array(POSE), 1280, 720, 106.0, 65.0, LN.class_segments(106.0, 65.0))
    for k, segs in S.items():
        for x0, y0, x1, y1 in segs:
            ok, p1, p2 = cv2.clipLine((0, 0, 1280, 720), (int(x0), int(y0)), (int(x1), int(y1)))
            if ok: cv2.line(img, p1, p2, (250, 200, 150) if k == 1 else (245, 245, 245), 8 if k == 1 else 2)
    fr = [FIT.Frame(img)]; po = [np.array(POSE)]
    r64 = FIT.evaluate(fr, po, CAM, W=64.0); r65 = FIT.evaluate(fr, po, CAM, W=65.0)
    assert r65["near"] < r64["near"] and r65["far"] <= r64["far"] + 0.2
    assert NL.measure(img, CAM, POSE)["y_med"] > 64.6
