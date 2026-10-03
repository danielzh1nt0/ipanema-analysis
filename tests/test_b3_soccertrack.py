"""B3: SoccerTrack v2 ball labels in pixels. Synthetic stand-in: a known fisheye camera, keypoints made by projecting
pitch points, a tracker XML (with 132877's y conventions: players flipped, ball not) and a video with a white ball drawn
where the camera sees it. The pipeline must recover the camera from the keypoints, honour the flips and the frame
offset, keep only ground balls, and land the label on the drawn ball."""
import json, os, subprocess, sys
import numpy as np, cv2, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import soccertrack as ST

W, H = 1600, 640


def camera():
    K = np.array([[520.0, 0, W / 2], [0, 520.0, H / 2 - 40], [0, 0, 1]]); D = np.array([[0.05], [-0.01], [0.0], [0.0]])
    C = np.array([52.5, -22.0, -14.0]); tgt = np.array([52.5, 34.0, 0.0])              # z up is negative (camera above)
    f = tgt - C; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, -1.0]); r /= np.linalg.norm(r); u = np.cross(f, r)
    R = np.stack([r, u, f]); rvec, _ = cv2.Rodrigues(R); tvec = -R @ C
    return dict(K=K, D=D, rvec=rvec.reshape(3, 1), tvec=tvec.reshape(3, 1), rms=0.0)


def keypoints(cal):
    pts = [(x, y) for x in (0, 16.5, 52.5, 88.5, 105) for y in (0, 13.84, 24.84, 43.16, 54.16, 68)]
    px = ST.project(cal, np.array(pts, float))
    return {f"({x:g},{y:g})": [float(a), float(b)] for (x, y), (a, b) in zip(pts, px)}


def ball_path(n):
    """ground ball: 60 frames at a player's feet, then rolls slowly, then 'in the air' (fast), repeated"""
    t = np.arange(n); x = 30 + 40 * (t % 400) / 400.0; y = 20 + 25 * np.sin(t / 150.0)
    return np.stack([x, y], 1)


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    root = tmp_path_factory.mktemp("st"); m = "132877"; cal = camera()
    os.makedirs(root / "raw" / m); os.makedirs(root / "videos" / m); os.makedirs(root / "ball")
    json.dump(keypoints(cal), open(root / "raw" / m / f"{m}_keypoints.json", "w"))
    (root / "ball" / "readme.txt").write_text("stand-in")
    off, n = 7, 600; bm = ball_path(n)
    lines = ["<root>"]
    for per, base in (("FIRST_HALF", off), ("SECOND_HALF", off + 5000)):
        for k in range(n):
            b = bm[k]; ploc = [b + [0.8, 0.0], b + [15, 10], b + [-20, 5]]
            lines.append(f'<frame frameNumber="{base + k}" eventPeriod="{per}">')
            for j, q in enumerate(ploc):                                                   # players: y flipped in 132877
                lines.append(f'<player playerId="{j}" loc="[{q[0] / 105:.5f}, {1 - q[1] / 68:.5f}]"/>')
            lines.append(f'<ball playerId="B" loc="[{b[0] / 105:.5f}, {b[1] / 68:.5f}]"/>')   # ball: NOT flipped in 132877
            lines.append("</frame>")
    lines.append("</root>"); (root / "raw" / m / f"{m}_tracker_box_data.xml").write_text("\n".join(lines))
    true = ST.project(cal, bm)
    for h in ("1st", "2nd"):
        vw = cv2.VideoWriter(str(root / "videos" / m / f"{m}_panorama_{h}_half.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
        for k in range(n):
            img = np.full((H, W, 3), (40, 120, 50), np.uint8)
            cv2.circle(img, (int(round(true[k, 0])), int(round(true[k, 1]))), 3, (235, 235, 235), -1)
            vw.write(img)
        vw.release()
    return root, cal, bm, true, off


def test_flips_and_projection(stand_in):
    root, cal, bm, true, off = stand_in
    per = ST.parse_xml(str(root / "raw" / "132877" / "132877_tracker_box_data.xml"))
    assert per["FIRST_HALF"]["frames"][0] == off and len(per["SECOND_HALF"]["frames"]) == 600
    b = ST.to_metres(per["FIRST_HALF"]["ball"][:5], "132877", "ball"); assert np.allclose(b, bm[:5], atol=0.01)
    p = ST.to_metres(per["FIRST_HALF"]["players"][0][:1], "132877", "player"); assert np.allclose(p[0], bm[0] + [0.8, 0], atol=0.01)
    pitch, image, _ = ST.load_keypoints(str(root / "raw" / "132877" / "132877_keypoints.json"), "132877")
    fit = ST.calibrate(pitch, image, W, H); assert fit["rms"] < 1.0
    assert np.abs(ST.project(fit, bm[:50]) - true[:50]).max() < 2.0


def test_ground_filter():
    per = {"frames": np.arange(30), "ball": np.zeros((30, 2)), "players": [np.array([[0.9, 0.9]])] * 30}
    x = np.r_[np.linspace(40, 44, 15), 44 + np.cumsum(np.full(15, 1.0))]                 # 4 m in 0.6 s, then 25 m/s (flight)
    per["ball"] = np.stack([x / 105, np.full(30, 30 / 68)], 1)
    g = ST.ground_candidates(per, "117092", 25.0)
    assert g["rolling"][6:9].all() and not g["rolling"][20:25].any() and not g["feet"].any()
    per["ball"][3] = [0.5, 0.5]; assert not ST.ground_candidates(per, "132831", 25.0)["inside"][3]   # placeholder dropped


def test_snap_finds_blob():
    img = np.full((200, 200, 3), (40, 120, 50), np.uint8); cv2.circle(img, (120, 90), 3, (240, 240, 240), -1)
    x, y, c, ok = ST.snap(img, (105, 100), 25, 7); assert ok and abs(x - 120) <= 1 and abs(y - 90) <= 1
    x, y, c, ok = ST.snap(np.full((200, 200, 3), (40, 120, 50), np.uint8), (105, 100), 25, 7); assert not ok


def test_dry_run_end_to_end(stand_in, tmp_path):
    root, cal, bm, true, off = stand_in
    env = dict(os.environ, ST_ROOT=str(root), B3_OUT=str(tmp_path), ST_MATCHES="132877", N_PER_HALF="8", MIN_GAP="40")
    r = subprocess.run([sys.executable, "tools/b3_soccertrack.py"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       env=env, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
    labs = json.load(open(tmp_path / "labels.json")); s = json.load(open(tmp_path / "summary.json"))
    assert len(labs) == 16 and s["matches"]["132877"]["1st"]["offset"] == off and s["ball_folder"] == ["ball/readme.txt"]
    for l in labs:
        k = l["xml_frame"] - off - (5000 if l["half"] == "2nd" else 0)
        assert l["kind"] in ("feet", "rolling") and l["snapped"]
        assert np.hypot(l["x"] - true[k, 0], l["y"] - true[k, 1]) <= 2.0
    d = np.load(tmp_path / "crops64.npz"); assert d["X"].shape == (16, 64, 64, 3) and len(d["N"]) == 16
    assert d["X"][:, 30:34, 30:34].mean() > d["N"][:, 30:34, 30:34].mean() + 30          # ball in the middle of X only
    assert (tmp_path / "sheet_132877.jpg").exists() and (tmp_path / "over_132877.jpg").exists()
