"""B3b: SoccerTrack labels checked by our finder. Helpers: window cut + way back to frame pixels at 1x/2x and at the
image edge, guesses near the projection, scale merging, keep rule. End to end: a stand-in video with a white ball 25 px
from each projection (and one frame with two balls, one with none) through tools/b3b_finder.py with DRY=1."""
import json, os, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import b3b as BB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_window_maps_back_at_both_scales_and_edges():
    img = np.zeros((1080, 4096, 3), np.uint8)
    for cx, cy in [(2000, 500), (10, 10), (4090, 1075)]:
        bx, by = int(np.clip(cx + 15, 3, 4092)), int(np.clip(cy - 12, 3, 1076)); img[:] = 0; cv2.circle(img, (bx, by), 3, (255, 255, 255), -1)
        for s in (1.0, 2.0):
            c, t = BB.window(img, cx, cy, 640, s)
            assert c.shape == (640, 640, 3)
            ys, xs = np.nonzero(c[:, :, 0] > 128)
            fx, fy, _ = BB.to_frame([(xs.mean(), ys.mean(), 1.0)], t)[0]
            assert abs(fx - bx) < 1.5 and abs(fy - by) < 1.5


def test_window_pads_a_short_image():
    img = np.full((300, 900, 3), 50, np.uint8)
    c, t = BB.window(img, 450, 150, 640, 1.0)
    assert c.shape == (640, 640, 3) and t[1] == 0 and c[400, 10, 0] == 114


def test_near_merge_decide():
    proj = (100, 100)
    d = BB.near([(130, 100, 0.9), (300, 100, 0.99), (110, 140, 0.2)], proj)
    assert [g[2] for g in d] == [0.9, 0.2]                                        # far one ignored, best first
    g = BB.merge_scales({1.0: [(130, 100, 0.7)], 2.0: [(131, 101, 0.9), (90, 80, 0.3)]})
    assert g[0]["conf"] == 0.9 and g[0]["n"] == 2 and g[0]["scales"] == [1.0, 2.0] and g[1]["n"] == 1
    assert BB.decide(g) == (True, "kept")
    assert BB.decide([dict(x=0, y=0, conf=0.4, scales=[1.0], n=1)]) == (False, "weak")
    assert BB.decide([dict(x=0, y=0, conf=0.9, scales=[1.0], n=1), dict(x=30, y=0, conf=0.6, scales=[2.0], n=1)]) == (False, "two guesses")
    assert BB.decide([dict(x=0, y=0, conf=0.9, scales=[1.0], n=1), dict(x=30, y=0, conf=0.4, scales=[2.0], n=1)])[0]
    assert BB.decide([]) == (False, "no guess")


def test_end_to_end_dry(tmp_path):
    W, H = 1600, 600; os.makedirs(tmp_path / "st" / "videos" / "m1")
    vid = tmp_path / "st" / "videos" / "m1" / "v.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H)); labels = []
    for k in range(40):
        f = np.full((H, W, 3), (40, 120, 40), np.uint8); px, py = 200 + 30 * k, 300
        if k % 10 != 8: cv2.circle(f, (px + 25, py), 4, (255, 255, 255), -1)          # ball 25 px from the projection
        if k == 4: cv2.circle(f, (px - 25, py + 10), 4, (255, 255, 255), -1)          # a second ball: unsure
        vw.write(f)
        if k % 2 == 0: labels.append(dict(match="m1", half="1st", video="videos/m1/v.mp4", frame=k, x_raw=px, y_raw=py, x=px, y=py,
                                          snapped=False, ball_px=8.0, kind="feet"))
    vw.release(); json.dump(labels, open(tmp_path / "labels.json", "w"))
    env = dict(os.environ, DRY="1", ST_ROOT=str(tmp_path / "st"), B3_LABELS=str(tmp_path / "labels.json"), B3B_OUT=str(tmp_path / "out"))
    subprocess.run([sys.executable, os.path.join(ROOT, "tools/b3b_finder.py")], env=env, check=True, cwd=ROOT)
    out = json.load(open(tmp_path / "out" / "labels.json")); summ = json.load(open(tmp_path / "out" / "summary.json"))
    frames = {l["frame"] for l in out}
    assert 4 not in frames                                                         # two balls -> not kept
    assert not frames & {k for k in range(40) if k % 10 == 8}                      # no ball -> not kept
    assert len(out) == 20 - 1 - 4 and summ["kept"] == len(out)
    for l in out: assert abs(l["x"] - (l["x_raw"] + 25)) < 2 and abs(l["y"] - l["y_raw"]) < 2
    assert np.load(tmp_path / "out" / "crops64.npz")["X"].shape == (len(out), 64, 64, 3)
    assert os.path.exists(tmp_path / "out" / "kept_m1.jpg")
