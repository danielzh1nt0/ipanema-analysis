"""B4: auto ball-key moments = picker + both finders agree; sheets cut from the clip (dry run on a fake clip)."""
import os, sys, json, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ballkey_auto as AK

def _data(n=60):
    pick = {i: [100.0 + i, 200.0] for i in range(n)}
    rf = {i: [(100.0 + i, 201.0, 0.8), (500.0, 500.0, 0.3)] for i in range(n)}
    wasb = {i: [(102.0 + i, 199.0, 0.7)] for i in range(n)}
    return pick, rf, wasb

def test_agreed_needs_all_three_close_and_confident():
    pick, rf, wasb = _data()
    assert AK.agreed(5, pick, rf, wasb)
    wasb[6] = [(300.0, 199.0, 0.9)]; assert not AK.agreed(6, pick, rf, wasb)          # WASB elsewhere
    rf[7] = [(107.0, 200.0, 0.2)]; assert not AK.agreed(7, pick, rf, wasb)            # RF-DETR unsure
    del pick[8]; assert not AK.agreed(8, pick, rf, wasb)                              # picker has no ball

def test_select_spacing_steadiness_and_avoid():
    pick, rf, wasb = _data()
    s = AK.select(pick, rf, wasb, 60, gap=15, steady=0)
    assert [f for f, *_ in s] == [0, 15, 30, 45]
    del pick[16]
    s = AK.select(pick, rf, wasb, 60, gap=15, steady=2, avoid=[45])
    fr = [f for f, *_ in s]
    assert 15 not in fr and 16 not in fr and 45 not in fr and all(b - a >= 15 for a, b in zip(fr, fr[1:]))

def test_tile_does_not_cover_the_ball():
    img = np.zeros((1080, 1920, 3), np.uint8); cv2.circle(img, (10, 10), 4, (255, 255, 255), -1)
    t = AK.tile(img, 10, 10, half=48, scale=2)
    assert t.shape == (192, 192, 3) and t[96, 96].min() == 255                      # ball at the centre, edge-padded
    assert t[96, 96 + 22].tolist() == [0, 255, 255]                                  # tick drawn outside the ball

def test_sheet_dry_run_on_fake_clip(tmp_path):
    clip = str(tmp_path / "clip.mp4"); vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for i in range(40):
        f = np.full((1080, 1920, 3), (40, 120, 40), np.uint8); cv2.circle(f, (300 + 10 * i, 500), 6, (255, 255, 255), -1); vw.write(f)
    vw.release()
    c = {"clip": "fake", "moments": [{"id": k, "frame": f, "x": 300 + 10 * f, "y": 500} for k, f in enumerate((3, 10, 25, 39))]}
    (tmp_path / "c.json").write_text(json.dumps(c))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, LOCAL_CLIP=clip, B4_CANDS=str(tmp_path / "c.json"), B4_OUT=str(tmp_path / "out"))
    r = subprocess.run([sys.executable, f"{repo}/tools/b4_sheet.py"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.load(open(tmp_path / "out/sheets.json")) == {"cut": 4, "of": 4, "sheets": 1}
    X = np.load(tmp_path / "out/crops.npz")["X"]; assert X.shape == (4, 64, 64, 3)
    assert (X[:, 32, 32] > 200).all()                                                # the ball sits in the middle of every crop
