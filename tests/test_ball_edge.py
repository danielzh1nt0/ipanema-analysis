"""BALL-EDGE (23 Oct): shard re-detection of the app test clips + the old vs IPANEMA_BALL_EDGE=1 picker grade."""
import os, sys, subprocess, json
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tools import ball_edge_shard as S


def test_span_covers_clip():
    for n in (1, 3, 8, 16):
        parts = [S.span(i, n) for i in range(n)]
        assert parts[0][0] == 0 and parts[-1][1] == S.N_FRAMES
        assert all(parts[i][1] == parts[i + 1][0] for i in range(n - 1))


def test_shard_dry_run(tmp_path):
    p = str(tmp_path / "v.mp4"); w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for i in range(12): w.write(np.full((360, 640, 3), i * 10, np.uint8))
    w.release()
    env = dict(os.environ, LOCAL_CLIP=p, FAKE_MODEL="1", N_FRAMES="12", PYTHONPATH=ROOT)
    for i in range(3):
        subprocess.run([sys.executable, f"{ROOT}/tools/ball_edge_shard.py", "aik", str(i), "3", str(tmp_path / "raw")], env=env, check=True, capture_output=True)
    fr = []
    for i in range(3):
        d = np.load(tmp_path / "raw" / f"aik_{i}of3.npz"); info = json.load(open(tmp_path / "raw" / f"aik_{i}of3.json"))
        assert info["frames_done"] == 4 and d["cut"].dtype == bool and d["cut"].any() and not d["cut"].all()
        fr += sorted(set(d["f"].tolist()))
    assert fr == list(range(12))


def test_wasb_read_back_reproduces_fuse():
    from tools import ball_edge_grade as G
    from ipanema import ball as BL
    rf = {0: [(100.0, 100.0, 0.6), (500.0, 300.0, 0.3), (900.0, 50.0, 0.9)], 1: [(10.0, 10.0, 0.2)], 2: []}
    wb = {0: [(104.0, 98.0, 0.5), (1500.0, 700.0, 0.4), (902.0, 51.0, 0.9)], 1: [], 2: [(5.0, 5.0, 0.7)]}
    F = BL.fuse_candidates(rf, wb)
    W = {k: G.wasb_back(F[k], rf.get(k, []))[0] for k in F}
    F2 = BL.fuse_candidates(rf, W)
    for k in F:                                                    # same scores at the same places (cap: 900,50 stays 0.99)
        assert len(F2[k]) == len(F[k])
        for a, b in zip(sorted(F[k]), sorted(F2[k])): assert abs(a[2] - b[2]) < 0.01 and abs(a[0] - b[0]) < 0.5


def test_edge_arm_drops_cut_guess_keeps_wasb():
    from tools import ball_edge_grade as G
    raw = {0: [(100.0, 100.0, 0.3, True), (300.0, 300.0, 0.4, False)]}
    from ipanema import ball as BL
    F = BL.fuse_candidates({0: [(100.0, 100.0, 0.3), (300.0, 300.0, 0.4)]}, {0: [(101.0, 100.0, 0.5)]})
    co, cn, W, ro, rn, st = G.arms(raw, F)
    assert len(ro[0]) == 2 and len(rn[0]) == 1 and st["rf_old"] - st["rf_new"] == 1
    assert any(abs(x - 100) < 2 and abs(c - 0.25) < 0.01 for x, y, c in cn[0])    # the WASB guess there survives alone (0.5 / 2)
    assert sorted(round(c, 3) for _, _, c in co[0]) == sorted(round(c, 3) for _, _, c in F[0])
