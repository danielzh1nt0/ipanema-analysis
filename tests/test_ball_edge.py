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
