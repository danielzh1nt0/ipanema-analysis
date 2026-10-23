"""BALL-EDGE (23 Oct, free runner): re-detect a slice of an app test clip with the app's RF-DETR ball finder
(ipanema/ballrf.py) and keep EVERY tile guess before merging, as (frame, x, y, conf, cut-by-tile-inner-side).
From the same tile guesses the grader (tools/ball_edge_grade.py) merges both ways: old and IPANEMA_BALL_EDGE=1.
The whole clip on one free runner is ~14 h of CPU, so it runs as shards in parallel (one push of triggers/free.txt each):
    R2_PUBLIC_URL=... python tools/ball_edge_shard.py sfk 3 8      # 4th of 8 slices of SFK-BP
    dry run: LOCAL_CLIP=v.mp4 FAKE_MODEL=1 N_FRAMES=40 python tools/ball_edge_shard.py sfk 0 2 out_dir
Writes <out>/<clip>_<i>of<n>.npz (f, x, y, c, cut) + .json (frames, minutes)."""
import os, sys, json, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from ipanema import ballrf as BR

CLIPS = {"sfk": "SFKBP1109_s1200", "aik": "p15u-vs-aik-2026-09-21-bd09_s2520"}
N_FRAMES = 8992                      # both clips (picker_inputs.pkl: frames 0-8991)
OUT = "results/qa/ball-edge/raw"


def span(i, n, total=N_FRAMES):
    """slice i of n -> [a, b) frame range; slices cover 0..total exactly"""
    return i * total // n, (i + 1) * total // n


def fetch(clip):
    if os.environ.get("LOCAL_CLIP"): return os.environ["LOCAL_CLIP"]
    dst = f"/tmp/be_{clip}.mp4"
    if os.path.exists(dst): return dst
    url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{clip}/video.mp4"; print("downloading", clip, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    return dst


class Fake:
    """stand-in detector: one box cut by every tile's right side, one whole box"""
    def predict(self, crops, threshold):
        class D: pass
        out = []
        for _ in crops:
            d = D(); d.xyxy = np.array([[630.0, 300, 640, 310], [300, 300, 312, 312]]); d.confidence = np.array([0.5, 0.3]); out.append(d)
        return out


def run(key, i, n, out=OUT, model=None, total=None):
    t0 = time.time(); clip = CLIPS[key]; total = total or int(os.environ.get("N_FRAMES", N_FRAMES)); a, b = span(i, n, total)
    model = model or (Fake() if os.environ.get("FAKE_MODEL") == "1" else BR.load(BR.WEIGHTS))
    cap = cv2.VideoCapture(fetch(clip)); k = 0; buf = []; rows = []
    def flush():
        raw = []; BR.detect_many(model, [f for _, f in buf], edge=False, raw=raw)
        for (kk, _), r in zip(buf, raw): rows.extend((kk, x, y, c, cut) for x, y, c, cut in r)
        buf.clear()
    while k < b:
        if k < a:
            if not cap.grab(): break
        else:
            ok, fr = cap.read()
            if not ok: break
            buf.append((k, fr if fr.shape[:2] == (1080, 1920) else cv2.resize(fr, (1920, 1080))))
            if len(buf) >= 4: flush()
            if (k - a) % 200 == 0: print(f"{key} {i}/{n}: frame {k} ({k - a}/{b - a}, {time.time() - t0:.0f} s)", flush=True)
        k += 1
    if buf: flush()
    cap.release(); os.makedirs(out, exist_ok=True); stem = f"{out}/{key}_{i}of{n}"
    R = np.array(rows, dtype=np.float64).reshape(-1, 5)
    np.savez_compressed(stem + ".npz", f=R[:, 0].astype(np.int32), x=R[:, 1].astype(np.float32), y=R[:, 2].astype(np.float32),
                        c=R[:, 3].astype(np.float32), cut=R[:, 4].astype(bool))
    done = sorted(set(R[:, 0].astype(int).tolist()))
    info = {"clip": clip, "slice": [i, n], "frames_wanted": [a, b], "frames_done": len(done), "first": done[0] if done else None,
            "last": done[-1] if done else None, "tile_guesses": len(R), "minutes": round((time.time() - t0) / 60, 1),
            "fake_model": os.environ.get("FAKE_MODEL") == "1"}
    json.dump(info, open(stem + ".json", "w"), indent=1); print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__":
    if os.environ.get("FAKE_MODEL") != "1":
        try: import rfdetr  # noqa
        except ImportError:
            import subprocess
            subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else OUT)
