"""N0-30b (30 Oct, worker, free runner, $0, nothing on Modal): cut random SECOND-half AIK frames for a blind by-eye
possession sample (N0-30a covered the first half only: 60 frames, 27 decided).

    python tools/n0_30b_aik_h2_cut.py            # on the free runner (R2_PUBLIC_URL set)
    SRC=some.mp4 python tools/n0_30b_aik_h2_cut.py   # dry run on a local video

Frames: N random times (seed 11) in the second half after the kick-off (3995 s - 6899 s; the periods file starts at 3950 s,
the kick-off is ~3990 s, N0-30a), full size, nothing drawn -> results/free/n0_30b/h2/q<NN>.jpg + times.json.
"""
import os, sys, json, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = "p15u-vs-aik-2026-09-21-bd09"
HALF = (3995.0, 6899.0)
N = int(os.environ.get("N", "90"))
SEED = 11
OUT = os.environ.get("OUT", "results/free/n0_30b/h2")


def sample_times(n=N, seed=SEED, half=HALF):
    r = random.Random(seed)
    return sorted(round(r.uniform(*half), 1) for _ in range(n))


def cut(src, times, out, cv, log=print):
    from clipopen import open_clip
    os.makedirs(out, exist_ok=True)
    cap, fps, n_all, tries = open_clip(src, cv, log=log)
    if cap is None:
        raise SystemExit(f"could not open {src}")
    got = []
    for i, t in enumerate(times):
        k = int(round(t * fps))
        if n_all and k >= n_all:
            continue
        ok, f = False, None
        for _ in range(3):                          # a seek over HTTP can fail once on a stall: re-open and retry
            cap.set(cv.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
            if ok: break
            cap.release(); cap, fps, n_all, _ = open_clip(src, cv, log=log)
        if ok:
            name = f"q{i:02d}.jpg"
            cv.imwrite(os.path.join(out, name), f, [cv.IMWRITE_JPEG_QUALITY, 85])
            got.append({"file": name, "t": t, "frame": k})
    cap.release()
    meta = {"match": M, "fps": fps, "n_frames_video": n_all, "seed": SEED, "half": HALF, "frames": got}
    json.dump(meta, open(os.path.join(out, "times.json"), "w"), indent=1)
    log(f"{len(got)}/{len(times)} frames cut, fps {fps}")
    return meta


def main():
    import cv2
    src = os.environ.get("SRC")
    if not src:
        base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        if not base:
            raise SystemExit("set R2_PUBLIC_URL or SRC")
        src = f"{base}/{M}/video.mp4"
        # R2 cuts long streams (N0-28c), and 90 seeks across a whole match over HTTP is fragile: download it whole
        # first (Kaggle did the same for N0-30a); falls back to streaming if the download fails.
        from clipopen import local_copy
        src = local_copy(src, dest="/tmp/aik_video.mp4", max_bytes=12e9)
    cut(src, sample_times(), OUT, cv2)


if __name__ == "__main__":
    main()
