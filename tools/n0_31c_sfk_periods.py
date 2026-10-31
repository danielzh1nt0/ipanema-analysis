"""N0-31c (31 Oct, worker, free runner, $0, nothing on Modal): check SFK-BP's half-time and full-time by eye.

periods/SFKBP1109.json has the first-half kick-off checked by eye (555 s, 2 Oct), but the first-half END (3060 s,
'51:00 from the coach'), the second-half kick-off (3733 s, from Veo's goal times +-30 s) and full time (5940 s,
'1:39:00 from the coach') were never looked at. AIK's first half once counted 3 min of empty pitch as play (D6) and
its match end was off (N0-30b), so this cuts frames around each boundary for a by-eye look.

    python tools/n0_31c_sfk_periods.py              # on the free runner (R2_PUBLIC_URL set)
    SRC=some.mp4 python tools/n0_31c_sfk_periods.py # dry run on a local video

-> results/free/n0_31c/<window>_<t>.jpg (960 px wide, time stamped) + sheet_<window>.jpg (4 columns) + times.json
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = "SFKBP1109"
# (name, start, end, step): around each boundary in periods/SFKBP1109.json
WINDOWS = [("ht", 2960.0, 3290.0, 10.0),      # first-half end 3060 (coach)
           ("ko2", 3620.0, 3830.0, 10.0),     # second-half kick-off 3733 (+-30 s)
           ("ft", 5820.0, 6080.0, 10.0)]      # full time 5940 (coach)
OUT = os.environ.get("OUT", "results/free/n0_31c")
W = 960


def window_times(windows=WINDOWS):
    out = []
    for name, a, b, step in windows:
        t = a
        while t <= b + 1e-6:
            out.append((name, round(t, 1))); t += step
    return out


def label(img, text, cv):
    cv.rectangle(img, (0, 0), (10 + 15 * len(text), 34), (0, 0, 0), -1)
    cv.putText(img, text, (6, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def sheet(paths, cv, np, cols=4):
    tiles = [cv.resize(cv.imread(p), (480, 270)) for p in paths]
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def cut(src, times, out, cv, np, log=print):
    from clipopen import open_clip
    os.makedirs(out, exist_ok=True)
    cap, fps, n_all, _ = open_clip(src, cv, log=log)
    if cap is None:
        raise SystemExit(f"could not open {src}")
    got, by_win = [], {}
    for name, t in times:
        k = int(round(t * fps))
        if n_all and k >= n_all:
            continue
        ok, f = False, None
        for _ in range(3):
            cap.set(cv.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
            if ok: break
            cap.release(); cap, fps, n_all, _ = open_clip(src, cv, log=log)
        if not ok:
            continue
        h, w = f.shape[:2]
        f = cv.resize(f, (W, int(h * W / w)))
        label(f, f"{name} {t:.0f} s ({mmss(t)})", cv)
        fn = f"{name}_{int(t):05d}.jpg"
        cv.imwrite(os.path.join(out, fn), f, [cv.IMWRITE_JPEG_QUALITY, 80])
        got.append({"window": name, "t": t, "frame": k, "file": fn})
        by_win.setdefault(name, []).append(os.path.join(out, fn))
    cap.release()
    for name, ps in by_win.items():
        cv.imwrite(os.path.join(out, f"sheet_{name}.jpg"), sheet(ps, cv, np), [cv.IMWRITE_JPEG_QUALITY, 80])
    meta = {"match": M, "fps": fps, "n_frames_video": n_all, "video_s": round(n_all / fps, 1) if n_all else None,
            "windows": WINDOWS, "frames": got}
    json.dump(meta, open(os.path.join(out, "times.json"), "w"), indent=1)
    log(f"{len(got)}/{len(times)} frames cut, fps {fps}, video {meta['video_s']} s")
    return meta


def main():
    import cv2, numpy as np
    src = os.environ.get("SRC")
    if not src:
        base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        if not base:
            raise SystemExit("set R2_PUBLIC_URL or SRC")
        from clipopen import local_copy
        src = local_copy(f"{base}/{M}/video.mp4", dest="/tmp/sfk_video.mp4", max_bytes=12e9)
    cut(src, window_times(), OUT, cv2, np)


if __name__ == "__main__":
    main()
