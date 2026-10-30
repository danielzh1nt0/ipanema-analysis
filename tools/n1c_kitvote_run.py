"""N1c (30 Oct, worker, free runner, $0, nothing on Modal): read the shirt colour of every exported AIK player every
STEP rows (0.3 s each) over both halves and count the kit classes per export track id (tools/n1c_kitread.py).

    python tools/n1c_kitvote_run.py                       # free runner (R2_PUBLIC_URL set): whole match
    SRC=some.mp4 LIMIT_S=20 python tools/n1c_kitvote_run.py   # dry run on a local stand-in video

The video is downloaded whole first (4 GB, as N0-30b) and read straight through (grab every frame, decode only the
wanted ones) - seeking 7,000+ times in a long mp4 is slower and less safe.
Output: results/free/n1c/reads.json.gz = {"step", "fps", "rows": [[t, [[id, class], ...]], ...]} + votes.json.
"""
import os, sys, json, gzip, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1c_kitread as K

M = "p15u-vs-aik-2026-09-21-bd09"
PERIODS = [(366.0, 3225.0), (3950.0, 6899.0)]          # periods/<match>.json
STEP = int(os.environ.get("STEP", "3"))                  # every 3rd export row = 0.9 s
OUT = os.environ.get("OUT", "results/free/n1c")
CLS = {c: i for i, c in enumerate(K.CLASSES)}


def wanted_rows(fr, step=STEP, periods=PERIODS, limit_s=None):
    """export rows with players inside the halves, every step-th -> list of row indices"""
    keep = [i for i, f in enumerate(fr) if f["players"] and any(a <= f["t"] <= b for a, b in periods)]
    keep = keep[::step]
    if limit_s is not None:
        t0 = fr[keep[0]]["t"] if keep else 0
        keep = [i for i in keep if fr[i]["t"] - t0 <= limit_s]
    return keep


def run(src, fr, rows, cv2, np, fps=None, log=print, t_shift=0.0):
    """read the frames at the wanted rows' times straight through -> (reads, votes). t_shift moves the export times onto
    the video clock (0 for the AIK video; a dry run on a short stand-in video uses -t of the first row)."""
    cap = cv2.VideoCapture(src)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 29.97
    want = sorted((int(round((fr[i]["t"] + t_shift) * fps)), i) for i in rows)
    reads, votes = [], {}
    k = -1; j = 0; t0 = time.time()
    while j < len(want):
        target = want[j][0]
        if target < k + 1:                                    # two rows on the same frame
            j += 1; continue
        while k + 1 < target:
            if not cap.grab(): break
            k += 1
        ok, img = cap.read(); k += 1
        if not ok:
            log(f"video ended at frame {k}"); break
        i = want[j][1]; row = fr[i]
        got = K.read_frame(img, row, cv2, np)
        reads.append([round(row["t"], 3), [[pid, CLS[c]] for pid, _, _, c, _, _ in got]])
        for pid, _, _, c, _, _ in got:
            v = votes.setdefault(str(pid), [0] * len(K.CLASSES)); v[CLS[c]] += 1
        j += 1
        if j % 500 == 0:
            log(f"{j}/{len(want)} rows read, video {row['t']:.0f} s, {time.time() - t0:.0f} s wall")
    cap.release()
    return reads, votes


def main():
    import cv2, numpy as np
    fr, _ = K.load_export()
    src = os.environ.get("SRC")
    limit = float(os.environ["LIMIT_S"]) if os.environ.get("LIMIT_S") else None
    shift = 0.0
    if not src:
        base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        if not base:
            raise SystemExit("set R2_PUBLIC_URL or SRC")
        from clipopen import local_copy
        src = local_copy(f"{base}/{M}/video.mp4", dest="/tmp/aik_video.mp4", max_bytes=12e9)
    rows = wanted_rows(fr, limit_s=limit)
    if os.environ.get("SRC") and limit is not None:
        shift = -fr[rows[0]]["t"]                             # stand-in video starts at the first wanted row
    print(f"{len(rows)} rows to read from {src}", flush=True)
    reads, votes = run(src, fr, rows, cv2, np, log=lambda *a: print(*a, flush=True), t_shift=shift)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"match": M, "step": STEP, "classes": list(K.CLASSES), "rows": reads}, gzip.open(os.path.join(OUT, "reads.json.gz"), "wt"))
    json.dump({"classes": list(K.CLASSES), "votes": votes}, open(os.path.join(OUT, "votes.json"), "w"))
    tot = [sum(v[c] for v in votes.values()) for c in range(len(K.CLASSES))]
    print(json.dumps({"rows_read": len(reads), "tracks": len(votes), "reads_per_class": dict(zip(K.CLASSES, tot))}), flush=True)


if __name__ == "__main__":
    main()
