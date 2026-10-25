"""TOUCHLINE (free runner, no Modal): picture sheets of the fresh blind sample (tools/touchline_moments.py): per person 4 moments,
each the whole picture (640 px, ring = feet) next to a 640x360 full-resolution crop. Which are dropped / kept is NOT on
the sheet (results/qa/touchline/moments_key.json).
    R2_PUBLIC_URL=... python tools/touchline_frames.py   (LOCAL_CLIP_VAL / _SFK / _AIK = local videos for a dry run)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n0_25b_frames as F


def run(src_dir="results/qa/touchline", out_root="results/free/touchline"):
    ok = True
    for short in ("val", "sfk", "aik"):
        ok &= sheets(json.load(open(f"{src_dir}/moments_{short}.json")), short, out_root)
    return ok


def sheets(M, short, out_root):
    import numpy as np, cv2
    out = f"{out_root}/{short}"; os.makedirs(out, exist_ok=True)
    # full-match videos are big: read them straight from R2 (OpenCV seeks over HTTP, as tools/k3c_piece_frames.py)
    src = os.environ.get(f"LOCAL_CLIP_{short.upper()}") or os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{M['clip']}/video.mp4"
    cap = cv2.VideoCapture(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); vfps = cap.get(cv2.CAP_PROP_FPS) or 29.97; made = {}
    print(short, M["clip"], "frames", n, "fps", vfps, flush=True)
    for m in M["moments"]:
        rows = []
        for tl in m["tiles"]:
            fno = tl["frame"] if "frame" in tl else int(round(tl["t"] * vfps))
            if fno >= n: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno); ok, fr = cap.read()
            if not ok: continue
            sx = fr.shape[1] / 1920.0; px, py = tl["ball_px"][0] * sx, tl["ball_px"][1] * sx
            full = cv2.resize(fr, (F.TW, F.TH), interpolation=cv2.INTER_AREA); f = F.TW / fr.shape[1]
            cv2.circle(full, (int(px * f), int(py * f)), 12, (0, 0, 255), 2, cv2.LINE_AA)
            x0 = int(np.clip(px - F.TW / 2, 0, fr.shape[1] - F.TW)); y0 = int(np.clip(py - F.TH * 0.7, 0, fr.shape[0] - F.TH))
            crop = fr[y0:y0 + F.TH, x0:x0 + F.TW].copy(); cv2.circle(crop, (int(px - x0), int(py - y0)), 22, (0, 0, 255), 2, cv2.LINE_AA)
            F.put(full, f"{m['id']}  t {tl['t']:.1f} s", 22); rows.append(np.hstack([full, crop]))
        if rows: cv2.imwrite(f"{out}/{m['id']}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); made[m["id"]] = len(rows)
    json.dump({"clip": M["clip"], "frames": n, "sheets": made}, open(f"{out}/sheets.json", "w"), indent=1); print(short, "sheets:", len(made), flush=True)
    return len(made) == len(M["moments"])


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
