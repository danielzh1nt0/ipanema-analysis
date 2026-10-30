"""N1c (30 Oct, worker, local + free runner, $0, nothing on Modal): AIK team labels from the shirt colour, read again
independently of the pipeline's kit model, at every exported player position.

Why: N0-30b found AIK carriers labelled SFK (team A): the red AIK keeper (3/3), white AIK players in some second-half
pieces, and the referee (light blue) usually gets a team. The Kaggle relabel stage (kaggle/full_match.py STAGE=relabel)
decides each track by the majority of its A/B reads and ignores its 'neither' reads, so a keeper or referee track that
reads 'neither' most of the time is still forced into a team by a few stray A/B reads.

The export (frames_lite) has foot points (px) but no boxes or colours, so here the torso is cut above the foot with a
height estimated from the foot's height in the picture, and each torso is put in one of five kit classes:
    dark   SFK shirt (black / maroon)          -> team A
    white  AIK shirt                            -> team B
    red    AIK keeper                           -> team B, keeper
    blue   referee (light blue)                 -> no team
    none   too few shirt pixels / can't tell
Per export track id the classes are counted over the match (tools/n1c_kitvote_run.py on the free runner).

    python tools/n1c_kitread.py sheet      # local: torso sheets per class from the 150 full-size frames already in the repo
"""
import os, sys, json, gzip, bisect

FR_PATH = "results/kaggle/full_match_52d795-join/frames_lite.json.gz"
FRAME_SETS = [("results/kaggle/aik_poss_sample/poss", "p"), ("results/free/n0_30b/h2", "q")]
OUT = "results/qa/n1c"
CLASSES = ("dark", "white", "red", "blue", "none")


def player_height_px(y_foot):
    """rough player height in a 1080p Veo frame from the foot's row: ~60 px at y 520, ~130 px at y 770 (measured by eye
    on the AIK frame q64, 6050.5 s)."""
    return max(24.0, 60.0 + (y_foot - 520.0) * 0.28)


def torso_box(px, shape):
    """(x0, y0, x1, y1) of the shirt above a foot point, clipped to the frame; None if it falls outside."""
    h_img, w_img = shape[:2]
    x, y = px; h = player_height_px(y)
    x0, x1 = int(round(x - 0.16 * h)), int(round(x + 0.16 * h))
    y0, y1 = int(round(y - 0.86 * h)), int(round(y - 0.62 * h))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w_img, x1), min(h_img, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def classify_patch(bgr, cv2, np):
    """-> (class, features). Grass pixels (green hue, some saturation) are dropped first."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0] * 2.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    grass = (h > 60) & (h < 170) & (s > 0.25) & (v > 0.15)
    keep = ~grass
    n = int(keep.sum())
    if n < max(8, 0.25 * len(h)):
        return "none", {"n": n}
    h, s, v = h[keep], s[keep], v[keep]
    # a white shirt in the shade reads blue-grey (hue 196-220, v 0.4-0.75 on the AIK frames, yellow-black AIK socks below);
    # the referee's light-blue shirt is greener and brighter (hue 180-196, v > 0.7) - measured on results/qa/n1c/sheet_blue.jpg
    white = ((v > 0.62) & (s < 0.22)) | ((h >= 196) & (h < 222) & (s > 0.25) & (v > 0.38) & (v < 0.80))
    dark = v < 0.30
    red = ((h < 20) | (h > 330)) & (s > 0.45) & (v > 0.30)
    blue = (h >= 168) & (h < 196) & (s > 0.30) & (v > 0.70)
    f = {"n": n, "white": round(float(white.mean()), 3), "dark": round(float(dark.mean()), 3),
         "red": round(float(red.mean()), 3), "blue": round(float(blue.mean()), 3), "v": round(float(np.median(v)), 3)}
    if f["blue"] >= 0.30 and f["blue"] > f["white"]:
        return "blue", f
    if f["red"] >= 0.30 and f["red"] > f["dark"]:
        return "red", f
    if f["white"] >= 0.30 and f["white"] > 1.5 * f["dark"]:
        return "white", f
    if f["dark"] >= 0.35 and f["dark"] > 2.0 * f["white"]:
        return "dark", f
    return "none", f


TEAM_OF = {"dark": "A", "white": "B", "red": "B"}


def load_export(path=FR_PATH):
    d = json.load(gzip.open(path))
    fr = d["frames"]
    return fr, [f["t"] for f in fr]


def nearest(ts, t):
    i = bisect.bisect_left(ts, t)
    if i == 0: return 0
    if i >= len(ts): return len(ts) - 1
    return i if ts[i] - t < t - ts[i - 1] else i - 1


def interp_row(fr, ts, t):
    """export row at time t with every player's foot point interpolated between the two rows around t (rows are 0.3 s
    apart; a running player moves 50+ px in that time near the camera). Players only in one of the two rows keep it."""
    i = bisect.bisect_right(ts, t)
    if i == 0 or i >= len(ts): return fr[nearest(ts, t)]
    a, b = fr[i - 1], fr[i]; w = (t - a["t"]) / max(1e-6, b["t"] - a["t"])
    pb = {p["id"]: p for p in b["players"]}; out = []
    for p in a["players"]:
        q = pb.get(p["id"])
        if q and p.get("px") and q.get("px"):
            p = dict(p, px=[p["px"][0] + w * (q["px"][0] - p["px"][0]), p["px"][1] + w * (q["px"][1] - p["px"][1])])
        out.append(p)
    return {"t": t, "players": out, "possession": (a if w < 0.5 else b)["possession"], "carrier": (a if w < 0.5 else b)["carrier"]}


def read_frame(img, row, cv2, np):
    """class of every exported player in one frame -> list of (id, team, gk, class, feats, box)"""
    out = []
    for p in row["players"]:
        if not p.get("px"):
            continue
        b = torso_box(p["px"], img.shape)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        c, f = classify_patch(img[y0:y1, x0:x1], cv2, np)
        out.append((p["id"], p["team"], bool(p.get("gk")), c, f, b))
    return out


def local_frames():
    """(path, t) of the full-size AIK frames already in the repo"""
    out = []
    for d, _ in FRAME_SETS:
        tj = os.path.join(d, "times.json")
        if not os.path.exists(tj):
            continue
        for f in json.load(open(tj))["frames"]:
            out.append((os.path.join(d, f["file"]), f["t"]))
    return out


def sheet():
    import cv2, numpy as np
    fr, ts = load_export()
    tiles = {c: [] for c in CLASSES}; stats = {}
    for path, t in local_frames():
        img = cv2.imread(path)
        if img is None: continue
        row = interp_row(fr, ts, t)
        for pid, team, gk, c, f, (x0, y0, x1, y1) in read_frame(img, row, cv2, np):
            h = y1 - y0; X0 = max(0, x0 - h); X1 = min(img.shape[1], x1 + h); Y0 = max(0, y0 - h); Y1 = min(img.shape[0], y1 + 2 * h)
            crop = img[Y0:Y1, X0:X1].copy()
            cv2.rectangle(crop, (x0 - X0, y0 - Y0), (x1 - X0, y1 - Y0), (0, 255, 255), 1)
            crop = cv2.resize(crop, (64, 96))
            cv2.putText(crop, team + ("k" if gk else ""), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if team == "A" else (255, 0, 0), 1)
            tiles[c].append(crop)
            stats.setdefault(c, {}).setdefault(team, 0); stats[c][team] += 1
    os.makedirs(OUT, exist_ok=True)
    for c, ts_ in tiles.items():
        if not ts_: continue
        ts_ = ts_[:240]; cols = 20
        rows = [np.hstack(ts_[i:i + cols] + [np.zeros_like(ts_[0])] * (cols - len(ts_[i:i + cols]))) for i in range(0, len(ts_), cols)]
        cv2.imwrite(os.path.join(OUT, f"sheet_{c}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    if sys.argv[1:2] == ["sheet"]:
        sheet()
