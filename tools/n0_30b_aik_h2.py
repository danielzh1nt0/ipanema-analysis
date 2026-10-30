"""N0-30b (30 Oct, worker, local, $0): AIK SECOND-half possession vs a blind by-eye sample (N0-30a did the first half).

Frames: results/free/n0_30b/h2 (90 random second-half times, seed 11, cut on the free runner by tools/n0_30b_aik_h2_cut.py).

    python tools/n0_30b_aik_h2.py views    # grading views: full frame + a zoom around the exported ball spot (no team,
                                           # no possession drawn - blind to our answer, not to our ball)
    python tools/n0_30b_aik_h2.py          # compare the committed grades with the export -> results/qa/n0-30b/result.json

Grades: results/qa/n0-30b/aik_h2_grades.json (A = dark SFK, B = white AIK, null = can't tell; sure; dead = restart).
"""
import os, sys, json, gzip, bisect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n0_30a_aik_poss import analyse, export_share, wilson, nearest, DEF_FR

FR_DIR = "results/free/n0_30b/h2"
OUT = "results/qa/n0-30b"
H2 = (3995.0, 6899.0)


def ball_px(fr, ts, t):
    b = fr[nearest(ts, t)].get("ball") or {}
    return b.get("px")


def make_view(img, px, cv2, zoom=2.2, w_full=1280):
    """Full frame scaled to w_full wide on top, a zoom around px (or the frame centre) below."""
    import numpy as np
    h, w = img.shape[:2]
    full = cv2.resize(img, (w_full, int(h * w_full / w)))
    cx, cy = (px if px else (w / 2, h / 2))
    zw, zh = int(w_full / zoom), int(w_full / zoom / 2)
    x0 = int(min(max(cx - zw / 2, 0), max(w - zw, 0))); y0 = int(min(max(cy - zh / 2, 0), max(h - zh, 0)))
    z = cv2.resize(img[y0:y0 + zh, x0:x0 + zw], (w_full, int(zh * w_full / zw)))
    return np.vstack([full, z])


def views(frames_json=os.path.join(FR_DIR, "times.json"), out=os.path.join(OUT, "views")):
    import cv2
    fr = json.load(gzip.open(DEF_FR))["frames"]; ts = [f["t"] for f in fr]
    meta = json.load(open(frames_json)); os.makedirs(out, exist_ok=True)
    for f in meta["frames"]:
        img = cv2.imread(os.path.join(os.path.dirname(frames_json), f["file"]))
        cv2.imwrite(os.path.join(out, f["file"]), make_view(img, ball_px(fr, ts, f["t"]), cv2), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(len(meta["frames"]), "views ->", out)


def compare(grades_path=os.path.join(OUT, "aik_h2_grades.json"), out_path=os.path.join(OUT, "result.json")):
    fr = json.load(gzip.open(DEF_FR))["frames"]
    G = json.load(open(grades_path))
    res, items = analyse(fr, G["items"], half=H2)
    res["export_second_half_sfk_pct"] = res.pop("export_first_half_sfk_pct")
    # pooled with N0-30a's first half (same rules, same grader) for a whole-match by-eye share
    g1 = "results/qa/n0-30a/aik_poss_grades.json"
    if os.path.exists(g1):
        d1 = [x["eye"] for x in json.load(open(g1))["items"] if x["eye"] in ("A", "B")]
        d2 = [x["eye"] for x in G["items"] if x["eye"] in ("A", "B")]
        k, n = sum(v == "A" for v in d1 + d2), len(d1 + d2)
        res["pooled_both_halves"] = {"eye_sfk": k, "eye_decided": n, "eye_sfk_pct": round(100 * k / n) if n else None,
                                     "eye_ci95": wilson(k, n)}
    res["misses"] = [{"file": x["file"], "t": x["t"], "eye": x["eye"], "ours": x["ours"], "ours_win": x["ours_win"]}
                     for x in items if x["eye"] in ("A", "B") and x["ours"] in ("A", "B") and x["eye"] != x["ours"]]
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    views() if sys.argv[1:2] == ["views"] else compare()
