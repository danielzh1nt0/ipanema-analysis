"""1 Oct (Daniel: "fix ball"): picture sheets of the moments the CURRENT app picker still gets wrong on both keyed clips
(AIK 39 blind key, SFK-BP 34), from the exact app inputs (results/volume/cache/<clip>/picker_inputs.pkl). Free runner.
Per miss: left = around the true ball (green ring), middle = around our pick (red cross), right = whole frame;
yellow = finder guesses with their score, blue dots = player feet; a strip of the pick 1 s before / after on the whole frame.
Output: results/qa/ballmiss2/<clip>_*.jpg + misses.json. Claude sorts the misses by eye into causes."""
import os, sys, json, pickle, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL
q = lambda *a: None
CLIPS = {"p15u-vs-aik-2026-09-21-bd09_s2520": "reference/p15u-vs-aik-2026-09-21-bd09_s2520/ball_gt.json",
         "SFKBP1109_s1200": "results/volume/reference/SFKBP1109_s1200/ball_gt.json"}
OUT = os.environ.get("OUT", "results/qa/ballmiss2"); os.makedirs(OUT, exist_ok=True)
R2 = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
def load(m):
    P = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    P["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P["per"].items()}
    return P
def crop(f, x, y, s=100, out=320):
    """2s x 2s pixels around (x, y), enlarged to out x out"""
    h, w = f.shape[:2]; x0 = int(min(max(x - s, 0), w - 2 * s)); y0 = int(min(max(y - s, 0), h - 2 * s))
    return cv2.resize(f[y0:y0 + 2 * s, x0:x0 + 2 * s], (out, out), interpolation=cv2.INTER_CUBIC)
def misses(P, gt, kw=None):
    b = BL.bridge(BL.pick_v2(P["cands"], P["H"], P["L"], P["W"], per=P["per"], fps=P["fps"], log=q, **(kw or {})), P["fps"])
    out = []
    for f, g in sorted(gt.items()):
        p = b.get(f)
        if p is None or np.hypot(p[0] - g[0], p[1] - g[1]) > 30: out.append((f, g[:2], p))
    return b, out
if __name__ == "__main__":
    report = {}
    for clip, gtp in CLIPS.items():
        gt = {int(k): v for k, v in json.load(open(gtp)).items() if v}
        P = load(clip); b, ms = misses(P, gt); report[clip] = {"right": len(gt) - len(ms), "of": len(gt), "misses": []}
        src = os.environ.get(f"LOCAL_{clip}") or f"{R2}/{clip}/video.mp4"; cap = cv2.VideoCapture(src); fps = P["fps"]; d = int(round(fps))
        print(clip, "frames", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "misses", len(ms), flush=True); tiles = []
        def frame(k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); return f if ok else None
        def draw(f, k, truth=None):
            f = f.copy()
            for r in P["per"].get(k, []):
                if r[3] is not None: cv2.circle(f, (int(r[3][0]), int(r[3][1])), 5, (255, 120, 0), -1)
            for x, y, c in sorted(P["cands"].get(k, []), key=lambda z: -z[2])[:12]:
                cv2.circle(f, (int(x), int(y)), 9, (0, 255, 255), 1); cv2.putText(f, f"{c:.2f}", (int(x) + 8, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if k in b: cv2.drawMarker(f, (int(b[k][0]), int(b[k][1])), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            if truth: cv2.circle(f, (int(truth[0]), int(truth[1])), 18, (0, 255, 0), 2)
            return f
        for f, g, p in ms:
            fr = frame(f)
            if fr is None: print("no frame", f); continue
            im = draw(fr, f, g); top = [crop(im, *g), crop(im, *(p if p else g)), cv2.resize(im, (int(320 * im.shape[1] / im.shape[0]), 320))]
            strip = []
            for dk in (-d, -d // 2, d // 2, d):
                fk = frame(f + dk); strip.append(cv2.resize(draw(fk, f + dk), (int(240 * fk.shape[1] / fk.shape[0]), 240)) if fk is not None else np.zeros((240, 427, 3), np.uint8))
            row1 = np.hstack(top); row2 = np.hstack(strip)
            W = max(row1.shape[1], row2.shape[1]); pad = lambda r: np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)])
            t = np.vstack([pad(row1), pad(row2)])
            dist = None if p is None else int(np.hypot(p[0] - g[0], p[1] - g[1]))
            cv2.putText(t, f"{clip[:14]} f{f}  pick {dist} px off  (strip: -1s -0.5s +0.5s +1s)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(f"{OUT}/{clip[:14]}_f{f}.jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 85])
            report[clip]["misses"].append({"frame": f, "truth": [round(g[0]), round(g[1])], "pick": p and [round(p[0]), round(p[1])], "px_off": dist})
        cap.release()
    json.dump(report, open(f"{OUT}/misses.json", "w"), indent=1); print(json.dumps({k: (v["right"], v["of"]) for k, v in report.items()}))
