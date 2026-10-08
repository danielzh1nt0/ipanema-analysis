"""C3c (8 Oct): raw frames where the near touchline is well in view, for the camera-base / pitch-size check.
  python tools/c3c_frames.py pick <match> <out dir> [n]   (local) trusted seconds (linecal.brave) during play whose
        drawn near touchline crosses >= 500 px of the lower picture, spread evenly over the match -> <out>/sample.json
  python tools/c3c_frames.py grab <out dir>                (free runner) raw 1280x720 frames <out>/<id>.jpg, no drawing
        (VIDEO=/path.mp4 for a local dry run; otherwise R2_PUBLIC_URL/<src_key>)"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import lines as LN, linecal as LC

def near_len_px(camera, pose, w=1280, h=720, top=0.45):
    """visible length (px) of the drawn near touchline in the lower part of the picture"""
    S = LN.projected_segments(camera, np.asarray(pose, float), w, h)[1]; tot = 0.0
    for x0, y0, x1, y1 in S:
        ok, p1, p2 = cv2.clipLine((0, int(top * h), w, h - int(top * h)), (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))))
        if ok: tot += float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return tot

def pick(rows, camera, periods, n=40, min_px=500.0, gap_s=20.0):
    inplay = [r for r in rows if r.get("pose") is not None and LC.brave(r) and any(a + 5 <= r["t"] <= b - 5 for a, b in periods)]
    cand = [r for r in inplay if near_len_px(camera, r["pose"]) >= min_px]
    if not cand: return []
    idx = np.unique(np.linspace(0, len(cand) - 1, min(n * 3, len(cand))).round().astype(int)); out = []
    for i in idx:                                                                  # evenly spread, at least gap_s apart
        r = cand[i]
        if all(abs(r["t"] - q["t"]) >= gap_s for q in out): out.append(r)
    if len(out) > n: out = [out[i] for i in np.linspace(0, len(out) - 1, n).round().astype(int)]
    return [{"id": f"n{i:02d}", "t": r["t"], "pose": r["pose"]} for i, r in enumerate(out)]

def grab(d):
    S = json.load(open(f"{d}/sample.json"))
    src = os.environ.get("VIDEO") or f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{S['src_key']}"
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; n = 0; miss = []
    print("video", "ok" if cap.isOpened() else "NOT OPEN", fps, flush=True)
    for q in S["rows"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(q["t"] * fps))); ok, f = cap.read()
        if not ok: miss.append(q["id"]); continue
        cv2.imwrite(f"{d}/{q['id']}.jpg", cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92]); n += 1
    json.dump({"ok": n > 0, "frames": n, "missing": miss}, open(f"{d}/result.json", "w")); print(d, "frames", n, "missing", miss, flush=True)
    return n

def main(argv=sys.argv[1:]):
    if argv[0] == "pick":
        match, out = argv[1], argv[2]; n = int(argv[3]) if len(argv) > 3 else 40
        d = json.load(open(f"calibration/{match}_lines_match.json")); per = json.load(open(f"periods/{match}.json"))["periods_s"]
        s = pick(sorted(d["rows"], key=lambda r: r["t"]), d["camera"], per, n)
        os.makedirs(out, exist_ok=True)
        json.dump({"match": match, "src_key": f"{match}/video.mp4", "camera": d["camera"], "rows": s}, open(f"{out}/sample.json", "w"), indent=0)
        print(match, len(s), "frames picked")
    else:
        for d in argv[1:]: grab(d)

if __name__ == "__main__": main()
