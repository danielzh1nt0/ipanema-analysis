"""F1 (17 Oct, free GitHub runner): picture sheet of the "who has the ball" moments a Kaggle export gets wrong - the frame from R2
with the exported players (red dot = A, blue = B, grey = other; hollow = gap-filled) and the exported ball (yellow cross), plus
a 2x zoom around the ball. Reads results/kaggle/<name>/frames_lite.json.gz (3 frames/s). No Modal, no hand tagging.
    R2_PUBLIC_URL=... python tools/poss_sheet.py <results/kaggle/name> <match_id> [all]  -> results/qa/poss/<name>/*.jpg + index.json"""
import os, sys, json, gzip, bisect, cv2, numpy as np
MAX_DT = 1.0   # s: a key moment further than this from every exported frame is skipped (17 Oct worker: was compared with the nearest, e.g. the last frame)

def main():
    src, M = sys.argv[1], sys.argv[2]; want_all = "all" in sys.argv[3:]; name = os.path.basename(src.rstrip("/")); out = f"results/qa/poss/{name}"; os.makedirs(out, exist_ok=True)
    d = json.load(gzip.open(f"{src}/frames_lite.json.gz")); fr = d["frames"]; ts = [f["t"] for f in fr]
    if not fr: sys.exit("export has no frames")
    ref = f"reference/{M}"; items = []
    for fn in ("ball_key_graded.json", "poss_sample.json"):
        if os.path.exists(f"{ref}/{fn}"): items += [it for it in json.load(open(f"{ref}/{fn}"))["items"] if it.get("owner") in ("A", "B")]
    cap = cv2.VideoCapture(f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{M}/video.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    if not cap.isOpened(): sys.exit("video not reachable")
    rows = []; skipped = 0
    for it in sorted(items, key=lambda x: x["t"]):
        i = bisect.bisect_left(ts, it["t"]); j = min([k for k in (i - 1, i) if 0 <= k < len(ts)], key=lambda k: abs(ts[k] - it["t"])); f = fr[j]
        if abs(f["t"] - it["t"]) > MAX_DT: skipped += 1; continue   # key moment outside the exported time: no frame to compare
        ours = f.get("possession"); miss = ours != it["owner"]
        if not (miss or want_all): continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f["t"] * fps))); ok, img = cap.read()
        if not ok: continue
        img = cv2.resize(img, (1920, 1080))
        for p in f["players"]:
            x, y = p["px"]; col = (0, 0, 255) if p["team"] == "A" else (255, 120, 0) if p["team"] == "B" else (160, 160, 160)
            filled = p.get("state") == "filled"; cv2.circle(img, (int(x), int(y)), 9, col, 2 if filled else -1); cv2.putText(img, str(p["id"]), (int(x) + 10, int(y) - 6), 0, 0.5, col, 2)
        b = f.get("ball"); bx = by = None
        if b: bx, by = b["px"]; cv2.drawMarker(img, (int(bx), int(by)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
        cx, cy = (bx, by) if b else (960, 540); R = 220; x0, y0 = int(min(max(cx - R, 0), 1920 - 2 * R)), int(min(max(cy - R, 0), 1080 - 2 * R))
        z = cv2.resize(img[y0:y0 + 2 * R, x0:x0 + 2 * R], (800, 800)); big = cv2.resize(img, (1422, 800)); g = np.hstack([big, z])
        label = f"{it['id']} t={it['t']:.1f} eye={it['owner']} ours={ours} ({f.get('phase')}) ball={'none' if not b else b.get('state')} A={sum(p['team'] == 'A' for p in f['players'])} B={sum(p['team'] == 'B' for p in f['players'])}"
        cv2.rectangle(g, (0, 0), (2222, 50), (0, 0, 0), -1); cv2.putText(g, label, (10, 36), 0, 1.0, (255, 255, 255), 2)
        cv2.imwrite(f"{out}/{it['id']}.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 82]); rows.append({"id": it["id"], "t": it["t"], "eye": it["owner"], "ours": ours, "phase": f.get("phase")}); print(label, flush=True)
    json.dump(rows, open(f"{out}/index.json", "w"), indent=1); print(len(rows), "pictures ->", out, f"({skipped} key moments outside the export skipped)" if skipped else "")

if __name__ == "__main__": main()
