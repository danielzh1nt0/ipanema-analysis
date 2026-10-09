"""S2 (9 Oct, free runner): blind picture sheets for 'is a set piece / restart taken here?' on the SFK-BP first half.
Each item in results/possession/s2/blind_key.json (shuffled ball-rule, motion-rule and both-rule restarts) gets one sheet:
4 full frames (t-4 s, t-1.5 s, t, t+2 s) at 960x540, nothing drawn, title = id + time only (never which rule found it).
    python tools/s2_strips.py                     (free runner: video from R2_PUBLIC_URL)
    VIDEO=/path/to.mp4 OUT=dir python tools/s2_strips.py   (dry run on a local file)
-> results/qa/s2/<id>.jpg + results/qa/s2/result.json"""
import os, json, cv2, numpy as np

KEY = "results/possession/s2/blind_key.json"
OFFS = [(-4.0, "t-4s"), (-1.5, "t-1.5s"), (0.0, "t"), (2.0, "t+2s")]


def tile(cap, fps, t, title, W=960, H=540):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    if not ok: f = np.zeros((H, W, 3), np.uint8)
    f = cv2.resize(f, (W, H)); cv2.rectangle(f, (0, 0), (W, 24), (0, 0, 0), -1)
    cv2.putText(f, title, (6, 17), 0, 0.55, (255, 255, 255), 1); return f


def main():
    K = json.load(open(KEY)); out = os.environ.get("OUT", "results/qa/s2"); os.makedirs(out, exist_ok=True)
    src = os.environ.get("VIDEO") or f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{K['src_key']}"
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; n = 0
    print("video", "ok" if cap.isOpened() else "NOT OPEN", fps, flush=True)
    for it in K["items"]:
        ts = [tile(cap, fps, it["t"] + dt, f"{it['id']} {lab} ({it['t'] + dt:.1f}s)") for dt, lab in OFFS]
        sheet = np.vstack([np.hstack(ts[:2]), np.hstack(ts[2:])])
        cv2.imwrite(f"{out}/{it['id']}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 82]); n += 1; print(it["id"], flush=True)
    json.dump({"ok": n == len(K["items"]), "sheets": n, "fps": fps}, open(f"{out}/result.json", "w")); print("done", n)


if __name__ == "__main__":
    main()
