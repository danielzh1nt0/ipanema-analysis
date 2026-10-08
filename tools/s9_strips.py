"""S9 (8 Oct, free runner): blind picture strips for the 'is this lost ball real or a duel' check.
For each item in results/possession/s9/blind_key.json: 6 moments (1 s before the change, the change, +1, +2, +3, +4.5 s),
each a 480x480 crop of the full-res frame around our exported ball (yellow ring = our ball, red/blue dots = our players
A/B), upscaled to 600. The strip title shows only the id, time and who we say lost it - never the rule's verdict.
    python tools/s9_strips.py            (free runner: video from R2_PUBLIC_URL)
    VIDEO=/path/to.mp4 OUT=dir python tools/s9_strips.py   (dry run on a local file)
-> results/qa/s9/<id>.jpg + results/qa/s9/result.json"""
import os, json, glob, bisect, cv2, numpy as np

KEY = "results/possession/s9/blind_key.json"
OFFS = [(-1.0, "before"), (0.0, "change"), (1.0, "+1s"), (2.0, "+2s"), (3.0, "+3s"), (4.5, "+4.5s")]

def load_frames(match):
    fr = []
    for f in sorted(glob.glob(f"results/volume/runs/matches/{match}/frames_*.json")): fr += json.load(open(f))["frames"]
    return fr, [f["t"] for f in fr]

def at(fr, ts, t):
    i = min(bisect.bisect_left(ts, t), len(ts) - 1); f = fr[i]; b = f.get("ball")
    return {"ball": [b["px"][0], b["px"][1]] if b and b.get("px") else None,
            "players": [[p["px"][0], p["px"][1], p.get("team")] for p in f.get("players", []) if p.get("px")]}

def tile(cap, fps, t, m, title, R=240, S=600):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    if not ok: f = np.zeros((1080, 1920, 3), np.uint8)
    f = cv2.resize(f, (1920, 1080)); bx, by = m["ball"] or (960, 540)
    for p in m["players"]: cv2.circle(f, (int(p[0]), int(p[1])), 6, (0, 0, 255) if p[2] == "A" else (255, 80, 0), -1)
    if m["ball"]: cv2.circle(f, (int(bx), int(by)), 14, (0, 255, 255), 2)
    x0, y0 = int(min(max(bx - R, 0), 1920 - 2 * R)), int(min(max(by - R, 0), 1080 - 2 * R)); c = cv2.resize(f[y0:y0 + 2 * R, x0:x0 + 2 * R], (S, S))
    cv2.rectangle(c, (0, 0), (S, 22), (0, 0, 0), -1); cv2.putText(c, title, (4, 16), 0, 0.45, (255, 255, 255), 1)
    return c

def main():
    K = json.load(open(KEY)); out = os.environ.get("OUT", "results/qa/s9"); os.makedirs(out, exist_ok=True)
    src = os.environ.get("VIDEO") or f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{K['src_key']}"
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; fr, ts = load_frames(K["match"]); n = 0
    print("video", "ok" if cap.isOpened() else "NOT OPEN", fps, "frames", len(fr), flush=True)
    for it in K["items"]:
        t0 = it["t"]; tiles = []
        for dt, lab in OFFS:
            t = t0 + dt; tiles.append(tile(cap, fps, t, at(fr, ts, t), f"{it['id']} {lab} t={t:.1f}s | we say {it['lost_by']} lost to {it['won_by']}"))
        cv2.imwrite(f"{out}/{it['id']}.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85]); n += 1; print(it["id"], flush=True)
    json.dump({"ok": n == len(K["items"]), "strips": n, "fps": fps}, open(f"{out}/result.json", "w")); print("done", n)

if __name__ == "__main__":
    main()
