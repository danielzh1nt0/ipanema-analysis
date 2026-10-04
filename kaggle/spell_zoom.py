# Kaggle (free, 4 Oct): zoomed strips for the possession-spell check. For each spell 5 moments (1.5 s before, start, middle, end,
# 1.5 s after); each a 480x480 crop of the full-res video frame around the exported ball position (yellow ring = our ball,
# red/blue dots = our players A/B, from results/review/spells/zoom.json), upscaled x1.5. Reads the video from R2.
import json, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/spells/zoom.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5); n = 0; R = 240
for s in D["spells"]:
    tiles = []
    for m in s["moments"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(m["t"] * fps))); ok, f = cap.read()
        if not ok: f = np.zeros((1080, 1920, 3), np.uint8)
        f = cv2.resize(f, (1920, 1080)); bx, by = m.get("ball") or (960, 540)
        for p in m.get("players", []): cv2.circle(f, (int(p[0]), int(p[1])), 6, (0, 0, 255) if p[2] == "A" else (255, 80, 0), -1)
        if m.get("ball"): cv2.circle(f, (int(bx), int(by)), 14, (0, 255, 255), 2)
        x0, y0 = int(min(max(bx - R, 0), 1920 - 2 * R)), int(min(max(by - R, 0), 1080 - 2 * R)); c = cv2.resize(f[y0:y0 + 2 * R, x0:x0 + 2 * R], (720, 720))
        cv2.rectangle(c, (0, 0), (720, 22), (0, 0, 0), -1); cv2.putText(c, f"{s['id']} {m['label']} t={m['t']:.1f}s poss={m.get('poss')} | spell {s['team']} {s['dur']}s ({s['before']}>{s['team']}>{s['after']})", (4, 16), 0, 0.45, (255, 255, 255), 1)
        tiles.append(c)
    cv2.imwrite(f"{W}/{s['id']}.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85]); n += 1; print(s["id"], flush=True)
json.dump({"ok": True, "strips": n}, open(f"{W}/result.json", "w")); print("done", n)
