# Kaggle (free, 4 Oct): by-eye check of SHORT possession spells (1.5-3 s) in the SFK-BP first half: is the team really taking the
# ball, or is it the ball dot jumping between players? One strip per spell: 5 frames (1.5 s before the spell, its start, middle,
# end, 1.5 s after), time + team written on each. 12 longer spells as control. Reads the match video from R2 (no download).
import json, cv2, urllib.request, os, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/spells/moments.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5); n = 0
for s in D["spells"]:
    ts = [s["t_start"] - 1.5, s["t_start"], (s["t_start"] + s["t_end"]) / 2, s["t_end"], s["t_end"] + 1.5]; tiles = []
    for j, t in enumerate(ts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
        if not ok: f = np.zeros((1080, 1920, 3), np.uint8)
        f = cv2.resize(f, (768, 432)); lab = ["before", "start", "middle", "end", "after"][j]
        cv2.rectangle(f, (0, 0), (768, 22), (0, 0, 0), -1); cv2.putText(f, f"{s['id']} {lab} t={t:.1f}s  spell: team {s['team']} {s['dur']}s ({s['before']} before, {s['after']} after)", (4, 16), 0, 0.5, (255, 255, 255), 1)
        tiles.append(f)
    cv2.imwrite(f"{W}/{s['id']}.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82]); n += 1; print(s["id"], flush=True)
json.dump({"ok": True, "strips": n, "fps": fps}, open(f"{W}/result.json", "w")); print("done", n)
