# Kaggle (free, 5 Oct, V1b): by-eye check of each candidate strike: for every candidate moment, 4 frames (-1, -0.5, 0, +0.5 s),
# each a 640x360 view of the full frame with the finder's ball ringed (yellow) - one row per candidate, one sheet per shot.
import json, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/vall/strike_crops.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5)
rows = {}
for it in D["items"]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(it["t"] * fps))); ok, f = cap.read()
    f = cv2.resize(f, (1920, 1080)) if ok else np.zeros((1080, 1920, 3), np.uint8)
    if it["px"]: cv2.circle(f, (int(it["px"][0]), int(it["px"][1])), 28, (0, 255, 255), 3)
    g = cv2.resize(f, (640, 360)); cv2.rectangle(g, (0, 0), (330, 20), (0, 0, 0), -1)
    cv2.putText(g, f"shot {it['shot']} cand {it['cand']} t={it['t']}", (3, 14), 0, 0.45, (255, 255, 255), 1)
    rows.setdefault(it["shot"], {}).setdefault(it["cand"], []).append(g)
for s, cs in rows.items():
    sheet = np.vstack([np.hstack(v[:4] + [np.zeros((360, 640, 3), np.uint8)] * (4 - len(v[:4]))) for c, v in sorted(cs.items())])
    cv2.imwrite(f"{W}/strike_{s}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(s, flush=True)
json.dump({"ok": True}, open(f"{W}/result.json", "w"))
