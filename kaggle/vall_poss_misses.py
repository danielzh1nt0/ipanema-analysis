# Kaggle (free, 4 Oct): the Vallentuna possession misses drawn: true ball (green ring), app ball (yellow cross), exported players
# (red dot = A/SFK, blue dot = B/Vallentuna) on the full frame, plus a 2x zoom around the true ball.
import json, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/vall/poss_misses.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5)
for m in D["moments"]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(m["t"] * fps))); ok, f = cap.read(); f = cv2.resize(f, (1920, 1080))
    for x, y, t in m["players"]: cv2.circle(f, (int(x), int(y)), 7, (0, 0, 255) if t == "A" else (255, 120, 0), -1)
    gx, gy = m["true_ball"]; cv2.circle(f, (int(gx), int(gy)), 18, (0, 255, 0), 2)
    if m["app_ball"]: cv2.drawMarker(f, (int(m["app_ball"][0]), int(m["app_ball"][1])), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 26, 3)
    R = 200; x0, y0 = int(min(max(gx - R, 0), 1920 - 2 * R)), int(min(max(gy - R, 0), 1080 - 2 * R)); z = cv2.resize(f[y0:y0 + 2 * R, x0:x0 + 2 * R], (800, 800))
    big = cv2.resize(f, (1422, 800)); g = np.hstack([big, z])
    cv2.putText(g, f"{m['id']} t={m['t']:.1f} owner(eye)={m['owner']} app={m['app_poss']}", (10, 40), 0, 1.2, (255, 255, 255), 3)
    cv2.imwrite(f"{W}/{m['id']}.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(m["id"], flush=True)
json.dump({"ok": True}, open(f"{W}/result.json", "w"))
