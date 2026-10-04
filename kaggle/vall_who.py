# Kaggle (free, 4 Oct): who-has-the-ball key for Vallentuna, graded blind: 40 random in-play moments, each as the full frame at t
# (1280x720) with two small frames 0.5 s before and after underneath. No app markers. Video from R2.
import json, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/vall/who_moments.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5)
def fr(t, w, h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    return cv2.resize(f, (w, h)) if ok else np.zeros((h, w, 3), np.uint8)
for j, t in enumerate(D["moments"]):
    big = fr(t, 1280, 720); small = np.hstack([fr(t - 0.5, 640, 360), fr(t + 0.5, 640, 360)])
    g = np.vstack([big, small]); cv2.putText(g, f"w{j:02d} t={t:.1f}s (top = t, bottom = -0.5 s | +0.5 s)", (8, 28), 0, 0.8, (255, 255, 255), 2)
    cv2.imwrite(f"{W}/w{j:02d}.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(j, flush=True)
json.dump({"ok": True, "n": len(D["moments"]), "fps": fps}, open(f"{W}/result.json", "w")); print("done")
