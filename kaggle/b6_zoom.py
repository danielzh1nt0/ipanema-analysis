# Kaggle (free, 3 Oct, B6 stage 2): the far band (y 380-700) of each of the 40 moments at 2x, in two halves (x 0-1000, 920-1920),
# with a 50-px grid labelled in ORIGINAL pixels, so a 10-px ball at night can be found and read by eye.
import json, cv2, urllib.request
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/b6/moments.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5); off = int(round(D["offset_s"] * fps)); n = 0
Y0, Y1 = 380, 700
for j, k in enumerate(D["frames"]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, off + k); ok, f = cap.read()
    if not ok: continue
    f = cv2.resize(f, (1920, 1080))
    for name, (x0, x1) in (("L", (0, 1000)), ("R", (920, 1920))):
        z = cv2.resize(f[Y0:Y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for x in range(x0 - x0 % 50, x1, 50):
            X = (x - x0) * 2; cv2.line(z, (X, 0), (X, z.shape[0]), (0, 255, 255) if x % 100 else (0, 0, 255), 1); cv2.putText(z, str(x), (X + 2, 12), 0, 0.4, (0, 255, 255), 1)
        for y in range(Y0, Y1, 50):
            Yy = (y - Y0) * 2; cv2.line(z, (0, Yy), (z.shape[1], Yy), (0, 255, 255) if y % 100 else (0, 0, 255), 1); cv2.putText(z, str(y), (2, Yy - 2), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(z, f"m{j:02d} {name}", (z.shape[1] - 120, z.shape[0] - 8), 0, 0.6, (255, 255, 255), 2)
        cv2.imwrite(f"{W}/m{j:02d}{name}.jpg", z, [cv2.IMWRITE_JPEG_QUALITY, 90]); n += 1
json.dump({"ok": True, "images": n}, open(f"{W}/result.json", "w")); print("done", n)
