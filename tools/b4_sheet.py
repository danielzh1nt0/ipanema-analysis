"""B4 (1 Oct, free runner): picture sheets of the auto ball-key candidates (results/ball/b4/candidates.json) so Claude can
check each one by eye. Per moment: close-up (2x) + wider view, yellow ticks around the guess (the ball itself is not covered).
Also saves the plain 64x64 crops (results/free/b4/crops.npz) for later training.
    R2_PUBLIC_URL=... python tools/b4_sheet.py          (or LOCAL_CLIP=clip.mp4 for a dry run)"""
import sys, os, json, urllib.request, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ballkey_auto as AK
OUT = os.environ.get("B4_OUT", "results/free/b4")   # the free job commits results/free
D = json.load(open(os.environ.get("B4_CANDS", "results/ball/b4/candidates.json")))
src = os.environ.get("LOCAL_CLIP")
if not src:
    src = "/tmp/b4_clip.mp4"; url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{D['clip']}/video.mp4"
    print("downloading", D["clip"], flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(src, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
want = {m["frame"]: m for m in D["moments"]}; tiles, crops, ids = [], [], []
cap = cv2.VideoCapture(src); i = 0; last = max(want) if want else -1
while i <= last:
    if i in want:
        ok, fr = cap.read()
        if not ok: break
        m = want[i]
        if fr.shape[1] != 1920: fr = cv2.resize(fr, (1920, 1080))     # guesses are in 1920x1080 pixels
        close = AK.tile(fr, m["x"], m["y"], half=48, scale=2, label=f"#{m['id']}")
        wide = cv2.resize(AK.tile(fr, m["x"], m["y"], half=160, scale=1), close.shape[1::-1], interpolation=cv2.INTER_AREA)
        tiles.append(np.hstack([close, wide, np.full((close.shape[0], 4, 3), 255, np.uint8)]))
        x, y = int(round(m["x"])), int(round(m["y"])); p = cv2.copyMakeBorder(fr, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=0)
        crops.append(p[y:y + 64, x:x + 64].copy()); ids.append(m["id"])
    elif not cap.grab(): break
    i += 1
print(f"{len(tiles)}/{len(want)} moments cut", flush=True)
os.makedirs(OUT, exist_ok=True); per = 24
for s in range(0, len(tiles), per):
    cv2.imwrite(f"{OUT}/sheet_{s // per:02d}.jpg", AK.sheet(tiles[s:s + per], cols=4), [cv2.IMWRITE_JPEG_QUALITY, 88])
np.savez_compressed(f"{OUT}/crops.npz", X=np.array(crops, np.uint8), ids=np.array(ids))
json.dump({"cut": len(tiles), "of": len(want), "sheets": -(-len(tiles) // per)}, open(f"{OUT}/sheets.json", "w"))
