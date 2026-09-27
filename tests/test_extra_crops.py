"""28 Sep: other-match labels (trainset_clicks format) -> crops with a per-match prefix; ball box inside the crop"""
import os, sys, json, tempfile, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ballclicks as BC
d = tempfile.mkdtemp(); v = f"{d}/v.mp4"; w = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
for k in range(400): f = np.full((1080, 1920, 3), 60, np.uint8); cv2.circle(f, (300 + 3 * k, 700), 6, (255, 255, 255), -1); w.write(f)
w.release()
labs = [{"frame": k, "x": 300 + 3 * k, "y": 700, "conf": 0.5} for k in range(0, 390, 3)]
tj = f"{d}/l.json"; json.dump({"labels": [[l["frame"], l["x"], l["y"], l["conf"]] for l in labs]}, open(tj, "w"))
ds = f"{d}/ds"; [os.makedirs(f"{ds}/{a}/train", exist_ok=True) for a in ("images", "labels")]
n = BC.add_auto_crops(tj, v, ds, n_auto=len(labs), log=print, prefix="x_vasalund")
files = sorted(os.listdir(f"{ds}/labels/train")); assert n == 2 * len(labs) and all(f.startswith("x_vasalund_f") for f in files), (n, files[:3])
pos = [open(f"{ds}/labels/train/{f}").read().split() for f in files if os.path.getsize(f"{ds}/labels/train/{f}")]
assert len(pos) == len(labs) and all(0 < float(p[1]) < 1 and 0 < float(p[2]) < 1 for p in pos)
im = cv2.imread(f"{ds}/images/train/" + files[0].replace(".txt", ".jpg")); p = open(f"{ds}/labels/train/{files[0]}").read().split()
cx, cy = int(float(p[1]) * 640), int(float(p[2]) * 640); assert im[cy, cx].mean() > 200, "ball pixel must be white"
print("OK", n)
