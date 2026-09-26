"""Teach the line network a new ground (26 Sep). Daniel's pitch-point clicks on a few frames give the camera and exact
poses (venueclicks); those poses render exact line masks, like Edsberg's. A held-back share of the clicked frames is the
exam: predicted lines -> cold pose fit under the venue's camera -> pixel distance to Daniel's clicks."""
import os, json, shutil, numpy as np, cv2
from . import lines as LN

def build(venue_dir, sol, out_ds, val_every=3, copies=6, size=(640, 360), near_m=None, log=print):
    """venue_dir has frames/<name>.jpg (1280x720) and points.json (Daniel's clicks). Writes a dataset dir in the same layout
    as lines.build_dataset (its own poses.json with this venue's camera), train frames repeated `copies` times so a few
    clicked frames carry weight against Edsberg's hundreds (augmentation makes each copy different)."""
    clicks = json.load(open(f"{venue_dir}/points.json")); cam = sol["camera"]; w, h = size; poses = {}; n = {"train": 0, "val": 0}
    for sub in ("images/train", "images/val", "masks/train", "masks/val", "images_full/val"): os.makedirs(f"{out_ds}/{sub}", exist_ok=True)
    for i, f in enumerate(sol["frames"]):
        name = f["frame"]; split = "val" if i % val_every == val_every - 1 else "train"; img = cv2.imread(f"{venue_dir}/frames/{name}")
        if img is None: continue
        if split == "val": cv2.imwrite(f"{out_ds}/images_full/val/venue_{name[:-4]}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        small = cv2.resize(img, size, interpolation=cv2.INTER_AREA); m = LN.render_mask(cam, f["pose"], w, h, near_m=near_m)
        for c in range(copies if split == "train" else 1):
            key = f"venue_{name[:-4]}" + (f"_c{c}" if split == "train" else "")
            cv2.imwrite(f"{out_ds}/images/{split}/{key}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92]); cv2.imwrite(f"{out_ds}/masks/{split}/{key}.png", m)
            poses[key] = {"pose": [float(v) for v in f["pose"]], "split": split, "source": "venue", "frame": name, "clicks": clicks.get(name, {}).get("pairs")}
        n[split] += 1
    json.dump({"camera": cam, "size": list(size), "classes": LN.CLASSES, "poses": poses}, open(f"{out_ds}/poses.json", "w"))
    log(f"  venue dataset: {n['train']} clicked frames x{copies} for training, {n['val']} held back for the exam")
    return n

def merge_train(src_ds, dst_ds):
    """copy src's training images/masks into dst (the two venues train together; each keeps its own exam)"""
    k = 0
    for sub in ("images", "masks"):
        for fn in os.listdir(f"{src_ds}/{sub}/train"): shutil.copy(f"{src_ds}/{sub}/train/{fn}", f"{dst_ds}/{sub}/train/{fn}"); k += 1
    return k // 2
