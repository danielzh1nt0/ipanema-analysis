"""H1: look at an outside ball dataset (e.g. martinjolif/football-ball-detection on Hugging Face) before using it.

Reads the three layouts such datasets come in, without knowing which one in advance:
  - YOLO folders:  .../images/x.jpg + .../labels/x.txt  ("cls cx cy w h", 0-1)
  - COCO json:     _annotations.coco.json / *.json with images + annotations (bbox = x, y, w, h in px)
  - parquet:       rows with an image (bytes or {"bytes": ...}) and objects {"bbox": [...], "category": [...]}
Every item is (split, image-loader, [ (x, y, w, h) px boxes of the ball ]).
Then: stats (image sizes, ball size in px and as a share of the image width, next to our Veo ball) and picture sheets:
full images with the ball boxed + zoomed ball crops, and a strip of our own Veo balls at the same zoom for comparison."""
import os, io, json, glob, random
import numpy as np, cv2

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VEO_W = 1920
VEO_BALL_PX = (14.0, 18.0)          # training box on our 1920-wide Veo frames (far / near), kaggle/ballfinder_rfdetr.py


def _split_of(path, root):
    parts = os.path.relpath(path, root).replace("\\", "/").lower().split("/")
    for p in parts:
        for s in ("train", "valid", "val", "test"):
            if p == s or p.startswith(s + "-") or p.startswith(s + "_") or p.startswith(s + "."):
                return "valid" if s == "val" else s
    return "all"


def _decode(b):
    a = np.frombuffer(b, np.uint8); return cv2.imdecode(a, cv2.IMREAD_COLOR)


def ball_ids(names):
    """Category ids that mean the ball. names: {id: name}. If no name says ball and there is one class, that one."""
    ids = {int(k) for k, v in names.items() if "ball" in str(v).lower()}
    if not ids and len(names) == 1: ids = {int(next(iter(names)))}
    return ids


def read_yolo(root):
    items = []; names = {}
    for y in glob.glob(f"{root}/**/*.yaml", recursive=True):              # data.yaml: names: [..] or {0: ..}
        try:
            txt = open(y).read()
            if "names" in txt:
                import re
                m = re.search(r"names:\s*\[([^\]]*)\]", txt)
                if m: names = {i: s.strip().strip("'\"") for i, s in enumerate(m.group(1).split(","))}
                else:
                    for i, n in re.findall(r"^\s*(\d+)\s*:\s*(.+)$", txt, re.M): names[int(i)] = n.strip().strip("'\"")
        except Exception: pass
    bid = ball_ids(names) if names else None
    for lab in glob.glob(f"{root}/**/labels/**/*.txt", recursive=True):
        stem = os.path.splitext(os.path.basename(lab))[0]
        imdir = os.path.dirname(lab).replace("/labels", "/images")
        img = next((f"{imdir}/{stem}{e}" for e in IMG_EXT if os.path.exists(f"{imdir}/{stem}{e}")), None)
        if img is None: continue
        rows = [r.split() for r in open(lab).read().strip().splitlines() if r.strip()]
        rel = [(int(float(r[0])), *map(float, r[1:5])) for r in rows if len(r) >= 5]
        items.append({"split": _split_of(img, root), "path": img, "yolo": rel, "ball_ids": bid})
    return items, names


def read_coco(root):
    items = []; names = {}
    for js in glob.glob(f"{root}/**/*.json", recursive=True):
        try: d = json.load(open(js))
        except Exception: continue
        if not isinstance(d, dict) or "images" not in d or "annotations" not in d: continue
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}; names.update(cats)
        bid = ball_ids(cats)
        by = {}
        for a in d["annotations"]:
            if a.get("category_id") in bid: by.setdefault(a["image_id"], []).append(tuple(map(float, a["bbox"])))
        for im in d["images"]:
            p = os.path.join(os.path.dirname(js), im["file_name"])
            if os.path.exists(p): items.append({"split": _split_of(js, root), "path": p, "boxes": by.get(im["id"], []), "size": (im.get("width"), im.get("height"))})
    return items, names


def read_parquet(root):
    import pandas as pd
    items = []; names = {}
    for pq in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
        df = pd.read_parquet(pq); split = _split_of(pq, root)
        icol = next((c for c in df.columns if c.lower() in ("image", "img", "images")), None)
        ocol = next((c for c in df.columns if c.lower() in ("objects", "annotations", "labels", "bboxes")), None)
        if icol is None: continue
        for _, r in df.iterrows():
            v = r[icol]; b = v.get("bytes") if isinstance(v, dict) else v
            boxes = []; o = r[ocol] if ocol else None
            if isinstance(o, dict):
                bb = o.get("bbox", o.get("bboxes", [])); cc = o.get("category", o.get("label", [0] * len(bb)))
                for box, c in zip(list(bb), list(cc)): boxes.append((tuple(float(t) for t in box), int(c)))
            items.append({"split": split, "bytes": b, "pboxes": boxes})
    # xywh (COCO, the HF default) vs xyxy: a small ball in xywh has w < x almost always; in xyxy x2 > x1 always
    bx = [bb for it in items for bb, _ in it["pboxes"]]
    xyxy = bool(bx) and np.mean([c > a and d > b for a, b, c, d in bx]) > 0.95 and np.mean([(c - a) < a for a, b, c, d in bx]) > 0.5
    for it in items: it["xyxy"] = xyxy
    return items, names


def load(root):
    """All items from whatever layout is under root. Box format for parquet (xywh vs xyxy) is decided from the data."""
    for fn in (read_yolo, read_coco, read_parquet):
        try: items, names = fn(root)
        except ImportError: continue
        if items: return fn.__name__[5:], items, names
    return None, [], {}


def image(it):
    return cv2.imread(it["path"]) if "path" in it else _decode(it["bytes"])


def balls(it, im, names=None):
    """Ball boxes (x, y, w, h) in px for one item."""
    h, w = im.shape[:2]
    if "yolo" in it:
        bid = it["ball_ids"] if it["ball_ids"] else {r[0] for r in it["yolo"]}
        return [((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h) for c, cx, cy, bw, bh in it["yolo"] if c in bid]
    if "boxes" in it: return list(it["boxes"])
    out = []; bid = ball_ids(names) if names else None
    for (a, b, c, d), cat in it["pboxes"]:
        if bid and cat not in bid: continue
        if max(a, b, c, d) <= 1.5: a, b, c, d = a * w, b * h, c * w, d * h                          # normalised
        if it.get("xyxy"): c, d = c - a, d - b
        out.append((a, b, c, d))
    return out


def stats(rows):
    """rows: [{split, w, h, balls: [(x,y,bw,bh)]}] -> plain summary."""
    sz = [max(b[2], b[3]) for r in rows for b in r["balls"]]
    rel = [max(b[2], b[3]) / r["w"] for r in rows for b in r["balls"]]
    veo = [v / VEO_W for v in VEO_BALL_PX]
    wh = {}
    for r in rows: wh[f"{r['w']}x{r['h']}"] = wh.get(f"{r['w']}x{r['h']}", 0) + 1
    q = lambda a, p: round(float(np.percentile(a, p)), 4) if len(a) else None
    return {"images": len(rows), "by_split": {s: sum(r["split"] == s for r in rows) for s in sorted({r["split"] for r in rows})},
            "images_with_ball": sum(bool(r["balls"]) for r in rows), "balls": len(sz),
            "image_sizes_top": dict(sorted(wh.items(), key=lambda kv: -kv[1])[:5]),
            "ball_px": {"p10": q(sz, 10), "median": q(sz, 50), "p90": q(sz, 90)},
            "ball_share_of_width": {"p10": q(rel, 10), "median": q(rel, 50), "p90": q(rel, 90)},
            "veo_ball_share_of_width": [round(v, 4) for v in veo],
            "share_as_small_as_veo": round(float(np.mean([x <= veo[1] * 1.5 for x in rel])), 3) if rel else None}


def crop_zoom(im, x, y, half=40, out=120):
    h, w = im.shape[:2]; x0 = int(min(max(x - half, 0), max(w - 2 * half, 0))); y0 = int(min(max(y - half, 0), max(h - 2 * half, 0)))
    c = im[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if c.size == 0: return np.zeros((out, out, 3), np.uint8)
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_CUBIC)


def tile(im, bxs, label, W=480, H=270):
    """Full image (letterboxed to W x H, ball boxed) + zoomed crop of the first ball at the image's own scale
    rescaled as if the image were 1920 wide (so the crop compares with our Veo strip)."""
    h, w = im.shape[:2]; s = min(W / w, H / h); r = cv2.resize(im, (int(w * s), int(h * s)))
    full = np.zeros((H, W, 3), np.uint8); full[:r.shape[0], :r.shape[1]] = r
    for (x, y, bw, bh) in bxs: cv2.rectangle(full, (int(x * s) - 3, int(y * s) - 3), (int((x + bw) * s) + 3, int((y + bh) * s) + 3), (0, 255, 255), 2)
    cv2.putText(full, label, (5, 20), 0, 0.55, (0, 0, 0), 3); cv2.putText(full, label, (5, 20), 0, 0.55, (255, 255, 255), 1)
    if bxs:
        x, y, bw, bh = bxs[0]; k = VEO_W / w                                   # same pixels-per-pitch-width idea as a Veo frame
        z = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        zc = crop_zoom(z, (x + bw / 2) * k, (y + bh / 2) * k, 40, H)
    else: zc = np.zeros((H, H, 3), np.uint8)
    return np.hstack([full, zc])


def sheets(tiles, out_dir, prefix="sheet", per=12, cols=2):
    os.makedirs(out_dir, exist_ok=True); files = []
    for s in range(0, len(tiles), per):
        ts = tiles[s:s + per]
        while len(ts) % cols: ts.append(np.zeros_like(ts[0]))
        g = np.vstack([np.hstack(ts[i:i + cols]) for i in range(0, len(ts), cols)])
        f = f"{out_dir}/{prefix}_{s // per:02d}.jpg"; cv2.imwrite(f, g, [cv2.IMWRITE_JPEG_QUALITY, 82]); files.append(f)
    return files


def veo_strip(exam_dir, rows, n=8, H=270):
    """Our own Veo balls at the same zoom (exam jpgs are half size of 1920; truth is in 1920 px)."""
    ts = []
    for r in [r for r in rows if r.get("truth") and os.path.exists(f"{exam_dir}/{r['file']}")][:n]:
        p = f"{exam_dir}/{r['file']}"
        im = cv2.resize(cv2.imread(p), (VEO_W, 1080), interpolation=cv2.INTER_CUBIC)
        c = crop_zoom(im, r["truth"][0], r["truth"][1], 40, H); cv2.putText(c, "Veo", (4, 18), 0, 0.5, (255, 255, 255), 1); ts.append(c)
    return np.hstack(ts) if ts else None


def run(root, out_dir, exam_dir=None, exam_rows=None, sample=48, seed=0):
    kind, items, names = load(root)
    if not items: raise SystemExit(f"no images found under {root}")
    rows = []; rng = random.Random(seed); pick = set(rng.sample(range(len(items)), min(sample, len(items)))); tiles = []
    for i, it in enumerate(items):
        im = image(it)
        if im is None: continue
        b = balls(it, im, names); rows.append({"split": it["split"], "w": im.shape[1], "h": im.shape[0], "balls": b})
        if i in pick: tiles.append(tile(im, b, f"{i} {it['split']} {im.shape[1]}x{im.shape[0]} balls={len(b)}"))
    st = stats(rows); st["layout"] = kind; st["class_names"] = {str(k): v for k, v in names.items()}
    os.makedirs(out_dir, exist_ok=True)
    st["sheets"] = sheets(tiles, out_dir)
    if exam_dir and exam_rows:
        v = veo_strip(exam_dir, exam_rows)
        if v is not None: cv2.imwrite(f"{out_dir}/veo_reference.jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 85]); st["veo_reference"] = f"{out_dir}/veo_reference.jpg"
    json.dump(st, open(f"{out_dir}/stats.json", "w"), indent=1)
    return st
