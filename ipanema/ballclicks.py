"""Ball detector round from Daniel's clicks (25 Sep): 161 hard frames, centre clicks, 40 held back as the exam.
A click becomes a box of a known size (the ball is ~14-22 px wide at 1920 in this footage; far balls smaller). The grade
is per exam frame: ball frames need a detection within hit_px of the click; no-ball frames need no detection above conf."""
import os, json, numpy as np, cv2

def build_dataset(clicks_json, video, ds, box=18, log=print):
    d = json.load(open(clicks_json))["frames"]; cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    for sub in ("images/train", "labels/train", "images/val", "labels/val"): os.makedirs(f"{ds}/{sub}", exist_ok=True)
    n = {"train": 0, "val": 0}
    for r in d:
        split = "val" if r["split"] == "exam" else "train"
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(r["t"] * fps))); ok, f = cap.read()
        if not ok: continue
        h, w = f.shape[:2]; name = r["file"][:-4]
        cv2.imwrite(f"{ds}/images/{split}/{name}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(f"{ds}/labels/{split}/{name}.txt", "w") as fh:
            if r["x"] is not None:
                b = box * (0.6 if r["y"] < h * 0.4 else 1.0)                              # far balls are smaller
                fh.write(f"0 {r['x'] / w:.6f} {r['y'] / h:.6f} {b / w:.6f} {b / h:.6f}\n")
        n[split] += 1
    cap.release(); open(f"{ds}/data.yaml", "w").write(f"path: {ds}\ntrain: images/train\nval: images/val\nnames: ['ball']\n")
    log(f"  ball dataset: {n['train']} training frames, {n['val']} exam frames"); return n

def detect(model, img, conf=0.05, imgsz=1920, far_zoom=True, top=0.45):
    """detections (x, y, conf) at full-frame pixels; with far_zoom, a second pass on the top `top` of the frame enlarged
    2x, so a far 4 px ball is seen as 8 px (25 Sep: every exam miss was a tiny far ball)"""
    res = model(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    det = [(float((a + c) / 2), float((b + d) / 2), float(cf)) for (a, b, c, d), cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
    if far_zoom:
        h = img.shape[0]; strip = cv2.resize(img[:int(h * top)], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        r2 = model(strip, conf=conf, imgsz=imgsz * 2, verbose=False)[0]
        det += [(float((a + c) / 4), float((b + d) / 4), float(cf)) for (a, b, c, d), cf in zip(r2.boxes.xyxy.cpu().numpy(), r2.boxes.conf.cpu().numpy())]
    det.sort(key=lambda z: -z[2]); keep = []
    for z in det:                                                            # merge duplicates from the two passes
        if all(np.hypot(z[0] - k[0], z[1] - k[1]) > 12 for k in keep): keep.append(z)
    return keep

def build_crop_dataset(clicks_json, video, ds, crop=640, box=18, negatives_per=1, log=print, seed=0, player_weights=None, at_feet_px=60, at_feet_copies=3):
    """player_weights: with a player detector, clicks within at_feet_px of a detected player's feet are 'at-feet' balls
    and get at_feet_copies crops each (26 Sep: 19 of 21 exam misses were balls at a player's feet)."""
    player = None
    if player_weights:
        from ultralytics import YOLO; player = YOLO(player_weights)
    """FAST training data: crops of `crop` px around each click (ball off-centre at random), plus crops of ball-free
    areas of the same frames as negatives. The exam frames are kept as FULL frames so the grade is unchanged."""
    import random
    rng = random.Random(seed); d = json.load(open(clicks_json))["frames"]; cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    for sub in ("images/train", "labels/train", "images/val", "labels/val"): os.makedirs(f"{ds}/{sub}", exist_ok=True)
    n = {"train": 0, "val": 0, "at_feet": 0}
    for r in d:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(r["t"] * fps))); ok, f = cap.read()
        if not ok: continue
        h, w = f.shape[:2]; name = r["file"][:-4]
        if r["split"] == "exam":                                                   # exam stays full-frame
            cv2.imwrite(f"{ds}/images/val/{name}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(f"{ds}/labels/val/{name}.txt", "w") as fh:
                if r["x"] is not None: b = box * (0.6 if r["y"] < h * 0.4 else 1.0); fh.write(f"0 {r['x'] / w:.6f} {r['y'] / h:.6f} {b / w:.6f} {b / h:.6f}\n")
            n["val"] += 1; continue
        crops = []; copies = 1
        if r["x"] is not None and player is not None:
            res = player(f, conf=0.3, imgsz=1280, verbose=False)[0]
            feet = [((a + c) / 2, dd) for (a, b, c, dd), cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy())]   # bottom-centre of each box
            if any(np.hypot(fx - r["x"], fy - r["y"]) <= at_feet_px for fx, fy in feet): copies = at_feet_copies; n["at_feet"] += 1
        if r["x"] is not None:
            for _ in range(copies):
                x0 = int(min(max(0, r["x"] - rng.randint(crop // 5, crop * 4 // 5)), w - crop)); y0 = int(min(max(0, r["y"] - rng.randint(crop // 5, crop * 4 // 5)), h - crop))
                crops.append((x0, y0, True))
        for _ in range(negatives_per):                                            # ball-free crop from the same frame
            for _try in range(20):
                x0, y0 = rng.randint(0, w - crop), rng.randint(0, h - crop)
                if r["x"] is None or not (x0 - 40 <= r["x"] <= x0 + crop + 40 and y0 - 40 <= r["y"] <= y0 + crop + 40): crops.append((x0, y0, False)); break
        for j, (x0, y0, has) in enumerate(crops):
            c = f[y0:y0 + crop, x0:x0 + crop]; cn = f"{name}_c{j}"
            cv2.imwrite(f"{ds}/images/train/{cn}.jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(f"{ds}/labels/train/{cn}.txt", "w") as fh:
                if has: b = box * (0.6 if r["y"] < h * 0.4 else 1.0); fh.write(f"0 {(r['x'] - x0) / crop:.6f} {(r['y'] - y0) / crop:.6f} {b / crop:.6f} {b / crop:.6f}\n")
            n["train"] += 1
    cap.release(); open(f"{ds}/data.yaml", "w").write(f"path: {ds}\ntrain: images/train\nval: images/val\nnames: ['ball']\n")
    log(f"  crop dataset: {n['train']} training crops ({crop} px), {n['val']} full-frame exam frames; {n['at_feet']} at-feet balls x{at_feet_copies}"); return n

def grade(weights, clicks_json, ds, hit_px=30, conf=0.25, imgsz=1920, log=print, pictures=None, old_weights=None):
    from ultralytics import YOLO
    m = YOLO(weights); old = YOLO(old_weights) if old_weights else None; d = [r for r in json.load(open(clicks_json))["frames"] if r["split"] == "exam"]; rows = []
    for r in d:
        img = cv2.imread(f"{ds}/images/val/{r['file'][:-4]}.jpg")
        if img is None: continue
        def run(model, far_zoom=False):
            det = detect(model, img, 0.05, imgsz, far_zoom=far_zoom); top = [z for z in det if z[2] >= conf]
            if r["x"] is None: return {"correct": len(top) == 0, "dist": None, "conf": top[0][2] if top else 0.0}
            if not top: return {"correct": False, "dist": None, "conf": 0.0}
            dist = min(np.hypot(x - r["x"], y - r["y"]) for x, y, _ in top[:3]); return {"correct": bool(dist <= hit_px), "dist": round(float(dist), 1), "conf": top[0][2]}
        new = run(m); rows.append({"file": r["file"], "t": r["t"], "has_ball": r["x"] is not None, "group": r["group"], "new": new, "new_farzoom": run(m, True), "old": run(old) if old else None})
        if pictures is not None:
            o = cv2.resize(img, (960, 540)); s = 960 / img.shape[1]
            for model, col, fz in ((old, (0, 0, 255), False), (m, (0, 255, 0), True)):
                if model is None: continue
                for x, y, cf in detect(model, img, conf, imgsz, far_zoom=fz): cv2.rectangle(o, (int(x * s) - 12, int(y * s) - 12), (int(x * s) + 12, int(y * s) + 12), col, 2)
            if r["x"] is not None: cv2.circle(o, (int(r["x"] * s), int(r["y"] * s)), 14, (0, 255, 255), 2)
            txt = f"t={r['t']:.0f}s " + ("HIT" if new["correct"] else "MISS") + (f" old:{'hit' if rows[-1]['old']['correct'] else 'miss'}" if old else "") + ("" if r["x"] is not None else " (no ball)")
            cv2.putText(o, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(o, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            pictures[r["file"]] = cv2.imencode(".jpg", o, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    def score(key):
        v = [x[key] for x in rows if x[key]]; b = [x for x in rows if x["has_ball"]]; nb = [x for x in rows if not x["has_ball"]]
        return {"correct": sum(x[key]["correct"] for x in rows if x[key]), "of": len(v), "ball_found": sum(x[key]["correct"] for x in b if x[key]), "ball_frames": len(b),
                "no_ball_right": sum(x[key]["correct"] for x in nb if x[key]), "no_ball_frames": len(nb)}
    s = {"new": score("new"), "new_farzoom": score("new_farzoom"), "old": score("old") if old else None, "frames": rows}
    log(f"  grade new: {s['new']}  new+far zoom: {s['new_farzoom']}  old: {s['old']}"); return s


def candidates(video, weights, cache, conf=0.05, log=print, batch=8, imgsz=1920, top=0.45):
    """ball candidates for the app pipeline with the click-trained model, as graded (full frame 1920 + far zoom on the
    top strip), batched on the GPU (26 Sep: one frame at a time ran at 1.5 frames/s = 90 min per 5-min clip);
    same {frame: [(x, y, conf), ...]} shape as ball.candidates, cached and resumable"""
    import pickle, os, time
    from ultralytics import YOLO
    from .video import frames
    out = {}; partial = cache + ".partial"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    if os.path.exists(partial): out = pickle.load(open(partial, "rb")); log(f"  ball: resuming from frame {len(out)}")
    model = YOLO(weights); t0 = time.time(); buf = []
    def flush():
        if not buf: return
        ks = [k for k, _ in buf]; ims = [f for _, f in buf]
        R1 = model(ims, conf=conf, imgsz=imgsz, verbose=False, half=True)
        h = ims[0].shape[0]; strips = [cv2.resize(f[:int(h * top)], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for f in ims]
        R2 = model(strips, conf=conf, imgsz=imgsz * 2, verbose=False, half=True)
        for k, r1, r2 in zip(ks, R1, R2):
            det = [(float((a + c) / 2), float((b + d) / 2), float(cf)) for (a, b, c, d), cf in zip(r1.boxes.xyxy.cpu().numpy(), r1.boxes.conf.cpu().numpy())]
            det += [(float((a + c) / 4), float((b + d) / 4), float(cf)) for (a, b, c, d), cf in zip(r2.boxes.xyxy.cpu().numpy(), r2.boxes.conf.cpu().numpy())]
            det.sort(key=lambda z: -z[2]); keep = []
            for z in det:
                if all(np.hypot(z[0] - q[0], z[1] - q[1]) > 12 for q in keep): keep.append(z)
            out[k] = keep
        buf.clear()
    for k, f in frames(video):
        if k in out: continue
        buf.append((k, f))
        if len(buf) >= batch: flush()
        if k % 500 == 0 and k: log(f"  ball frame {k}: {len(out.get(k - 1, []))} candidates, {(k / max(time.time() - t0, 1e-6)):.1f} frames/s"); pickle.dump(out, open(partial, "wb"))
    flush(); pickle.dump(out, open(cache, "wb"))
    if os.path.exists(partial): os.remove(partial)
    return out
