"""Fine-tune the ball detector on the labelled reference frames (reference/<clip>/ball_gt.json). Writes weights to <root>/models/ball_finetuned.pt."""
import os, json, glob, cv2, shutil, random

def build_dataset(root, videos_dir, box=16, log=print):
    ds = os.path.join(root, "models", "ball_ds"); shutil.rmtree(ds, ignore_errors=True)
    for sp in ("train", "val"):
        os.makedirs(f"{ds}/images/{sp}", exist_ok=True); os.makedirs(f"{ds}/labels/{sp}", exist_ok=True)
    n = 0
    for gt_path in glob.glob(os.path.join(root, "reference", "*", "ball_gt.json")):
        clip = os.path.basename(os.path.dirname(gt_path))
        vid = next((p for p in glob.glob(f"{videos_dir}/{clip}.*")), None)
        if vid is None and os.path.isdir(f"{videos_dir}/{clip}"):
            vids = [p for p in glob.glob(f"{videos_dir}/{clip}/*") if p.lower().endswith((".mp4", ".mov", ".mkv"))]
            vid = max(vids, key=os.path.getsize) if vids else None          # the full match in its folder
        if vid is None: log(f"  no video for {clip}"); continue
        gt = {int(k): v for k, v in json.load(open(gt_path)).items()}; cap = cv2.VideoCapture(vid)
        for i, g in gt.items():
            if g is None: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
            if not ok: continue
            h, w = f.shape[:2]; sp = "val" if random.random() < 0.2 else "train"
            cv2.imwrite(f"{ds}/images/{sp}/{clip}_{i}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(f"{ds}/labels/{sp}/{clip}_{i}.txt", "w").write(f"0 {g[0]/w:.6f} {g[1]/h:.6f} {box/w:.6f} {box/h:.6f}\n"); n += 1
        cap.release()
    open(f"{ds}/data.yaml", "w").write(f"path: {ds}\ntrain: images/train\nval: images/val\nnames: ['ball']\n")
    log(f"dataset: {n} labelled frames"); return ds, n

def train(root, videos_dir, base_weights, epochs=60, imgsz=1920, log=print):
    from ultralytics import YOLO
    ds, n = build_dataset(root, videos_dir, log=log)
    if n < 40: log("fewer than 40 labelled frames — label more clips before training"); return None
    model = YOLO(base_weights)
    # small ball: full resolution, more of the network trainable, longer schedule, gentler augmentation
    model.train(data=f"{ds}/data.yaml", epochs=epochs, imgsz=imgsz, batch=4, lr0=0.001, freeze=4, mosaic=0.5, scale=0.3, project=f"{root}/models", name="ball_ft", exist_ok=True, verbose=False, patience=25)
    best = f"{root}/models/ball_ft/weights/best.pt"; out = f"{root}/models/ball_finetuned.pt"; shutil.copy(best, out)
    log(f"fine-tuned ball weights -> {out}"); return out
