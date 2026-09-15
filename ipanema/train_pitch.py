"""Fine-tune the pitch keypoint model on reference/<clip>/pitch_kp.json labels (YOLO pose format, 32 keypoints). Writes models/pitch_finetuned.pt."""
import os, sys, json, glob, cv2, shutil, random

def build_dataset(root, videos_dir, sports_dir="/content/sports", log=print, min_pts=5):
    sys.path.append(sports_dir)
    from sports.configs.soccer import SoccerPitchConfiguration
    K = len(SoccerPitchConfiguration().vertices)
    ds = os.path.join(root, "models", "pitch_ds"); shutil.rmtree(ds, ignore_errors=True)
    for sp in ("train", "val"):
        os.makedirs(f"{ds}/images/{sp}", exist_ok=True); os.makedirs(f"{ds}/labels/{sp}", exist_ok=True)
    n = 0
    for gt_path in glob.glob(os.path.join(root, "reference", "*", "pitch_kp.json")):
        clip = os.path.basename(os.path.dirname(gt_path)); gt = json.load(open(gt_path))
        vid = next((p for p in glob.glob(f"{videos_dir}/{clip}.*") + glob.glob(f"{videos_dir}/{clip}/*.mp4")), None)
        if vid is None: log(f"  no video for {clip}"); continue
        if os.path.isdir(f"{videos_dir}/{clip}"): vid = max(glob.glob(f"{videos_dir}/{clip}/*.mp4"), key=os.path.getsize)
        cap = cv2.VideoCapture(vid)
        for fi, pts in gt.items():
            vis = {int(k): v for k, v in pts.items() if v}
            if len(vis) < min_pts: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
            if not ok: continue
            h, w = f.shape[:2]; sp = "val" if random.random() < 0.2 else "train"
            xs = [v[0] for v in vis.values()]; ys = [v[1] for v in vis.values()]
            cx, cy, bw, bh = (min(xs) + max(xs)) / 2 / w, (min(ys) + max(ys)) / 2 / h, (max(xs) - min(xs)) / w, (max(ys) - min(ys)) / h
            kp = " ".join(f"{vis[k][0]/w:.6f} {vis[k][1]/h:.6f} 2" if k in vis else "0 0 0" for k in range(K))
            cv2.imwrite(f"{ds}/images/{sp}/{clip}_{fi}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(f"{ds}/labels/{sp}/{clip}_{fi}.txt", "w").write(f"0 {cx:.6f} {cy:.6f} {max(bw,0.05):.6f} {max(bh,0.05):.6f} {kp}\n"); n += 1
        cap.release()
    open(f"{ds}/data.yaml", "w").write(f"path: {ds}\ntrain: images/train\nval: images/val\nkpt_shape: [{K}, 3]\nnames: ['pitch']\n")
    log(f"pitch dataset: {n} labelled frames"); return ds, n

def train(root, videos_dir, base_weights, epochs=60, imgsz=1280, log=print):
    from ultralytics import YOLO
    ds, n = build_dataset(root, videos_dir, log=log)
    if n < 40: log("fewer than 40 labelled pitch frames — label more before training"); return None
    model = YOLO(base_weights)
    model.train(data=f"{ds}/data.yaml", epochs=epochs, imgsz=imgsz, batch=4, lr0=0.001, freeze=10, project=f"{root}/models", name="pitch_ft", exist_ok=True, verbose=False, fliplr=0.0)
    best = f"{root}/models/pitch_ft/weights/best.pt"; out = f"{root}/models/pitch_finetuned.pt"; shutil.copy(best, out)
    log(f"fine-tuned pitch weights -> {out}"); return out
