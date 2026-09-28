"""free runner: current detector (roboflow YOLOv8, AGPL, tiled like tracking) vs RF-DETR Medium COCO person (Apache) on the
clean frames of results/qa/players/*/raw_*.jpg. Side-by-side pictures + counts -> results/qa/detectors/"""
import os, sys, glob, json, subprocess, time
subprocess.run("pip install -q ultralytics rfdetr torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True)
subprocess.run(["gdown", "-q", "-O", "player.pt", "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"])
sys.path.insert(0, os.getcwd())
import cv2, numpy as np
from PIL import Image
from ultralytics import YOLO
from rfdetr import RFDETRMedium
from ipanema import tracking as TR
yolo = YOLO("player.pt"); rf = RFDETRMedium(); OUT = "results/qa/detectors"; os.makedirs(OUT, exist_ok=True); rep = []
def rf_tiled(im, thr):
    h, w = im.shape[:2]; boxes = []
    for x0, y0, x1, y1 in TR.FOLLOW_TILES:
        a, b, c, d = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        det = rf.predict(Image.fromarray(cv2.cvtColor(im[b:d, a:c], cv2.COLOR_BGR2RGB)), threshold=thr)
        for (p, q, r, s), cid, cf in zip(det.xyxy, det.class_id, det.confidence):
            if int(cid) == 1: boxes.append([p + a, q + b, r + a, s + b, float(cf)])
    if not boxes: return []
    B = np.array(boxes); keep = []
    for i in np.argsort(-B[:, 4]):                                                 # simple NMS across tiles
        if all((min(B[i, 2], B[j, 2]) - max(B[i, 0], B[j, 0])) * (min(B[i, 3], B[j, 3]) - max(B[i, 1], B[j, 1])) < 0.5 * (B[i, 2] - B[i, 0]) * (B[i, 3] - B[i, 1]) or min(B[i, 2], B[j, 2]) < max(B[i, 0], B[j, 0]) or min(B[i, 3], B[j, 3]) < max(B[i, 1], B[j, 1]) for j in keep): keep.append(i)
    return B[keep].tolist()
for f in sorted(glob.glob("results/qa/players/*/raw_*.jpg")):
    im = cv2.imread(f); m = f.split("/")[-2]; t0 = time.time()
    det, names = TR.detect_tiled_batch(yolo, [im], 0.30, TR.FOLLOW_TILES, imgsz=960, half=False)[0]
    ref = next((i for i, n in names.items() if "referee" in n.lower()), -1); ball = next((i for i, n in names.items() if n.lower() == "ball"), -1)
    Y = [(b, int(c)) for b, c in zip(det.xyxy, det.class_id) if int(c) != ball]; t1 = time.time()
    R = rf_tiled(im, 0.35); t2 = time.time()
    a = im.copy(); b = im.copy()
    for (x1, y1, x2, y2), c in Y: cv2.rectangle(a, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255) if c == ref else (0, 200, 0), 3)
    for x1, y1, x2, y2, cf in R: cv2.rectangle(b, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 3)
    for img, t in ((a, f"current: {sum(c != ref for _, c in Y)} kept + {sum(c == ref for _, c in Y)} 'referee' (deleted)"), (b, f"RF-DETR person: {len(R)}")):
        cv2.putText(img, t, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8); cv2.putText(img, t, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    name = f"{m}_{os.path.basename(f)[4:]}"; cv2.imwrite(f"{OUT}/{name}", cv2.resize(np.vstack([a, b]), (1280, 1440)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    rep.append({"frame": name, "current_kept": sum(c != ref for _, c in Y), "current_referee": sum(c == ref for _, c in Y), "rfdetr": len(R), "sec_current": round(t1 - t0, 1), "sec_rfdetr": round(t2 - t1, 1)}); print(rep[-1], flush=True)
json.dump(rep, open(f"{OUT}/summary.json", "w"), indent=1)
