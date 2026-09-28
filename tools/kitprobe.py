"""P4 (28 Sep): kit split on several grounds without GPU. Per ground: 24 frames from a 5-min piece (R2), RF-DETR people
(CPU), saved as results/qa/kitprobe/<ground>/f*.jpg + boxes.json so kit variants can be tried offline, plus the kit
strips of 4 variants (current / pitch_only / green_kit / both)."""
import os, sys, json, subprocess, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: import rfdetr  # noqa
except ImportError: subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
from ipanema import tracking as TR, kits as KT
GROUNDS = [("p15u-vs-reymersholm-2026-09-18", 1500), ("p15u-vs-spanga-2026-09-25", 1500), ("SFKBP1109_s1200", 0)]
det = TR.RFDetrPerson("medium"); OUT0 = "results/qa/kitprobe"; summary = {}
for match, start in GROUNDS:
    OUT = f"{OUT0}/{match}"; os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{match}/video.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    fb = []; boxes = {}
    for i, t in enumerate(np.linspace(start + 3, start + 297, 24)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
        if not ok: continue
        b = [list(map(float, x)) for x in det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0].xyxy]
        cv2.imwrite(f"{OUT}/f{i:02d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 88]); boxes[f"f{i:02d}.jpg"] = b; fb.append((f, np.array(b)))
        print(match, i, len(b), flush=True)
    json.dump(boxes, open(f"{OUT}/boxes.json", "w"))
    rows = []
    for name, kw in (("current", {}), ("pitch_only", {"pitch_only": True}), ("green_kit", {"green_kit": True}), ("both", {"pitch_only": True, "green_kit": True})):
        m = KT.KitTeamModel().fit_frames(fb, log=print, **kw)
        cnt = {"A": 0, "B": 0, "K": 0, None: 0}
        for f, b in fb:
            for c in m.predict_batch(f, b): cnt[c if c in cnt else "K"] += 1
        summary[f"{match} {name}"] = {k if k else "none": v for k, v in cnt.items()}
        strip = np.vstack([cv2.resize(m.strips[t], (576, 144)) for t in ("A", "B")]); cv2.putText(strip, name, (4, 18), 0, 0.6, (255, 255, 255), 2)
        rows.append(strip)
    cv2.imwrite(f"{OUT0}/{match}_variants.jpg", np.vstack(rows))
json.dump(summary, open(f"{OUT0}/summary.json", "w"), indent=1); print(json.dumps(summary, indent=1))
