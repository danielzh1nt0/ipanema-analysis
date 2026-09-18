"""Team assignment by visual embedding. The pipeline names teams A (light? no: A/B are arbitrary) — A = the cluster with MORE dark shirt pixels."""
import sys, cv2, numpy as np, logging

def shirt(frame, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]; h = y2 - y1
    return frame[max(0, y1): max(0, y1) + max(8, int(0.55 * h)), max(0, x1): max(0, x1) + max(4, x2 - x1)]

def _dark_frac(c):
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV); grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 80)
    v = hsv[..., 2][~grass]; return float((v < 110).mean()) if v.size else 0.0

class TeamModel:
    def __init__(self, sports_dir, device="cuda"):
        sys.path.append(sports_dir)
        import torch
        from sports.common.team import TeamClassifier
        self.clf = TeamClassifier(device=device if torch.cuda.is_available() else "cpu")
        self.cluster_to_team = None; self.dark_share = None
    def fit(self, video, weights_player, conf=0.3, sample_every=15, max_crops=600, log=print):
        import supervision as sv
        from ultralytics import YOLO
        from .video import frames
        model = YOLO(weights_player); crops = []
        for i, f in frames(video):
            if i % sample_every: continue
            res = model(f, conf=conf, verbose=False)[0]; det = sv.Detections.from_ultralytics(res).with_nms(0.5, class_agnostic=True)
            for j in range(len(det)):
                cls = res.names[int(det.class_id[j])].lower() if det.class_id is not None else "player"
                if "referee" in cls or "goalkeeper" in cls: continue
                c = shirt(f, det.xyxy[j])
                if c.size and c.shape[0] >= 12 and c.shape[1] >= 8: crops.append(c)
            if len(crops) >= max_crops: break
        import random, torch; random.seed(0); np.random.seed(0); torch.manual_seed(0)
        self.clf.fit(crops); labels = np.array(self.clf.predict(crops))
        dark = np.array([_dark_frac(c) for c in crops])
        sizes = np.bincount(labels, minlength=2)
        # embedding clusters can collapse; when they do (or kits are plainly light vs dark) split on shirt brightness instead
        if sizes.min() < 0.2 * len(crops) or (abs(np.median(dark[labels == 0]) - np.median(dark[labels == 1])) < 0.08 and dark.std() > 0.2):
            thr = float(np.median(dark)); labels = (dark < thr).astype(int)     # 1 = lighter kit, 0 = darker
            self._brightness_split = thr; log(f"teams: embedding split {sizes.tolist()} unreliable -> brightness split at {thr:.2f}")
        else: self._brightness_split = None
        d0 = np.mean(dark[labels == 0]) if (labels == 0).any() else 0.0; d1 = np.mean(dark[labels == 1]) if (labels == 1).any() else 0.0
        dark = 0 if d0 > d1 else 1
        self.cluster_to_team = {dark: "A", 1 - dark: "B"}; self.dark_share = {"A": round(max(d0, d1), 2), "B": round(min(d0, d1), 2)}
        sizes = np.bincount(labels).tolist()
        log(f"teams: {len(crops)} crops, clusters {sizes}, dark share A {self.dark_share['A']} / B {self.dark_share['B']} (A = darker kit)")
        if abs(d0 - d1) < 0.08: log("  WARNING: kits are not clearly one dark and one light — check kit strips")
        self._strips(crops, labels); return self
    def _strips(self, crops, labels):
        self.strips = {}
        for cl in (0, 1):
            cs = [c for c, l in zip(crops, labels) if l == cl][:24]
            tiles = [cv2.resize(c, (48, 72)) for c in cs] or [np.zeros((72, 48, 3), np.uint8)]
            while len(tiles) % 12: tiles.append(np.zeros_like(tiles[0]))
            self.strips[self.cluster_to_team[cl]] = np.vstack([np.hstack(tiles[r:r + 12]) for r in range(0, len(tiles), 12)])
    def predict_batch(self, frame, xyxys):
        labs = ["B"] * len(xyxys); idx, cs = [], []
        for j, b in enumerate(xyxys):
            c = shirt(frame, b)
            if c.size and c.shape[0] >= 12 and c.shape[1] >= 8: idx.append(j); cs.append(c)
        if cs:
            if getattr(self, "_brightness_split", None) is not None:
                for j, c in zip(idx, cs): labs[j] = self.cluster_to_team[int(_dark_frac(c) < self._brightness_split)]
            else:
                for j, cl in zip(idx, self.clf.predict(cs)): labs[j] = self.cluster_to_team[int(cl)]
        return labs

def silence_progress():
    import os, warnings; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"; os.environ["TRANSFORMERS_VERBOSITY"] = "error"; warnings.filterwarnings("ignore")
    import tqdm, functools
    tqdm.tqdm.__init__ = functools.partialmethod(tqdm.tqdm.__init__, disable=True)
    logging.getLogger("transformers").setLevel(logging.ERROR); logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
