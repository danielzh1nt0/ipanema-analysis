"""Train and grade the line-class network (Colab free T4). Needs torch + segmentation_models_pytorch.
Everything is saved as it goes (progress json + last.pt every epoch), so a disconnect loses at most one epoch."""
import os, json, time, numpy as np, cv2
from . import lines as LN

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
PAD_H = 384                                                     # 360 -> 384 (network needs multiples of 32); padding is "don't know"

def _prep(img):
    x = (img[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD; h, w = x.shape[:2]
    x = np.pad(x, ((0, PAD_H - h), (0, 0), (0, 0))) if h < PAD_H else x
    return x.transpose(2, 0, 1).copy()

def _augment(img, m, rng):
    h, w = m.shape
    if rng.rand() < 0.5: img, m = img[:, ::-1], m[:, ::-1]                    # valid: classes have no left/right
    if rng.rand() < 0.6:                                                         # zoom in (what the follow-cam does)
        s = rng.uniform(1.0, 1.8); cw, ch = int(w / s), int(h / s); x0, y0 = rng.randint(0, w - cw + 1), rng.randint(0, h - ch + 1)
        img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR); m = cv2.resize(m[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)
    img = np.ascontiguousarray(img).astype(np.float32)
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)                   # light / contrast
    if rng.rand() < 0.3: img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.2))  # soft focus / motion
    return np.clip(img, 0, 255).astype(np.uint8), np.ascontiguousarray(m)

def load_split(ds_dir, split):
    names = sorted(f[:-4] for f in os.listdir(f"{ds_dir}/images/{split}"))
    return [(cv2.imread(f"{ds_dir}/images/{split}/{n}.jpg"), cv2.imread(f"{ds_dir}/masks/{split}/{n}.png", cv2.IMREAD_GRAYSCALE), n) for n in names]

def make_model(encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, classes=len(LN.CLASSES))

def train(ds_dir, out_dir, epochs=80, batch=8, lr=3e-4, encoder="resnet34", weights="imagenet", max_minutes=45, device=None, log=print, seed=0):
    import torch, torch.nn.functional as F
    device = device or ("cuda" if torch.cuda.is_available() else "cpu"); os.makedirs(out_dir, exist_ok=True)
    tr, va = load_split(ds_dir, "train"), load_split(ds_dir, "val")
    torch.manual_seed(seed); rng = np.random.RandomState(seed); model = make_model(encoder, weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); steps = epochs * max(1, len(tr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    cw = torch.ones(len(LN.CLASSES), device=device); cw[0] = 0.15                # thin lines vs lots of grass
    scaler = torch.amp.GradScaler(enabled=device == "cuda"); t0 = time.time(); hist = []
    def batch_tensors(items, aug):
        X, Y = [], []
        for img, m, _ in items:
            if aug: img, m = _augment(img, m, rng)
            X.append(_prep(img)); Y.append(np.pad(m, ((0, PAD_H - m.shape[0]), (0, 0)), constant_values=LN.IGNORE))
        return torch.from_numpy(np.stack(X)).to(device), torch.from_numpy(np.stack(Y).astype(np.int64)).to(device)
    def loss_fn(logits, y):
        ce = F.cross_entropy(logits, y, weight=cw, ignore_index=LN.IGNORE)
        p = logits.float().softmax(1); valid = (y != LN.IGNORE).unsqueeze(1); oh = F.one_hot(y.clamp(max=len(LN.CLASSES) - 1), len(LN.CLASSES)).permute(0, 3, 1, 2).float()
        p, oh = p * valid, oh * valid; inter = (p * oh).sum((0, 2, 3))[1:]; den = (p + oh).sum((0, 2, 3))[1:]
        present = oh.sum((0, 2, 3))[1:] > 0
        dice = 1 - (2 * inter + 1) / (den + 1); return ce + (dice[present].mean() if present.any() else 0.0)
    for ep in range(epochs):
        model.train(); order = rng.permutation(len(tr)); tl = []
        for i in range(0, len(order) - batch + 1, batch):
            x, y = batch_tensors([tr[j] for j in order[i:i + batch]], True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"): out = model(x)
            loss = loss_fn(out.float(), y); opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl.append(float(loss.detach()))
        iou = val_iou(model, va, device); mins = (time.time() - t0) / 60
        hist.append({"epoch": ep + 1, "train_loss": round(float(np.mean(tl)), 4), "val_line_iou": iou, "minutes": round(mins, 1)})
        torch.save(model.state_dict(), f"{out_dir}/last.pt"); json.dump(hist, open(f"{out_dir}/progress.json", "w"))
        if ep % 5 == 4 or ep == epochs - 1: log(f"  epoch {ep + 1}/{epochs}: loss {hist[-1]['train_loss']}, held-back line IoU {iou}, {mins:.1f} min")
        if mins > max_minutes: log(f"  stopped at the {max_minutes}-minute budget (epoch {ep + 1})"); break
    return f"{out_dir}/last.pt"

def val_iou(model, items, device):
    """mean IoU over line classes present in the held-back labels (a sanity number only; the grade is the pose error)"""
    import torch
    model.eval(); I = np.zeros(len(LN.CLASSES)); U = np.zeros(len(LN.CLASSES))
    with torch.no_grad():
        for img, m, _ in items:
            pred = predict(model, img, device); ok = m != LN.IGNORE
            for k in range(1, len(LN.CLASSES)):
                a, b = (pred == k) & ok, (m == k) & ok; I[k] += (a & b).sum(); U[k] += (a | b).sum()
    have = U[1:] > 0; return round(float((I[1:][have] / U[1:][have]).mean()), 3) if have.any() else None

def predict(model, img, device, min_prob=0.4):
    """class mask for a 640x360 frame; left-right flip averaged in (classes are mirror-symmetric); unsure pixels -> 0"""
    import torch
    model.eval(); h = img.shape[0]
    x = torch.from_numpy(np.stack([_prep(img), _prep(np.ascontiguousarray(img[:, ::-1]))])).to(device)
    with torch.no_grad(): p = model(x).float().softmax(1).cpu().numpy()
    p = (p[0] + p[1][:, :, ::-1]) / 2; p = p[:, :h]; k = p.argmax(0).astype(np.uint8); k[p.max(0) < min_prob] = 0
    return k

def load(weights_path, encoder="resnet34", device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu"); m = make_model(encoder, None)
    m.load_state_dict(torch.load(weights_path, map_location=device)); return m.to(device).eval(), device

def evaluate(weights_path, ds_dir, out_dir, extra=None, encoder="resnet34", near_px=10.0, log=print, snap=False, polish_choice="old_snap"):
    """GRADE on the held-back clicked frames: predict lines -> cold pose fit (known base) -> pixel error of the whole pitch
    against Daniel's clicked pose. Saves one picture per frame (left: predicted lines, right: the fitted pitch) and a
    summary JSON. extra = [(name, image)] frames without a known answer: pictures only, to be judged by eye."""
    model, device = load(weights_path, encoder); meta = json.load(open(f"{ds_dir}/poses.json")); cam = meta["camera"]
    os.makedirs(out_dir, exist_ok=True); rows = []
    for img, m, name in load_split(ds_dir, "val") + [(im, None, n) for n, im in (extra or [])]:
        pred = predict(model, img, device); pose, info = LN.fit_pose(pred, cam)
        ref = meta["poses"].get(name, {}).get("pose") if m is not None else None
        pairs = meta["poses"].get(name, {}).get("clicks") if m is not None else None
        grade = (lambda P: LN.click_error(cam, P, pairs)) if pairs else (lambda P: LN.pose_error(cam, P, ref) if P is not None else {"median_px": float("inf")})
        e = None if ref is None else grade(pose)
        full = f"{ds_dir}/images_full/val/{name}.jpg"; e_fit = e
        e_ref = LN.click_error(cam, ref, pairs)["median_px"] if (pairs and ref is not None) else None      # the reference pose's own miss
        variants = {"fit": pose}
        if snap and pose is not None:                                           # every polish graded side by side
            big = cv2.imread(full) if (m is not None and os.path.exists(full)) else cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
            variants["old_snap"] = np.array(LN.snap(big, cam, pose)); variants["sharp_polish"] = LN.polish(big, pred, cam, pose)[0]
        errs = {k: (None if (ref is None or v is None) else round(grade(v)["median_px"], 1)) for k, v in variants.items()}
        pose = variants.get(polish_choice, pose) if variants.get(polish_choice) is not None else pose   # the one used for the pictures
        if ref is not None: e = grade(pose)
        row = {"frame": name, "held_back": ref is not None, "confident": info.get("confident"), "error_median_px_1280": None if e is None else (round(e["median_px"], 1) if np.isfinite(e["median_px"]) else None),
               "correct": None if e is None else bool(e["median_px"] <= near_px),
               "reference_pose_click_error_px": None if e_ref is None else round(e_ref, 1), "variants_px": errs, "error_before_snap_px": None if (e_fit is None or not np.isfinite(e_fit["median_px"])) else round(e_fit["median_px"], 1), "classes_seen": info.get("classes_seen"), "families": info.get("supported_families")}
        rows.append(row)
        left = LN.overlay(img, pred, 0.75); right = LN.draw_pose(img, cam, pose) if pose is not None else img.copy()
        txt = ("no pose found" if pose is None else ("%.0f px from your clicks" % e["median_px"] if e else "no answer known")) + (" | confident" if info.get("confident") else " | NOT sure")
        cv2.putText(right, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(right, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(f"{out_dir}/{'held' if ref is not None else 'extra'}_{name}.jpg", np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 82])
        json.dump(rows, open(f"{out_dir}/eval.json", "w"), indent=0)
    held = [r for r in rows if r["held_back"]]; conf = [r for r in held if r["confident"]]
    s = {"held_back_frames": len(held), "placed_correctly": sum(r["correct"] for r in held), "confident": len(conf),
         "confident_and_correct": sum(r["correct"] for r in conf), "confident_but_wrong": [r["frame"] for r in conf if not r["correct"]],
         "median_error_px_1280": float(np.median([r["error_median_px_1280"] if r["error_median_px_1280"] is not None else np.inf for r in held])) if held else None,
         "errors_px": [r["error_median_px_1280"] for r in held], "errors_before_snap_px": [r["error_before_snap_px"] for r in held],
         "per_variant_within_10px": {k: sum(1 for r in held if r["variants_px"].get(k) is not None and r["variants_px"][k] <= near_px) for k in (held[0]["variants_px"] if held else {})},
         "per_variant_median_px": {k: float(np.median([r["variants_px"][k] for r in held if r["variants_px"].get(k) is not None])) for k in (held[0]["variants_px"] if held else {})}, "extra_frames_pictured": len(rows) - len(held)}
    json.dump({"summary": s, "frames": rows}, open(f"{out_dir}/eval.json", "w"), indent=0); log(f"grade: {s}")
    return s
