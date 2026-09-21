"""PnLCalib (Gutierrez-Perez & Agudo, CVIU 2026; GPL-2.0) on a stitched panorama -> pitch homography.
Their model expects 960x540 (16:9) input and a 105x68 m pitch with the origin at the centre; our panorama is ~2.8:1,
so it is letterboxed (not stretched). The pitch is symmetric, so the answer can come back mirrored: validation tries
all four flips and keeps the best. Nothing here replaces the hand calibration until validate() shows agreement."""
import os, sys, json, numpy as np, cv2

PNL_DIR = os.environ.get("PNL_DIR", "/content/PnLCalib")
REL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"

def _ensure(root, log=print):
    import subprocess, requests
    if not os.path.isdir(f"{PNL_DIR}/model"): subprocess.run(f"git clone -q --depth 1 https://github.com/mguti97/PnLCalib.git {PNL_DIR}", shell=True, check=True)
    md = os.path.join(root, "models"); os.makedirs(md, exist_ok=True)
    for name in ("SV_kp", "SV_lines"):
        p = os.path.join(md, f"pnl_{name}")
        if not os.path.exists(p):
            r = requests.get(f"{REL}/{name}", timeout=900); r.raise_for_status(); open(p, "wb").write(r.content); log(f"pnlcalib: downloaded {name} ({len(r.content)/1e6:.0f} MB)")
    return os.path.join(md, "pnl_SV_kp"), os.path.join(md, "pnl_SV_lines")

def calibrate(image, root, log=print, kp_threshold=0.3434, line_threshold=0.7867, refine=True):
    """returns H mapping centre-origin pitch metres (X in [-52.5,52.5], Y in [-34,34], Z=0) to image pixels, or None"""
    import torch, yaml, torchvision.transforms as T
    wkp, wl = _ensure(root, log=log)
    sys.path.insert(0, PNL_DIR); cwd = os.getcwd(); os.chdir(PNL_DIR)
    try:
        import inference as PI
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils.utils_calib import FramebyFrameCalib
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        PI.device = dev; PI.transform2 = T.Resize((540, 960))          # module globals their inference() relies on
        m = get_cls_net(yaml.safe_load(open("config/hrnetv2_w48.yaml"))); m.load_state_dict(torch.load(wkp, map_location=dev)); m.to(dev).eval()
        ml = get_cls_net_l(yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))); ml.load_state_dict(torch.load(wl, map_location=dev)); ml.to(dev).eval()
        h, w = image.shape[:2]; H16 = int(round(w * 9 / 16)); top = max(0, (H16 - h) // 2)
        lb = cv2.copyMakeBorder(image, top, max(0, H16 - h - top), 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))   # letterbox to 16:9
        cam = FramebyFrameCalib(iwidth=lb.shape[1], iheight=lb.shape[0], denormalize=True)
        params = PI.inference(cam, lb, m, ml, kp_threshold, line_threshold, refine)
        if params is None: log("pnlcalib: no calibration found"); return None
        P = PI.projection_from_cam_params(params)
        Hc = P[:, [0, 1, 3]]                                          # ground plane Z=0
        return np.array([[1, 0, 0], [0, 1, -top], [0, 0, 1]], float) @ Hc   # undo the letterbox offset
    finally: os.chdir(cwd)

def _lines_px(Hc):
    """pitch-model line segments (105x68, centre origin) projected with Hc"""
    segs = [((-52.5, -34), (52.5, -34)), ((-52.5, 34), (52.5, 34)), ((-52.5, -34), (-52.5, 34)), ((52.5, -34), (52.5, 34)), ((0, -34), (0, 34)),
            ((-52.5, -20.16), (-36, -20.16)), ((-36, -20.16), (-36, 20.16)), ((-36, 20.16), (-52.5, 20.16)),
            ((52.5, -20.16), (36, -20.16)), ((36, -20.16), (36, 20.16)), ((36, 20.16), (52.5, 20.16))]
    th = np.linspace(0, 2 * np.pi, 120); circ = [(9.15 * np.cos(a), 9.15 * np.sin(a)) for a in th]
    segs += list(zip(circ[:-1], circ[1:])); out = []
    for a, b in segs:
        p = cv2.perspectiveTransform(np.float32([[a], [b]]), Hc).reshape(-1, 2)
        if np.isfinite(p).all() and np.abs(p).max() < 1e5: out.append(p)
    return out

def validate(root, code_dir="/content/ipanema-analysis", match="SFKBP1109", log=print):
    """run PnLCalib on the hand-calibrated panorama and report agreement in metres (within 20 m of the centre spot)"""
    spec = json.load(open(os.path.join(code_dir, "calibration", f"{match}.json")))
    img = cv2.imread(os.path.join(code_dir, spec["mosaic"])); Hhand = np.array(spec["H_pitch_to_mosaic"], float)   # our metres (120x70 model) -> px
    Hc = calibrate(img, root, log=log)
    if Hc is None: return {"ok": False, "reason": "no calibration"}
    g = np.array([(x, y) for x in np.linspace(-20, 20, 9) for y in np.linspace(-18, 18, 7)], np.float32)
    px = cv2.perspectiveTransform(g.reshape(-1, 1, 2), Hc).reshape(-1, 2)
    ours = cv2.perspectiveTransform(px.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(Hhand)).reshape(-1, 2)   # back to our metres
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            exp = np.c_[60 + sx * g[:, 0], 35 + sy * g[:, 1]]           # our centre spot is (60, 35)
            e = np.linalg.norm(ours - exp, axis=1); cand = (float(np.median(e)), float(np.percentile(e, 90)), sx, sy)
            if best is None or cand[0] < best[0]: best = cand
    res = {"ok": True, "median_err_m": round(best[0], 2), "p90_err_m": round(best[1], 2), "flip": [best[2], best[3]]}
    vis = img.copy()
    for seg in _lines_px(Hc): cv2.line(vis, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (0, 0, 255), 3)
    th = np.linspace(0, 2 * np.pi, 120); hc = cv2.perspectiveTransform(np.float32([[[60 + 9.15 * np.cos(a), 35 + 9.15 * np.sin(a)]] for a in th]), Hhand).reshape(-1, 2)
    cv2.polylines(vis, [hc.astype(np.int32)], True, (0, 255, 0), 3)
    hw = cv2.perspectiveTransform(np.float32([[[60, 0]], [[60, 70]]]), Hhand).reshape(-1, 2); cv2.line(vis, tuple(hw[0].astype(int)), tuple(hw[1].astype(int)), (0, 255, 0), 3)
    d = os.path.join(code_dir, "results", "debug"); os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, f"pnlcalib_{match}.jpg"), cv2.resize(vis, (1800, int(1800 * vis.shape[0] / vis.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 85])
    log(f"pnlcalib validation on {match} panorama: median {res['median_err_m']} m, p90 {res['p90_err_m']} m (flip {res['flip']}); red = PnLCalib, green = hand calibration")
    return res
