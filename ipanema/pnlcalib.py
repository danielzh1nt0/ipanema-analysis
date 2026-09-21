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

def _agreement(Hc, Hours, centre=(60.0, 35.0)):
    """median/p90 disagreement in metres between PnLCalib (centre-origin 105x68) and our calibration (0..120 x 0..70), best of 4 mirror flips"""
    g = np.array([(x, y) for x in np.linspace(-20, 20, 9) for y in np.linspace(-18, 18, 7)], np.float32)
    px = cv2.perspectiveTransform(g.reshape(-1, 1, 2), Hc).reshape(-1, 2)
    ours = cv2.perspectiveTransform(px.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(Hours)).reshape(-1, 2)
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            e = np.linalg.norm(ours - np.c_[centre[0] + sx * g[:, 0], centre[1] + sy * g[:, 1]], axis=1)
            c = (round(float(np.median(e)), 2), round(float(np.percentile(e, 90)), 2), [sx, sy])
            if best is None or c[0] < best[0]: best = c
    return {"median_err_m": best[0], "p90_err_m": best[1], "flip": best[2]}

def _overlay(img, Hc, Hours, path):
    vis = img.copy()
    for seg in _lines_px(Hc): cv2.line(vis, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (0, 0, 255), 3)
    th = np.linspace(0, 2 * np.pi, 120)
    hc = cv2.perspectiveTransform(np.float32([[[60 + 9.15 * np.cos(a), 35 + 9.15 * np.sin(a)]] for a in th]), Hours).reshape(-1, 2)
    cv2.polylines(vis, [hc.astype(np.int32)], True, (0, 255, 0), 3)
    w = 1400; cv2.imwrite(path, cv2.resize(vis, (w, int(w * vis.shape[0] / vis.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 85])

def validate(root, code_dir="/content/ipanema-analysis", match="SFKBP1109", clip="SFKBP1109_s1200", log=print):
    """three inputs, each measured against a calibration we have verified by eye:
       (1) the panorama letterboxed, (2) a 16:9 crop of the panorama around the centre circle, (3) three follow-cam frames"""
    import pickle, glob
    spec = json.load(open(os.path.join(code_dir, "calibration", f"{match}.json")))
    pano = cv2.imread(os.path.join(code_dir, spec["mosaic"])); Hp = np.array(spec["H_pitch_to_mosaic"], float)
    d = os.path.join(code_dir, "results", "debug"); os.makedirs(d, exist_ok=True); out = {}
    def run(name, img, Hours):
        try:
            Hc = calibrate(img, root, log=log)
            if Hc is None: out[name] = "no calibration"; return
            out[name] = _agreement(Hc, Hours); _overlay(img, Hc, Hours, os.path.join(d, f"pnlcalib_{name}.jpg"))
        except Exception as e: out[name] = f"error {e!r}"[:200]
    run("panorama", pano, Hp)
    c = cv2.perspectiveTransform(np.float32([[[60, 35]]]), Hp).reshape(2); cw = 1920; ch = 1080
    x0 = int(np.clip(c[0] - cw / 2, 0, pano.shape[1] - cw)); y0 = int(np.clip(c[1] - ch * 0.35, 0, pano.shape[0] - ch))
    crop = pano[y0:y0 + ch, x0:x0 + cw]; Hcrop = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ Hp
    run("pano_crop", crop, Hcrop)
    hm = sorted(glob.glob(os.path.join(root, "cache", clip, "calibration_pano_*.pkl")))
    vid = os.path.join(root, "videos", f"{clip}.mp4")
    if hm and os.path.exists(vid):
        Hm = pickle.load(open(hm[-1], "rb")); cap = cv2.VideoCapture(vid); n = int(cap.get(7))
        for q in (0.25, 0.5, 0.75):
            k = int(n * q); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
            if ok and k in Hm: run(f"frame{k}", f, np.array(Hm[k], float))
        cap.release()
    log("pnlcalib validation: " + json.dumps(out))
    return out
