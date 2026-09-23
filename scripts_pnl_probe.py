"""PnLCalib on our frames at several detection thresholds: how many keypoints/lines it finds, whether it calibrates,
and the drawn result at the most permissive setting (run on GitHub's free runner from /tmp/pnl)."""
import sys, os, glob, cv2, torch, yaml
sys.path.insert(0, "/tmp/pnl"); os.chdir("/tmp/pnl")
import inference as I
I.device = "cpu"
from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml")); cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))
m = get_cls_net(cfg); m.load_state_dict(torch.load("SV_kp", map_location="cpu")); m.eval()
ml = get_cls_net_l(cfg_l); ml.load_state_dict(torch.load("SV_lines", map_location="cpu")); ml.eval()
I.model, I.model_l = m, ml
out = sys.argv[1]; os.makedirs(out, exist_ok=True); rows = []
for f in sorted(glob.glob("/tmp/frames/*.jpg")):
    img = cv2.imread(f); name = os.path.basename(f)[:-4]; res = []
    for kt, lt in ((0.3434, 0.7867), (0.2, 0.5), (0.1, 0.3)):
        cam = I.FramebyFrameCalib(iwidth=img.shape[1], iheight=img.shape[0], denormalize=True)
        rgb = I.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); t = I.f.to_tensor(rgb).float().unsqueeze(0)
        t = t if t.size()[-1] == 960 else I.transform2(t)
        with torch.no_grad(): hm = m(t); hl = ml(t)
        kpd = I.coords_to_dict(I.get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1, :, :]), threshold=kt)
        lnd = I.coords_to_dict(I.get_keypoints_from_heatmap_batch_maxpool_l(hl[:, :-1, :, :]), threshold=lt)
        nk, nl = len(kpd[0]), len(lnd[0])
        kpd2, lnd2 = I.complete_keypoints(kpd[0], lnd[0], w=960, h=540, normalize=True)
        cam.update(kpd2, lnd2); p = cam.heuristic_voting(refine_lines=True)
        res.append(f"thr {kt}/{lt}: {nk} keypoints, {nl} lines -> {'CALIBRATED' if p is not None else 'no calibration'}")
        if p is not None and (kt, lt) == (0.1, 0.3) or (p is not None and not os.path.exists(f"{out}/{name}.png")):
            cv2.imwrite(f"{out}/{name}.png", I.project(img, I.projection_from_cam_params(p)))
    rows.append(f"{name}: " + " | ".join(res)); print(rows[-1], flush=True)
open(f"{out}/summary.txt", "w").write("\n".join(rows) + "\n")
