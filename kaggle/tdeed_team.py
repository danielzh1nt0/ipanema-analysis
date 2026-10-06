# Kaggle (free GPU, 6 Oct, P-PASS): SoccerNet 2025 Team Ball Action Spotting baseline (T-DEED with a team head: action +
# left/right team in the picture, GPL-3.0, checkpoint on Google Drive) on our 4 pass clips. Saves per-frame scores
# (stride 2 at 25 fps) for every class x side, to tune offline. -> /kaggle/working/tdt/<clip>.npz
import subprocess, os, sys, json, glob, time, types
from collections import deque
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
t0 = time.time(); W = "/kaggle/working/tdt"; os.makedirs(W, exist_ok=True)
sh("git clone -q --depth 1 https://github.com/SoccerNet/sn-teamspotting.git /kaggle/temp/ts")
sh("git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git /kaggle/temp/ia")
sh("pip install -q timm==1.0.3 gdown wandb tabulate SoccerNet")
sh("cd /kaggle/temp && gdown -q --folder https://drive.google.com/drive/folders/16IqSkctIGp76ZYKKvJvMB_ggHcQsessM -O tck || true; find /kaggle/temp/tck | head")
ck = glob.glob("/kaggle/temp/tck/**/*.pt", recursive=True); print("ckpt", ck, flush=True)
os.chdir("/kaggle/temp/ts"); sys.path.insert(0, "/kaggle/temp/ts")
import torch, numpy as np, cv2
from model.model import TDEEDModel
cfg = json.load(open("config/SoccerNetBall/SoccerNetBall_baseline.json"))
a = types.SimpleNamespace(**cfg); a.crop_dim = None; a.model = "SoccerNetBall_baseline"
sd = torch.load(ck[0], map_location="cpu"); sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
heads = {k: tuple(v.shape) for k, v in sd.items() if "pred" in k.lower() and k.endswith("weight")}; print("head shapes", heads, flush=True)
model = None
for nc in ([13, 9], [13, 18], [13, 17], [13]):
    try:
        m = TDEEDModel(args=a)
        if len(nc) > 1: m._model.update_pred_head(nc); m._num_classes = int(np.array(nc).sum())
        m.load(sd); model = m; print("loaded with heads", nc, flush=True); break
    except Exception as e: print("heads", nc, "failed:", repr(e)[:300], flush=True)
if model is None: json.dump({"ok": False, "heads": heads}, open(f"{W}/run.json", "w")); raise SystemExit
def clips(path, L=100, ov=75, stride=2, size=(796, 448)):
    cap = cv2.VideoCapture(path); buf = deque(); i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % stride == 0:
            f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size); buf.append(torch.from_numpy(f).permute(2, 0, 1))
            if len(buf) == L:
                yield torch.stack(list(buf)), i // stride - L + 1
                for _ in range(L - ov): buf.popleft()
        i += 1
    cap.release()
res = {}
for clip in sorted(glob.glob("/kaggle/temp/ia/results/kaggle/pass_clips/*.mp4")):
    name = os.path.basename(clip)[:-4]; v25 = f"/kaggle/temp/{name}_25.mp4"
    sh(f"ffmpeg -loglevel error -y -i {clip} -r 25 -an {v25}")
    n = int(cv2.VideoCapture(v25).get(cv2.CAP_PROP_FRAME_COUNT)) // 2 + 1; acc = None; sup = np.zeros(n)
    for x, s in clips(v25):
        _, p = model.predict(x.unsqueeze(0)); p = p[0]
        if acc is None: acc = np.zeros((n, p.shape[-1]), np.float32)
        e = min(n, s + p.shape[0]); acc[s:e] += p[:e - s]; sup[s:e] += 1
    acc = acc / np.maximum(sup, 1)[:, None]; np.savez_compressed(f"{W}/{name}.npz", scores=acc, fps=25 / 2)
    res[name] = list(acc.shape)
json.dump({"ok": True, "shapes": res, "classes_note": "col 0 background; class c (0..11 in data/soccernetball/class.txt): col 1+2c left, 2+2c right", "minutes": round((time.time() - t0) / 60, 1)}, open(f"{W}/run.json", "w")); print(res)
