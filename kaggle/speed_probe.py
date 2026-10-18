# Kaggle (free GPU), 16 Oct (F1): where does the time go? The 5-min clip took 111 min on a T4 (players 7.8 frames/s, RF-DETR
# ball 2.7 frames/s, WASB 6 frames/s) and IPANEMA_RF_FP16=1 changed nothing. Time each finder on 120 frames of the clip in
# several settings and print the exception if fp16 does not apply. Output: /kaggle/working/speed.json
import subprocess, os, sys, json, time, traceback
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2000:], flush=True); return r.returncode
R2 = "{{R2}}".rstrip("/"); T = "/kaggle/temp"; W = "/kaggle/working"; IA = f"{T}/ia"
sh("nvidia-smi --query-gpu=name --format=csv; nproc; free -g | head -2")
sh(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {IA}"); sh("pip install -q rfdetr==1.11.0 'supervision>=0.29,<0.31' omegaconf gdown 'numpy<2'")
os.makedirs(f"{T}/v", exist_ok=True); sh(f"wget -q -O {T}/v/clip.mp4 {R2}/SFKBP1109_s1200/video.mp4; ls -la {T}/v")
sys.path.insert(0, IA); os.environ["WASB_DIR"] = f"{T}/WASB-SBDT"
import numpy as np, cv2, torch
from ipanema import ballrf as BR, tracking as TR
out = {}
cap = cv2.VideoCapture(f"{T}/v/clip.mp4"); frames = []
for k in range(120):
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release(); print(len(frames), "frames", flush=True)
def timeit(name, fn, n):
    torch.cuda.synchronize(); t = time.time(); r = fn(); torch.cuda.synchronize(); dt = time.time() - t; out[name] = round(n / dt, 2); print(f"{name}: {n / dt:.2f} frames/s", flush=True); return r
# RF-DETR ball: fp32 as today, fp16 (no compile), bigger batch, fp16 + torch.autocast
for mode in ("fp32", "fp16"):
    os.environ["IPANEMA_RF_FP16"] = "1" if mode == "fp16" else "0"
    try:
        m = BR.load(f"{IA}/{BR.WEIGHTS}")
        out[f"ball {mode} optimized"] = f"{getattr(m, '_is_optimized_for_inference', None)} dtype {getattr(m, '_optimized_dtype', None)}"; print(out[f"ball {mode} optimized"], flush=True)
        BR.detect_many(m, frames[:4])                                            # warm-up
        for b in (4, 8, 16):
            timeit(f"ball {mode} batch {b}", lambda: [BR.detect_many(m, frames[i:i + b]) for i in range(0, 96, b)], 96)
        if mode == "fp32":
            with torch.autocast("cuda", dtype=torch.float16):
                timeit("ball fp32 model + autocast batch 8", lambda: [BR.detect_many(m, frames[i:i + 8]) for i in range(0, 96, 8)], 96)
        # how much is the GPU vs the crop / post-processing: the raw forward on ready tensors
        x = torch.rand(32, 3, 640, 640, device="cuda")
        net = m.model.inference_model if getattr(m, "_is_optimized_for_inference", False) and m.model.inference_model is not None else m.model.model
        dt_ = torch.float16 if mode == "fp16" and getattr(m, "_is_optimized_for_inference", False) else torch.float32
        with torch.inference_mode():
            net(x.to(dt_)); torch.cuda.synchronize(); t = time.time()
            for _ in range(5): net(x.to(dt_))
            torch.cuda.synchronize(); out[f"ball {mode} raw forward 32 crops"] = round(5 * 32 / (time.time() - t), 1); print("raw forward crops/s", out[f"ball {mode} raw forward 32 crops"], flush=True)
        del m; torch.cuda.empty_cache()
    except Exception: out[f"ball {mode}"] = traceback.format_exc()[-800:]; print(out[f"ball {mode}"], flush=True)
# players
for mode in ("fp32", "fp16"):
    os.environ["IPANEMA_RF_FP16"] = "1" if mode == "fp16" else "0"
    try:
        det = TR.RFDetrPerson("medium"); det.detect_batch(frames[:2], 0.3, TR.FOLLOW_TILES)
        for b in (8, 16):
            timeit(f"players {mode} batch {b}", lambda: [det.detect_batch(frames[i:i + b], 0.3, TR.FOLLOW_TILES) for i in range(0, 96, b)], 96)
        del det; torch.cuda.empty_cache()
    except Exception: out[f"players {mode}"] = traceback.format_exc()[-800:]; print(out[f"players {mode}"], flush=True)
# WASB (fine-tuned weights from the repo)
try:
    from ipanema import wasb as WB, wasb_train as WT
    root = f"{T}/root"; os.makedirs(f"{root}/models", exist_ok=True); import shutil; shutil.copy(f"{IA}/results/volume/models/wasb_finetuned.pth", f"{root}/models/wasb_finetuned.pth")
    WB.ensure(root, log=print); net = WB._model(root, "cuda", finetuned=True); net.eval()
    base = [WB.to_base(f) for f in frames[:33]]
    timeit("wasb tiled_predict (3 frames each)", lambda: [WT.tiled_predict(net, "cuda", base[i:i + 3]) for i in range(0, 30)], 30)
    with torch.autocast("cuda", dtype=torch.float16):
        timeit("wasb tiled_predict autocast fp16", lambda: [WT.tiled_predict(net, "cuda", base[i:i + 3]) for i in range(0, 30)], 30)
except Exception: out["wasb"] = traceback.format_exc()[-800:]; print(out["wasb"], flush=True)
json.dump(out, open(f"{W}/speed.json", "w"), indent=1); print(json.dumps(out, indent=1), flush=True)
