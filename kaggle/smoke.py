# Kaggle smoke test (28 Sep): GPU, internet, R2 video access, WASB repo clone. Writes /kaggle/working/result.json
import json, subprocess, time, os
out = {}
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); return (r.stdout + r.stderr)[-800:]
out["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
try:
    import torch; out["torch"] = torch.__version__; out["cuda"] = torch.cuda.is_available(); out["gpus"] = torch.cuda.device_count()
except Exception as e: out["torch"] = repr(e)
t = time.time(); out["git_clone_wasb"] = sh("git clone -q --depth 1 https://github.com/nttcom/WASB-SBDT.git /tmp/wasb && ls /tmp/wasb | head"); out["clone_s"] = round(time.time() - t, 1)
try:
    import cv2
    src = "{{R2}}/p15u-vs-vasalund-2026-09-20/video.mp4"; t = time.time(); cap = cv2.VideoCapture(src)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, 30000); ok, f = cap.read()
    out["r2_video"] = {"frames": n, "fps": cap.get(cv2.CAP_PROP_FPS), "seek_read_ok": bool(ok), "shape": list(f.shape) if ok else None, "s": round(time.time() - t, 1)}
except Exception as e: out["r2_video"] = repr(e)
out["pip_ultralytics"] = sh("pip install -q ultralytics && python -c 'import ultralytics; print(ultralytics.__version__)'")
out["disk"] = sh("df -h /kaggle/working | tail -1")
json.dump(out, open("/kaggle/working/result.json", "w"), indent=1); print(json.dumps(out, indent=1))
