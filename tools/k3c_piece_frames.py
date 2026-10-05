"""K3c (5 Oct, free runner, no Modal): do the 5-min piece videos the GPU labels players on have different colours from
the full video the kit model is learned on? For the 22 V3 carrier moments (results/qa/v3/carriers/moments.json, frames
from the full video), make the same frame the way rf_piece sees it: fullmatch.cut (x264 veryfast crf 20) then
video.normalise (x264 fast crf 20 yuv420p), and save it next to a fresh full-video frame.
    R2_PUBLIC_URL=... python tools/k3c_piece_frames.py   -> results/qa/k3c/{full,piece}/mNN.png"""
import os, sys, json, subprocess, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import fullmatch as FM, video as V
M = "p15u-vs-vallentuna-2026-10-03-6cce"; R2 = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/"); OUT = "results/qa/k3c"
src = f"{R2}/{M}/video.mp4"; ms = json.load(open("results/qa/v3/carriers/moments.json"))["moments"]
for d in ("full", "piece"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; print("fps", fps, flush=True)
info = {}
for m in ms:
    if "file" not in m: continue
    name = m["file"].replace(".jpg", ".png"); t = m["t"]; st = max(0.0, t - 10.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    if ok: cv2.imwrite(f"{OUT}/full/{name}", cv2.resize(f, (1920, 1080)))
    tmp = f"/tmp/k3c/{name}"; cutp = FM.cut(src, f"{tmp}/cut.mp4", st, 20.0); norm = V.normalise(cutp, f"{tmp}/norm.mp4")
    c2 = cv2.VideoCapture(norm); f2fps = c2.get(cv2.CAP_PROP_FPS) or fps; c2.set(cv2.CAP_PROP_POS_FRAMES, int(round((t - st) * f2fps))); ok2, g = c2.read(); c2.release()
    if ok2: cv2.imwrite(f"{OUT}/piece/{name}", cv2.resize(g, (1920, 1080)))
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt,color_range,color_space,width,height", "-of", "json", norm], capture_output=True, text=True).stdout
    info[name] = {"t": t, "full": ok, "piece": ok2, "probe": json.loads(probe or "{}")}
    print(name, t, ok, ok2, flush=True)
src_probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt,color_range,color_space,color_transfer,width,height,codec_name", "-of", "json", src], capture_output=True, text=True).stdout
json.dump({"source": json.loads(src_probe or "{}"), "frames": info}, open(f"{OUT}/info.json", "w"), indent=1)
