# Kaggle (free GPU, 6 Oct, P-PASS research): run T-DEED's SoccerNet Ball Action Spotting model (12 classes incl. PASS,
# DRIVE; trained on broadcast EFL video, GPL-3.0 code, checkpoints on Google Drive) on our 4 two-minute pass clips and
# save its events, to compare with Daniel's taps. -> /kaggle/working/tdeed/<clip>.json
import subprocess, os, json, glob, shutil, time
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
t0 = time.time(); W = "/kaggle/working/tdeed"; os.makedirs(W, exist_ok=True)
sh("nvidia-smi --query-gpu=name --format=csv")
sh("git clone -q --depth 1 https://github.com/arturxe2/T-DEED.git /kaggle/temp/td")
sh("git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git /kaggle/temp/ia")
sh("pip install -q timm==1.0.3 gdown wandb tabulate")
sh("cd /kaggle/temp && gdown -q --folder https://drive.google.com/drive/folders/1sxZalU_hCwL8ITZCU9VqSWE8dB94lJty -O ck || true; find /kaggle/temp/ck | head -40")
ck = [p for p in glob.glob("/kaggle/temp/ck/**/*.pt", recursive=True) if "SoccerNetBall_challenge1" in p] or [p for p in glob.glob("/kaggle/temp/ck/**/*.pt", recursive=True) if "SoccerNetBall" in p]
print("checkpoints", ck, flush=True)
if not ck: json.dump({"ok": False, "why": "no SoccerNetBall checkpoint"}, open(f"{W}/run.json", "w")); raise SystemExit
model = "SoccerNetBall_challenge1" if "challenge1" in ck[0] else "SoccerNetBall_challenge2"
dst = f"/kaggle/temp/td/checkpoints/SoccerNetBall/{model}"; os.makedirs(dst, exist_ok=True); shutil.copy(ck[0], f"{dst}/checkpoint_best.pt")
res = {}
for clip in sorted(glob.glob("/kaggle/temp/ia/results/kaggle/pass_clips/*.mp4")):
    name = os.path.basename(clip)[:-4]; v25 = f"/kaggle/temp/{name}_25.mp4"
    sh(f"ffmpeg -loglevel error -y -i {clip} -r 25 -vf scale=796:448 -an {v25}")
    shutil.rmtree("/kaggle/temp/td/inference_output", ignore_errors=True)
    rc = sh(f"cd /kaggle/temp/td && python3 inference.py --model {model} --video_path {v25} --frame_width 796 --frame_height 448 --inference_threshold 0.2")
    out = "/kaggle/temp/td/inference_output/results_inference.json"
    if os.path.exists(out):
        d = json.load(open(out)); d["fps"] = 25; json.dump(d, open(f"{W}/{name}.json", "w")); res[name] = len(d["predictions"])
    else: res[name] = f"failed rc {rc}"
json.dump({"ok": True, "model": model, "events": res, "minutes": round((time.time() - t0) / 60, 1)}, open(f"{W}/run.json", "w")); print(res)
