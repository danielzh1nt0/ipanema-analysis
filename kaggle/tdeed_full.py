# Kaggle (free GPU, 6 Oct, P-PASS): T-DEED SoccerNet ball-action model (challenge1 checkpoint) on the playing time of the 3
# demo matches (video from R2, cut to the periods, 25 fps, 796x448). -> /kaggle/working/tdf/<match>_<t0>.json (frames at 25 fps)
import subprocess, os, json, glob, shutil, time
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:], flush=True); return r.returncode
t0 = time.time(); W = "/kaggle/working/tdf"; os.makedirs(W, exist_ok=True); R2 = "{{R2}}"
MATCHES = {"p15u-vs-vallentuna-2026-10-03-6cce": [[435, 2890], [3220, 5645]], "SFKBP1109": [[555, 3060]], "p15u-vs-aik-2026-09-21-bd09": [[366, 3225]]}
sh("git clone -q --depth 1 https://github.com/arturxe2/T-DEED.git /kaggle/temp/td")
sh("pip install -q timm==1.0.3 gdown wandb tabulate SoccerNet")
sh("cd /kaggle/temp && gdown -q --folder https://drive.google.com/drive/folders/1sxZalU_hCwL8ITZCU9VqSWE8dB94lJty -O ck || true")
ck = [p for p in glob.glob("/kaggle/temp/ck/**/*.pt", recursive=True) if "SoccerNetBall_challenge1" in p]
dst = "/kaggle/temp/td/checkpoints/SoccerNetBall/SoccerNetBall_challenge1"; os.makedirs(dst, exist_ok=True); shutil.copy(ck[0], f"{dst}/checkpoint_best.pt")
os.makedirs("/kaggle/temp/v", exist_ok=True); res = {}
for m, per in MATCHES.items():
    sh(f"wget -q -O /kaggle/temp/v/full.mp4 {R2}/{m}/video.mp4")
    for a, b in per:
        name = f"{m}_{a}"; v = f"/kaggle/temp/v/{name}.mp4"
        sh(f"ffmpeg -loglevel error -y -ss {a} -i /kaggle/temp/v/full.mp4 -t {b - a} -r 25 -vf scale=796:448 -an -c:v libx264 -preset veryfast -crf 23 {v}")
        shutil.rmtree("/kaggle/temp/td/inference_output", ignore_errors=True)
        sh(f"cd /kaggle/temp/td && python3 inference.py --model SoccerNetBall_challenge1 --video_path {v} --frame_width 796 --frame_height 448 --inference_threshold 0.2")
        out = "/kaggle/temp/td/inference_output/results_inference.json"
        if os.path.exists(out):
            d = json.load(open(out)); d.update(fps=25, t0=a, t1=b, match=m); json.dump(d, open(f"{W}/{name}.json", "w")); res[name] = len(d["predictions"])
        else: res[name] = "failed"
        os.remove(v); print(name, res[name], round((time.time() - t0) / 60, 1), "min", flush=True)
json.dump({"ok": True, "events": res, "minutes": round((time.time() - t0) / 60, 1)}, open(f"{W}/run.json", "w"))
