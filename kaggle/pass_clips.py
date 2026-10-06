# Kaggle (free, 6 Oct, P-PASS): 2-min clips for the Pass Tapper page (Daniel taps each pass -> the answer key for passes).
import json, subprocess, os
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; tot = 0
CLIPS = [("p15u-vs-vallentuna-2026-10-03-6cce", 4000), ("p15u-vs-vallentuna-2026-10-03-6cce", 1200), ("SFKBP1109", 1500), ("p15u-vs-aik-2026-09-21-bd09", 1500)]
for m, t in CLIPS:
    out = f"{W}/{m}_{t}.mp4"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", f"{R2}/{m}/video.mp4", "-t", "120", "-an",
           "-vf", "scale=1280:720", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True); sz = os.path.getsize(out) if os.path.exists(out) else 0; tot += sz
    print(m, t, sz, r.stderr[-200:], flush=True)
json.dump({"ok": True, "bytes": tot}, open(f"{W}/result.json", "w")); print("done", tot)
