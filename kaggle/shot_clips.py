# Kaggle (free, 5 Oct, V1c): short MP4 clips for the Shot Spot Marker page (the page can't stream from R2: the artifact frame
# only plays media published with the page). Per Veo shot: -2 s to +28 s, 960x540, H.264, no audio, faststart.
import json, subprocess, urllib.request, os
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/app/shotclick/shots.json", headers={"User-Agent": "Mozilla/5.0"})
S = json.load(urllib.request.urlopen(req)); tot = 0
for m, v in S.items():
    for s in v["shots"]:
        t = int(s["t"]); out = f"{W}/{m}_{t}.mp4"
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(max(0, t - 2)), "-i", f"{R2}/{m}/video.mp4", "-t", "30", "-an",
               "-vf", "scale=960:540", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        r = subprocess.run(cmd, capture_output=True, text=True); sz = os.path.getsize(out) if os.path.exists(out) else 0; tot += sz
        print(m, t, sz, r.stderr[-200:], flush=True)
json.dump({"ok": True, "bytes": tot}, open(f"{W}/result.json", "w")); print("done", tot)
