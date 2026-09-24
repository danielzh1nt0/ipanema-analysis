"""GitHub runner: download the match from R2, cut the frames, run the line-model round, write results into the repo."""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import linerun
tag = os.environ.get("TAG", time.strftime("%Y%m%d_%H%M")); OUT = f"results/lines/{tag}"; LAB = "/tmp/lab"; os.makedirs(OUT, exist_ok=True)
log_path = f"{OUT}/log.txt"
def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True); open(log_path, "a").write(line + "\n")
url = os.environ["R2_PUBLIC_URL"].rstrip("/"); video = "/tmp/v.mp4"
for k in ("SFKBP1109/video.mp4", "SFKBP1109/full.mp4", "SFKBP1109.mp4", "SFKBP1109/video_cropped.mp4", "videos/SFKBP1109.mp4", "SFKBP1109/SFKBP1109.mp4"):
    code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-r", "0-0", f"{url}/{k}"], capture_output=True, text=True).stdout
    if code in ("200", "206"):
        log(f"downloading {k}"); subprocess.run(["curl", "-sfL", "-o", video, f"{url}/{k}"], check=True); break
else: log("follow-cam video not found on R2"); sys.exit(1)
log(f"video {os.path.getsize(video) / 1e9:.2f} GB")
linerun.cut_frames(video, LAB, log=log)
s = linerun.run(LAB, OUT, epochs=int(os.environ.get("EPOCHS", "40")), max_minutes=int(os.environ.get("MAX_MIN", "150")), log=log, work="/tmp")
r = s; g = f"{r['placed_correctly']}/{r['held_back_frames']} held-back frames placed within 10 px, median error {r['median_error_px_1280']:.1f} px; base fix {'accepted' if r['base_fix_accepted'] else 'rejected'}; {r['minutes']:.0f} min"
open("/tmp/issue_title", "w").write(f"Ipanema lines {tag}: {g[:120]}")
open("/tmp/issue_body.md", "w").write(f"**{g}**\n\nErrors per frame (px at 1280): {r['errors_px']}\nBefore snap: {r['errors_before_snap_px']}\nConfident but wrong: {r['confident_but_wrong']}\n\nPictures: `results/lines/{tag}/eval` and `results/lines/{tag}/base_check` in the repo.\n")
log(g)
