"""Full-game handling: cut segments of videos/<match>/full.mp4 into videos/<match>_segN.mp4 so the normal loop processes them."""
import os, subprocess, json

DEFAULT_SEGMENTS = {"SFKBP1109": [(0, 300)]}     # (start_s, duration_s); more segments are added once the first one is checked

def prepare_segments(root, log=print):
    vids = os.path.join(root, "videos"); made = []
    for match in sorted(d for d in os.listdir(vids) if os.path.isdir(os.path.join(vids, d))):
        full = next((os.path.join(vids, match, f) for f in os.listdir(os.path.join(vids, match)) if f.lower().startswith("full") and f.lower().endswith((".mp4", ".mov", ".mkv"))), None)
        if not full: continue
        cfg = os.path.join(vids, match, "segments.json")
        segs = json.load(open(cfg)) if os.path.exists(cfg) else DEFAULT_SEGMENTS.get(match, [(0, 300)])
        for i, (start, dur) in enumerate(segs, 1):
            out = os.path.join(vids, f"{match}_seg{i}.mp4")
            if os.path.exists(out): continue
            log(f"segment: {match} seg{i} {start}s +{dur}s -> {os.path.basename(out)}")
            r = subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-i", full, "-t", str(dur), "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", out + ".tmp.mp4"], capture_output=True)
            if r.returncode == 0: os.replace(out + ".tmp.mp4", out); made.append(out)
            else: log(f"  ffmpeg failed: {r.stderr[-200:]}")
    return made
