"""Pressing answer key, step 1 (9 Oct): short clips of lost balls for the Press Tapper page, where Daniel says by eye
whether a team-mate pressed the new carrier within 2 s. 24 of this team's losses picked at random (seed 7), each clip
runs from 1 s before the loss to 4 s after, 854x480 H.264 (~150 KB).  Free runner (R2_PUBLIC_URL) or VIDEO=<file>.
    MATCH=SFKBP1109 TEAM=A python tools/press_clips.py -> results/qa/press/<match>/{index.json, <id>.mp4}"""
import os, json, random, subprocess, shutil
M = os.environ.get("MATCH", "SFKBP1109"); TEAM = os.environ.get("TEAM", "A"); N = int(os.environ.get("N", "24"))
OUT = f"results/qa/press/{M}"; os.makedirs(OUT, exist_ok=True)
md = json.load(open(f"results/volume/runs/matches/{M}/match_data.json"))
per = md.get("periods") or []
inplay = lambda t: any(p["t_start"] + 3 <= t <= p["t_end"] - 5 for p in per) if per else True
losses = [e for e in md["events"] if e["type"] == "turnover_lost" and e["team"] == TEAM and inplay(e["t"])]
random.Random(7).shuffle(losses); picked = sorted(losses[:N], key=lambda e: e["t"])
src = os.environ.get("VIDEO") or f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{M}/video.mp4"
if not shutil.which("ffmpeg"): subprocess.run("sudo apt-get install -y -qq ffmpeg > /dev/null 2>&1 || apt-get install -y -qq ffmpeg", shell=True)
rows = []
for i, e in enumerate(picked):
    cid = f"l{i:02d}"; t0 = max(0.0, e["t"] - 1.0); out = f"{OUT}/{cid}.mp4"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t0:.2f}", "-i", src, "-t", "5", "-an", "-vf", "scale=854:480", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True); ok = os.path.exists(out) and os.path.getsize(out) > 10000
    rows.append({"id": cid, "event": e["id"], "t": e["t"], "clip_t0": round(t0, 2), "loss_at_s": round(e["t"] - t0, 2),
                 "pipeline": {"time_to_press": e["payload"].get("time_to_press"), "pressed_within_2s": e["payload"].get("pressed_within_2s"),
                              "near_at_2s": e["payload"].get("near_at_2s"), "regained_within_5s": e["payload"].get("regained_within_5s")}, "ok": ok})
    print(cid, round(e["t"], 1), "ok" if ok else r.stderr[-120:], flush=True)
json.dump({"match": M, "team": TEAM, "clips": rows}, open(f"{OUT}/index.json", "w"), indent=1)
print(sum(r["ok"] for r in rows), "clips")
