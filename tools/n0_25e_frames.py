"""N0-25e (25 Oct, free runner): picture sheets of the app-clip sequences whose 'reached final third' or end reason
changes with N0-25e (results/qa/n0-25e/moments.json from tools/n0_25e_clips.py), drawn by tools/n0_20b_frames.py (tiles
every 0.5 s, app ball pick = red ring, the team's 2/3 line = yellow). Which rule says what is NOT on the sheet
(results/qa/n0-25e/moments.json is opened after grading).
    R2_PUBLIC_URL=... python tools/n0_25e_frames.py      (LOCAL_CLIP_SFK / LOCAL_CLIP_AIK = local videos for a dry run)"""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); rc = 0
M = json.load(open("results/qa/n0-25e/moments.json")); AR = json.load(open("results/qa/n0-25e/clips.json"))
for short, clip in (("sfk", "SFKBP1109_s1200"), ("aik", "p15u-vs-aik-2026-09-21-bd09_s2520")):
    ms = [{"id": m["id"], "clip": clip, "team": m["team"], "attacks": "right" if AR[clip]["attack_right"][m["team"]] else "left", "tiles": m["tiles"]}
          for m in M if m["clip"] == clip]
    f = f"results/free/n0-25e/moments_{short}.json"; os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump({"clip": clip, "moments": ms}, open(f, "w"), indent=1)
    env = {**os.environ, "N0_20B_MOMENTS": f, "N0_20B_OUT": f"results/free/n0-25e/{short}"}
    env.pop("LOCAL_CLIP", None)
    if os.environ.get(f"LOCAL_CLIP_{short.upper()}"): env["LOCAL_CLIP"] = os.environ[f"LOCAL_CLIP_{short.upper()}"]
    print("==", short, len(ms), "moments", flush=True); rc |= subprocess.call([sys.executable, os.path.join(HERE, "n0_20b_frames.py")], env=env)
sys.exit(rc)
