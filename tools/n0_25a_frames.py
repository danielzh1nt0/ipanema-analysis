"""N0-25a (25 Oct, free runner): picture sheets of every run (sprint) on the two app clips (new rule + the old
rule's own), drawn by tools/n0_20b_frames.py (tiles every 0.33 s from 0.7 s before the run to 0.3 s after,
the runner's feet = red ring). The rule is not on the sheet
(results/qa/n0-25a/moments_key.json).
    R2_PUBLIC_URL=... python tools/n0_25a_frames.py      (LOCAL_CLIP_SFK / LOCAL_CLIP_AIK = local videos for a dry run)"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); rc = 0
for short in ("sfk", "aik"):
    env = {**os.environ, "N0_20B_MOMENTS": f"results/qa/n0-25a/moments_{short}.json", "N0_20B_OUT": f"results/free/n0-25a/{short}"}
    env.pop("LOCAL_CLIP", None)
    if os.environ.get(f"LOCAL_CLIP_{short.upper()}"): env["LOCAL_CLIP"] = os.environ[f"LOCAL_CLIP_{short.upper()}"]
    print("==", short, flush=True); rc |= subprocess.call([sys.executable, os.path.join(HERE, "n0_20b_frames.py")], env=env)
sys.exit(rc)
