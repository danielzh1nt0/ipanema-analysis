"""N0-24c (24 Oct, free runner): picture sheets of the final-third entries that only one rule counts (old = hold 0,
new = IPANEMA_ENTRY_HOLD_S=3) on the two app clips, drawn by tools/n0_20b_frames.py (tiles every 0.5 s from 3 s before
to 1.5 s after, app ball pick = red ring, the attacking team's 2/3 line = yellow). The rule is not on the sheet
(results/qa/n0-24c/moments_key.json).
    R2_PUBLIC_URL=... python tools/n0_24c_frames.py      (LOCAL_CLIP_SFK / LOCAL_CLIP_AIK = local videos for a dry run)"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); rc = 0
for short in ("sfk", "aik"):
    env = {**os.environ, "N0_20B_MOMENTS": f"results/qa/n0-24c/moments_{short}.json", "N0_20B_OUT": f"results/free/n0-24c/{short}"}
    env.pop("LOCAL_CLIP", None)
    if os.environ.get(f"LOCAL_CLIP_{short.upper()}"): env["LOCAL_CLIP"] = os.environ[f"LOCAL_CLIP_{short.upper()}"]
    print("==", short, flush=True); rc |= subprocess.call([sys.executable, os.path.join(HERE, "n0_20b_frames.py")], env=env)
sys.exit(rc)
