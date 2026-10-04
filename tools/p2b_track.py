"""P2b (4 Oct), free GitHub runner, no Modal: re-track the 3 Reymersholm P2 pieces (20 s each) with the kit-sample
fallback (kits.fit_frames: grass test keeps < 35% -> pitch-edge test). Same spots and settings as tools/p2_track.py;
output results/qa/p2b/track/<match>_<start>/ to compare with results/qa/p2/<match>_<start>/.
Usage (first line of triggers/free.txt): tools/p2b_track.py
Dry run: LOCAL_CLIP=<mp4> P2_START=0 P2_DUR=1 P2_OUT=<scratch> python tools/p2b_track.py"""
import os, subprocess, sys
M = "p15u-vs-reymersholm-2026-09-18"
env = dict(os.environ, P2_OUT=os.environ.get("P2_OUT", "results/qa/p2b/track"),
           P2_ONLY=os.environ.get("P2_ONLY", ",".join(f"{M}_{s}" for s in (726, 2227, 4227))))
sys.exit(subprocess.run([sys.executable, "tools/p2_track.py", "c"], env=env).returncode)
