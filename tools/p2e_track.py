"""P2e (4 Oct), free GitHub runner, no Modal: re-track Reymersholm piece 4227 (20 s) with the goalmouth keeper rules off
on grounds without calibration (tracking.clean keepers_by_zone=False; positions there are screen positions, so the pink
referee and near whites at the picture's left edge were made keepers of the green team). Same spot and settings as
tools/p2_track.py / p2d_track.py; output results/qa/p2e/track/<match>_4227/ to compare with results/qa/p2d/track/.
Usage (first line of triggers/free.txt): tools/p2e_track.py
Dry run: LOCAL_CLIP=<mp4> P2_START=0 P2_DUR=1 P2_OUT=<scratch> python tools/p2e_track.py"""
import os, subprocess, sys
M = "p15u-vs-reymersholm-2026-09-18"
env = dict(os.environ, P2_OUT=os.environ.get("P2_OUT", "results/qa/p2e/track"), P2_ONLY=os.environ.get("P2_ONLY", f"{M}_4227"))
sys.exit(subprocess.run([sys.executable, "tools/p2_track.py", "c"], env=env).returncode)
