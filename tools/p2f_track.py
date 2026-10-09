"""P2f (9 Oct), free GitHub runner, no Modal: re-track Reymersholm piece 4227 (20 s) with the dim-light-kit rule on
(kits.dim_light_team: on uncalibrated grounds a green-kit reading whose hue leans >= 18 deg towards yellow - the cream
colour of far white shirts under floodlights - is the white team). Same spot and settings as tools/p2_track.py /
p2e_track.py; output results/qa/p2f/track/<match>_4227/ to compare with results/qa/p2e/track/.
Usage (first line of triggers/free.txt): tools/p2f_track.py
Dry run: LOCAL_CLIP=<mp4> P2_START=0 P2_DUR=1 P2_OUT=<scratch> python tools/p2f_track.py"""
import os, subprocess, sys
M = "p15u-vs-reymersholm-2026-09-18"
env = dict(os.environ, P2_OUT=os.environ.get("P2_OUT", "results/qa/p2f/track"), P2_ONLY=os.environ.get("P2_ONLY", f"{M}_4227"))
sys.exit(subprocess.run([sys.executable, "tools/p2_track.py", "c"], env=env).returncode)
