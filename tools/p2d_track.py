"""P2d (4 Oct), free GitHub runner, no Modal: re-track Reymersholm piece 4227 (20 s) with the side-touchline test
(kits.side_lines: people beyond a boundary side line with path / fence beyond it read off the pitch). Same spot and
settings as tools/p2_track.py / tools/p2b_track.py; output results/qa/p2d/track/<match>_4227/ to compare with
results/qa/p2c/track/<match>_4227/.
Usage (first line of triggers/free.txt): tools/p2d_track.py
Dry run: LOCAL_CLIP=<mp4> P2_START=0 P2_DUR=1 P2_OUT=<scratch> python tools/p2d_track.py"""
import os, subprocess, sys
M = "p15u-vs-reymersholm-2026-09-18"
env = dict(os.environ, P2_OUT=os.environ.get("P2_OUT", "results/qa/p2d/track"), P2_ONLY=os.environ.get("P2_ONLY", f"{M}_4227"))
sys.exit(subprocess.run([sys.executable, "tools/p2_track.py", "c"], env=env).returncode)
