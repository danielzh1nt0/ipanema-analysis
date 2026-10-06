"""N1a (6 Oct), free GitHub runner, no Modal: re-track the nightly Solberga spot (1500 s, 20 s) and Spånga 1159 (the other
piece where the new bright-kit rule moves people) with kits.bright_team on (uncalibrated grounds: a 'neither' reading
with the light kit's colour that is lighter still joins that kit - floodlit near whites on Solberga were removed as
referee / staff). Same spot and settings as tools/p2_track.py; output results/qa/n1a/track/<match>_<start>/ to compare
with results/qa/p2/<match>_<start>/ (P2, 4 Oct) and the nightly results/qa/tracktest_solberga-... (5 Oct).
Usage (first line of triggers/free.txt): tools/n1a_track.py
Dry run: TRACKTEST_PY=<stand-in> LOCAL_CLIP=<mp4> P2_START=0 P2_DUR=1 P2_OUT=<scratch> python tools/n1a_track.py"""
import os, subprocess, sys
PIECES = [("b", "solberga-vs-p09-norrviken-2026-09-11_1500"), ("a", "p15u-vs-spanga-2026-09-25_1159")]
bad = 0
for i, (group, piece) in enumerate(PIECES):
    env = dict(os.environ, P2_OUT=os.environ.get("P2_OUT", "results/qa/n1a/track"), P2_ONLY=piece)
    if i: env["SKIP_INSTALL"] = "1"
    bad += subprocess.run([sys.executable, "tools/p2_track.py", group], env=env).returncode != 0
sys.exit(1 if bad else 0)
