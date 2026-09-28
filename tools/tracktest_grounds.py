"""free runner: tracktest on other grounds (20 s of play each), one after the other"""
import os, subprocess, sys
for m, st in (("p15u-vs-spanga-2026-09-25", "1500"), ("p15u-vs-reymersholm-2026-09-18", "1500"), ("SFKBP1109_s1200", "60")):
    subprocess.run([sys.executable, "tools/tracktest.py"], env=dict(os.environ, MATCH=m, START_S=st, DUR_S="20"))
