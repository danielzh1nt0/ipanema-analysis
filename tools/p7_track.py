"""P7 (29 Sep), free GitHub runner: Reymersholm (night, no calibration) 20 s from 1500 s tracked twice with RF-DETR on CPU,
without and with the off-pitch drop (pitch-edge test). -> results/qa/p7_track_{off,on}/ (counts per frame, compare_*.jpg)"""
import os, subprocess, sys
for mode in ("0", "1"):
    env = dict(os.environ, MATCH=os.environ.get("P7_MATCH", "p15u-vs-reymersholm-2026-09-18"), START_S=os.environ.get("P7_START", "1500"),
               DUR_S=os.environ.get("P7_DUR", "20"), DETECTORS="rfdetr", IPANEMA_OFFPITCH=mode, SAVE_ALL_ROWS="1",
               TRACKTEST_OUT=f"results/qa/p7_track_{'on' if mode == '1' else 'off'}")
    if mode == "1": env["SKIP_INSTALL"] = "1"                                  # installed by the first pass
    r = subprocess.run([sys.executable, "tools/tracktest.py"], env=env)
    print("pass", mode, "returncode", r.returncode, flush=True)
    if r.returncode: sys.exit(r.returncode)
