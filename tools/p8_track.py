"""P8 (29 Sep), free GitHub runner: Reymersholm (night, no calibration) 20 s tracked twice with RF-DETR on CPU,
without and with the per-player team classifier (IPANEMA_KIT_CLS). -> results/qa/p8_track_{off,on}/"""
import os, subprocess, sys
for mode in ("0", "1"):
    env = dict(os.environ, MATCH=os.environ.get("P8_MATCH", "p15u-vs-reymersholm-2026-09-18"), START_S=os.environ.get("P8_START", "1500"),
               DUR_S=os.environ.get("P8_DUR", "20"), DETECTORS="rfdetr", IPANEMA_KIT_CLS=mode, SAVE_ALL_ROWS="1",
               TRACKTEST_OUT=os.environ.get("P8_OUT", "results/qa/p8_track") + ("_on" if mode == "1" else "_off"))
    if mode == "1": env["SKIP_INSTALL"] = "1"                                  # installed by the first pass
    r = subprocess.run([sys.executable, "tools/tracktest.py"], env=env)
    print("pass", mode, "returncode", r.returncode, flush=True)
    if r.returncode: sys.exit(r.returncode)
