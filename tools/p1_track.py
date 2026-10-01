"""P1 v3 (1 Oct), free GitHub runner: does the striped-kit switch hold in real tracking? 20 s tracked twice with RF-DETR
on CPU, switch off (IPANEMA_KIT_AUTO=0 = the reading before P1) and on (the new default), on Spånga (striped kits, the
same minute as the Kaggle piece 1125) and Djursholm (piece 2714: spectators became a team).
-> results/qa/p1_track_<ground>_{off,on}/ (summary, kit strips, compare pictures, all rows).
Dry run: LOCAL_CLIP=<mp4> P1_DUR=1 P1_GROUNDS=spanga (no installs, no R2)."""
import os, subprocess, sys
GROUNDS = {"spanga": ("p15u-vs-spanga-2026-09-25", "1125"), "djursholm": ("p15u-vs-djursholm-2026-09-26", "2714")}
first = True
for g in os.environ.get("P1_GROUNDS", "spanga,djursholm").split(","):
    match, start = GROUNDS[g]
    for mode in ("0", "1"):
        env = dict(os.environ, MATCH=match, START_S=os.environ.get("P1_START", start), DUR_S=os.environ.get("P1_DUR", "20"), DETECTORS="rfdetr",
                   IPANEMA_KIT_AUTO=mode, SAVE_ALL_ROWS="1", TRACKTEST_OUT=f"{os.environ.get('P1_OUT', 'results/qa/p1_track')}_{g}_" + ("on" if mode == "1" else "off"))
        if not first: env["SKIP_INSTALL"] = "1"                                 # installed by the first pass
        first = False
        r = subprocess.run([sys.executable, "tools/tracktest.py"], env=env)
        print(g, "switch", "on" if mode == "1" else "off", "returncode", r.returncode, flush=True)
        if r.returncode: sys.exit(r.returncode)
