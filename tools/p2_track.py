"""P2 (4 Oct), free GitHub runner, no Modal: players on the 6 training matches, 20 s tracked at 3 spots per match
(RF-DETR on CPU, same kit step / clean-up / gap filling as tracktest). Spots were picked by eye from the playerbench
frames (results/qa/players/<match>/) where play is on: one early first half, one late first half, one second half.
Each piece -> results/qa/p2/<match>_<start>/ (summary, compare pictures, all rows, the detector's people on the 8 key
frames = keydets.json for the 'missed players' count). Table: tools/p2_table.py.
Usage (first line of triggers/free.txt): tools/p2_track.py <group>   group = a | b | c (2 matches each, ~6 x 17 min)
Dry run: LOCAL_CLIP=<mp4> P2_START=0 P2_DUR=1 P2_ONLY=<match>_<start> python tools/p2_track.py a"""
import os, subprocess, sys
SPOTS = {   # match -> start seconds (frame / 29.97 of a live-play playerbench frame)
    "p15u-vs-vasalund-2026-09-20": [1164, 2113, 4012],
    "p15u-vs-spanga-2026-09-25": [1159, 2576, 3994],
    "p15u-vs-djursholm-2026-09-26": [2072, 3272, 5072],
    "solberga-vs-p09-norrviken-2026-09-11": [1500, 2723, 4558],
    "p15u-vs-reymersholm-2026-09-18": [726, 2227, 4227],
    "p09-norrviken-vs-solheim-2026-08-30": [1856, 2931, 5081],
}
GROUPS = {"a": list(SPOTS)[0:2], "b": list(SPOTS)[2:4], "c": list(SPOTS)[4:6]}
OUT = os.environ.get("P2_OUT", "results/qa/p2")

def pieces(group):
    only = [x for x in os.environ.get("P2_ONLY", "").split(",") if x]
    out = [(m, s) for m in GROUPS[group] for s in SPOTS[m]]
    return [p for p in out if not only or f"{p[0]}_{p[1]}" in only]

if __name__ == "__main__":
    first = True; bad = 0
    for match, start in pieces(sys.argv[1] if len(sys.argv) > 1 else "a"):
        d = f"{OUT}/{match}_{start}"
        if os.path.exists(f"{d}/comparison.md"): print("done already", d, flush=True); continue
        env = dict(os.environ, MATCH=match, START_S=os.environ.get("P2_START", str(start)), DUR_S=os.environ.get("P2_DUR", "20"), DETECTORS="rfdetr",
                   SAVE_ALL_ROWS="1", KEY_DETS="1", TRACKTEST_OUT=d)
        if not first: env["SKIP_INSTALL"] = "1"
        first = False
        r = subprocess.run([sys.executable, "tools/tracktest.py"], env=env)
        print(match, start, "returncode", r.returncode, flush=True); bad += r.returncode != 0
    sys.exit(1 if bad else 0)
