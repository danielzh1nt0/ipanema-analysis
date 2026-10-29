"""N0-29e: does a wider 'on the ball' radius (possession_simple near_m, IPANEMA_NEAR_M, default 1.5 m) find more of the
real balls lost on Metrica? Runs the n0_29e_lossrec check (perfect + flicker ball, seed 0, 4 halves) under the env.

    IPANEMA_NEAR_M=3.0 PYTHONPATH=. python tools/n0_29e_nearm.py /tmp/metrica/data
"""
import sys, os, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n0_24d_ppda as PP, n0_29e_lossrec as LR

if __name__ == "__main__":
    rows = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr, ev = PP.load_game(sys.argv[1], g)
        for per in (1, 2):
            tr = LR.truth_half(ev, per)
            for nz in (False, "flicker"):
                rows.setdefault(str(nz), []).append(LR.score(tr, LR.ours_half(fr, per, nz, 0))); gc.collect()
    print(os.environ.get("IPANEMA_NEAR_M"), json.dumps({k: LR.summarise(v) for k, v in rows.items()}))
