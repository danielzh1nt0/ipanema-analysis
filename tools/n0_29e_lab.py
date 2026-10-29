"""N0-29e lab: WHERE do the missing balls lost go? On Metrica (4 halves, perfect ball + 'flicker'), the pipeline's
inputs are captured once (n0_24d_ppda.ours_run) and only the turnover step is re-run with other settings:
spell state take/join (what the counted stats read), turnovers()' before/after hold, confirm_losses' keep rule.
Scored against the truth losses of tools/n0_29e_lossrec.truth_half (all, and S9-type = winner keeps 3 s or passes).

    PYTHONPATH=. python tools/n0_29e_lab.py /tmp/metrica/data [--out results/qa/n0-29e/lab.json]
"""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P

FPS = 25.0
SPELLS = [(1.5, 3.0), (1.0, 3.0), (1.0, 2.0), (0.5, 2.0), (0.0, 0.0)]
HOLDS = [1.0, 0.5]
KEEPS = [3.0, 2.0, 0.0]


def tvs_variant(got, take, join, hold, keep):
    cst = P.spell_state(got["state"], FPS, take, join)
    tv = P.turnovers(got["per"], got["frames_"], cst, got["ballm"], FPS, got["ar"], min_before_s=hold, min_after_s=hold)
    kept, _ = P.confirm_losses(tv, cst, got["ps"], FPS, keep)
    return kept


def main(d, out):
    """one half at a time (all captured halves at once ran out of memory): capture, run every variant, keep counts only"""
    import gc, n0_24d_ppda as PP, n0_29e_lossrec as LR
    V = list(itertools.product(SPELLS, HOLDS, KEEPS)); acc = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr, ev = PP.load_game(d, g)
        for per in (1, 2):
            tr = LR.truth_half(ev, per); k0 = min(k for k in fr if fr[k][0] == per)
            for noisy in (False, "flicker"):
                _, got = PP.ours_run(fr, per, noisy, 0); nz = str(noisy)
                for (take, join), hold, keep in V:
                    kept = tvs_variant(got, take, join, hold, keep); lf = [(k0 + int(t["frame_lost"]), t["lost_by"]) for t in kept]
                    h9, _ = LR.match(tr["loss_s9_frames"], lf); ha, ma = LR.match(tr["loss_frames"], lf)
                    a = acc.setdefault((take, join, hold, keep, nz), {"s9": [], "all": [], "f9": [], "fa": [], "real": []})
                    a["s9"].append(len(lf) / len(tr["loss_s9_frames"])); a["all"].append(len(lf) / len(tr["loss_frames"]))
                    a["f9"].append(h9 / len(tr["loss_s9_frames"])); a["fa"].append(ha / len(tr["loss_frames"])); a["real"].append(ma / max(1, len(lf)))
                print("done", g, per, nz, flush=True); del got; gc.collect()
        del fr, ev; gc.collect()
    res = []
    for (take, join, hold, keep, nz), a in acc.items():
        m = lambda v: float(np.mean(v))
        r = {"take": take, "join": join, "hold": hold, "keep": keep, "noise": nz, "count_vs_s9_x": round(m(a["s9"]), 2), "count_vs_all_x": round(m(a["all"]), 2),
             "found_s9_pct": round(100 * m(a["f9"])), "found_all_pct": round(100 * m(a["fa"])), "ours_real_pct": round(100 * m(a["real"]))}
        res.append(r); print(json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-29e/lab.json")
