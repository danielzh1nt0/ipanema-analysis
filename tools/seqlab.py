"""E5 (29 Sep): possession sequences on the new possession state (possession_simple).
Answer key: Metrica's free pro games (hand-labelled events). A true sequence = a chain of on-ball events (PASS, CARRY,
SHOT, SET PIECE, RECOVERY, BALL LOST) by one team; it ends when the other team acts or the ball goes out.
Ours = P.sequences on possession_simple (and the old viterbi state for reference), clean positions and 'streak' noise
(errors in runs, like our Veo tracking; same noise as tools/passlab.py). A true sequence is found if one of ours of the
same team starts within +-TOL s of it. Also our own footage (SFK-BP, Reymersholm; no sequence key there): count, median
length, and the who-has-the-ball keys (must not change: the state is not touched).
Free, CPU, ~5 min first time (then cached in PASSLAB_CACHE).
    PYTHONPATH=. python tools/seqlab.py <metrica data dir> -> results/possession/e5_2026-09-29.json"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P
import tools.passlab as PL
FPS = PL.FPS; TOL = 2.0
ONBALL = ("PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST")
GRID = [(0.0, 1.0)] + [(t, j) for t in (0.3, 0.5, 1.0) for j in (1.0, 3.0, 6.0)]   # (take_s, join_s); (0, 1) = rule before E5


def truth_sequences(ev, per):
    E = sorted([e for e in ev if e["period"] == per], key=lambda e: (e["f0"], e["f1"])); out = []; cur = None
    for e in E:
        if e["type"] == "BALL OUT":
            if cur: out.append(cur); cur = None
            continue
        if e["type"] not in ONBALL or e["team"] not in ("Home", "Away"): continue
        tm = "A" if e["team"] == "Home" else "B"
        if cur and cur["team"] == tm: cur["end"] = max(cur["end"], e["f1"] or e["f0"]); continue
        if cur: out.append(cur)
        cur = {"team": tm, "start": e["f0"], "end": max(e["f0"], e["f1"] or e["f0"])}
    if cur: out.append(cur)
    return out


def found(tr, ours, tol=TOL * FPS):
    pairs = sorted((abs(o["start"] - t["start"]), i, j) for i, t in enumerate(tr) for j, o in enumerate(ours) if o["team"] == t["team"] and abs(o["start"] - t["start"]) <= tol)
    ut, uo = set(), set()
    for _, i, j in pairs:
        if i in ut or j in uo: continue
        ut.add(i); uo.add(j)
    return len(ut), len(uo)


def summ(seqs, fps):
    d = [(s["end"] - s["start"] + 1) / fps for s in seqs]
    return {"count": len(seqs), "median_s": round(float(np.median(d)), 1) if d else None}


def metrica(root):
    rows = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        _, ev = PL.load_game(root, g)
        for per in (1, 2):
            tr = truth_sequences(ev, per)
            for noisy in (False, "streak"):
                S = PL.cached(root, g, per, noisy); k0 = S["k0"]
                trs = [dict(t, start=t["start"] - k0, end=t["end"] - k0) for t in tr]
                r = {"game": g[-1], "half": per, "noise": str(noisy), "truth": summ(trs, FPS)}
                for name, st in (("viterbi", S["state_viterbi"]), ("simple", S["state_simple"])):
                    for t, j in (GRID if name == "simple" else [(0.0, 1.0)]):
                        sq = P.sequences(st, S["bm"], {}, FPS, PL.ML.L, S["ar"], take_s=t, join_s=j)
                        ft, fo = found(trs, sq)
                        r[f"{name} take {t} join {j}"] = dict(summ(sq, FPS), found=ft, real=fo)
                rows.append(r); print(json.dumps(r), flush=True)
    return rows


def own():
    import tools.e4lab as E4
    out = {}
    for c in (E4.sfk(), E4.reym()):
        per, ball, H, L, W, fps = c["per"], c["ball"], c["H"], c["L"], c["W"], c["fps"]
        fr, bm = P.carriers(per, ball, H, E4.CARRIER_R, E4.NEAR_R); o = {}
        for mode in ("viterbi", "simple"):
            st, bs, ds, info = P.pipeline_state(per, ball, bm, H, fps, L, W, mode=mode, boxh=c["boxh"], log=E4.q)
            ar, _ = P.direction(st, bm, log=E4.q)
            o[mode] = {"who_has_ball_right": E4.grade(st, c["key"])}
            for t, j in (GRID if mode == "simple" else [(0.0, 1.0)]):
                o[mode][f"take {t} join {j}"] = summ(P.sequences(st, bm, bs, fps, L, ar, take_s=t, join_s=j), fps)
            o[mode]["pipeline"] = summ(P.sequences(st, bm, bs, fps, L, ar, **info["seq"]), fps)
        out[c["name"]] = o; print(c["name"], json.dumps(o), flush=True)
    return out


if __name__ == "__main__":
    res = {"own_footage": own()}
    if len(sys.argv) > 1:
        res["metrica"] = metrica(sys.argv[1]); tot = {}
        for r in res["metrica"]:
            for k, v in r.items():
                if isinstance(v, dict) and "found" in v:
                    a = tot.setdefault(f"{r['noise']:6s} {k}", [0, 0, 0]); a[0] += r["truth"]["count"]; a[1] += v["count"]; a[2] += v["found"]
        res["metrica_total"] = {k: {"truth": t, "ours": c, "found_pct": round(100 * f / t), "real_pct": round(100 * f / max(1, c))} for k, (t, c, f) in tot.items()}
        for k, v in res["metrica_total"].items(): print(k, v)
    os.makedirs("results/possession", exist_ok=True)
    json.dump(res, open("results/possession/e5_2026-09-29.json", "w"), indent=1)
