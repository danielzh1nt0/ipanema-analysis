"""BALL-EDGE (23 Oct, local, free): old vs IPANEMA_BALL_EDGE=1 through the app picker on both app test clips.
Input: the raw tile guesses of the whole clip (results/qa/ball-edge/raw/<clip>_<i>of<n>.npz, tools/ball_edge_shard.py
on the free runner). Both arms are merged from the SAME tile guesses (ballrf.merge, edge off / on) and fused with the
SAME WASB guesses, so CPU-vs-GPU drift in the re-detection cancels out.
WASB guesses: the clip's app inputs (picker_inputs.pkl) only hold the fused list, so they are read back out of it:
fused = finder guesses first (score 1.5 c, + WASB s + 0.8 when a WASB guess sits within 12 px), then the WASB-only ones
(score s), all / 2. A fused guess at a re-detected finder guess gives s = 2 f - 1.5 c - 0.8; any other fused guess is
WASB-only (s = 2 f). For SFK-BP the real WASB file is in the cache, so the read-back is checked against it.
Grades: SFK-BP 34 checked balls, B4 ball moments, 99 who-has-the-ball, 28 graded passes (real / fake kept), flips,
sequences; AIK 39 checked balls. Writes results/qa/ball-edge/grade.json.
    PYTHONPATH=. python tools/ball_edge_grade.py [raw_dir] [out_dir]"""
import sys, os, glob, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, ballrf as BR

SFK, AIK = "SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"
KEY = {"sfk": SFK, "aik": AIK}
q = lambda *a: None
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30


def load_raw(raw_dir, key):
    """all shards of a clip -> {frame: [(x, y, c, cut)]}, and the frame count covered"""
    out = {}
    for p in sorted(glob.glob(f"{raw_dir}/{key}_*of*.npz")):
        d = np.load(p)
        for f, x, y, c, cut in zip(d["f"].tolist(), d["x"].tolist(), d["y"].tolist(), d["c"].tolist(), d["cut"].tolist()):
            out.setdefault(int(f), []).append((x, y, c, bool(cut)))
    return out


def wasb_back(fused, rf, px=1.5):
    """one frame: app fused list + the re-detected finder guesses (old merge) -> (WASB guesses [(x, y, s)], n fused guesses
    matched to a finder guess)"""
    W = []; used = set(); m = 0
    for x, y, f in fused:
        j = next((i for i, g in enumerate(rf) if i not in used and np.hypot(g[0] - x, g[1] - y) <= px), None)
        if j is not None:
            used.add(j); m += 1; s = 2 * f - 1.5 * rf[j][2] - BL_BONUS
            if f >= 0.989: CAPPED[0] += 1; s = max(s, 0.05)      # capped at 0.99: a WASB guess was there, s only known >= this
            if s > 0.02: W.append((x, y, float(min(1.0, s))))
        else: W.append((x, y, float(min(1.0, 2 * f))))
    return W, m


BL_BONUS = 0.8
CAPPED = [0]


def arms(raw, fused_all):
    """{frame: guesses} for app / old / new from the same tile guesses and the same read-back WASB"""
    old, new, W = {}, {}, {}; st = {"frames": 0, "fused_rf_matched": 0, "rf_old": 0, "rf_new": 0, "no_raw": 0}
    for k, F in fused_all.items():
        r = raw.get(k)
        if r is None: st["no_raw"] += 1; r = []
        o, n = BR.merge(r, False), BR.merge(r, True)
        w, m = wasb_back(F, o); W[k] = w; old[k], new[k] = o, n
        st["frames"] += 1; st["fused_rf_matched"] += m; st["rf_old"] += len(o); st["rf_new"] += len(n)
    st["capped_fused"] = CAPPED[0]; CAPPED[0] = 0
    st["dropped_capped"] = sum(1 for k in old for g in old[k] if g[2] * 1.5 + BL_BONUS >= 1.97 and all(np.hypot(g[0] - h[0], g[1] - h[1]) > 12 for h in new[k]))
    return BL.fuse_candidates(old, W), BL.fuse_candidates(new, W), W, old, new, st


def load_inputs(clip):
    P = pickle.load(open(f"results/volume/cache/{clip}/picker_inputs.pkl", "rb"))
    P["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P["per"].items()}
    return P


def pick(P, cands): return BL.bridge(BL.pick_v2(cands, P["H"], P["L"], P["W"], per=P["per"], fps=P["fps"], log=q), P["fps"])


class SfkGrader:
    def __init__(self, P):
        from ipanema import possession as PO, analytics as AN
        self.P, self.PO, self.AN = P, PO, AN
        self.gt = {int(k): v[:2] for k, v in json.load(open(f"results/volume/reference/{SFK}/ball_gt.json")).items() if v}
        self.b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
        PA = json.load(open("results/review/passcheck_answers.json")); self.fake = set(PA["fake"]); uns = set(PA["unsure"])
        self.graded = [i for i in range(PA["graded_first_n"]) if i not in uns]
        self.exp = json.load(open(f"results/volume/runs/matches/{SFK}/stats.json"))["passes"]
        E4 = {"__file__": os.path.abspath("tools/e4lab.py")}
        exec(compile(open("tools/e4lab.py").read().split("\nif __name__")[0], "e4lab", "exec"), E4); self.E4 = E4
        n = len(P["H"])
        self.c = dict(name="SFK-BP", per={k: P["per"].get(k, []) for k in range(n)}, H=[P["H"][k] for k in range(n)], L=P["L"], W=P["W"],
                      fps=P["fps"], boxh=None, key=json.load(open("results/review/who_answers.json"))["moments"])

    def __call__(self, ball):
        P, PO, AN = self.P, self.PO, self.AN; fps, H, L, W, per = P["fps"], P["H"], P["L"], P["W"], P["per"]
        g = {"sfk34": int(sum(on(ball, f, *x) for f, x in self.gt.items())), "b4": int(sum(on(ball, m["frame"], m["x"], m["y"]) for m in self.b4)), "b4_of": len(self.b4)}
        fr, bm = PO.carriers(per, ball, H, 2.5, 5.0); state, bs, _, pinfo = PO.pipeline_state(per, ball, bm, H, fps, L, W, log=q); ar, _ = PO.direction(state, bm, log=q)
        tvs = PO.turnovers(per, fr, state, bm, fps, ar, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
        ln = AN.lanes(per, fr, ar, 1.5, 30.0); ps, _ = AN.passes(per, fr, tvs, ln, ar, fps); seqs = PO.sequences(state, bm, bs, fps, L, ar, **pinfo["seq"])
        keep = [any(abs(p["t"] - self.exp[i]["t"]) < 0.3 for p in ps) for i in self.graded]
        g.update({"passes": len(ps), "real_kept": int(sum(k for i, k in zip(self.graded, keep) if i not in self.fake)),
                  "real_of": sum(1 for i in self.graded if i not in self.fake), "fake_kept": int(sum(k for i, k in zip(self.graded, keep) if i in self.fake)),
                  "fake_of": sum(1 for i in self.graded if i in self.fake), "turnovers": len(tvs), "sequences": len(seqs),
                  "flips": int(sum(1 for i in range(1, len(state)) if state[i] < 2 and state[i - 1] < 2 and state[i] != state[i - 1]))})
        self.c["ball"] = ball; s = self.E4["stats"](self.c, "simple"); g["who99"] = s["who_has_ball_right"]
        return g


def diff_frames(a, b, px=30):
    ks = set(a) | set(b)
    return sorted(k for k in ks if (k in a) != (k in b) or (k in a and np.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]) > px))


def main(raw_dir="results/qa/ball-edge/raw", out="results/qa/ball-edge", clips=("sfk", "aik"), graders=None):
    os.makedirs(out, exist_ok=True); res = {}; picks = {}
    for key in clips:
        clip = KEY[key]; P = load_inputs(clip); raw = load_raw(raw_dir, key)
        print(f"{key}: raw frames {len(raw)}, app frames {len(P['cands'])}", flush=True)
        co, cn, W, ro, rn, st = arms(raw, P["cands"])
        G = (graders or {}).get(key) or (SfkGrader(P) if key == "sfk" else
             (lambda b, gt={int(k): v for k, v in json.load(open(f"reference/{AIK}/ball_gt.json")).items()}: {"aik39": int(sum(on(b, f, *x[:2]) for f, x in gt.items() if x))}))
        r = {"stats": st, "dropped_rf_guesses": st["rf_old"] - st["rf_new"]}
        B = {}
        for nm, cd in (("app inputs", P["cands"]), ("old (re-detected)", co), ("new (BALL_EDGE=1)", cn)):
            B[nm] = pick(P, cd); r[nm] = G(B[nm]); print(f"  {nm:20s}", r[nm], flush=True)
        if key == "sfk":                                           # read-back WASB vs the real WASB files (exact fuse)
            for p in sorted(glob.glob(f"results/volume/cache/{SFK}/ball_cands_wasb_*.pkl")):
                o = pickle.load(open(p, "rb")); o = o[0] if isinstance(o, tuple) else o
                wb = {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
                nm = os.path.basename(p)[:-4]; bo, bn = pick(P, BL.fuse_candidates(ro, wb)), pick(P, BL.fuse_candidates(rn, wb))
                r[f"exact old + {nm}"], r[f"exact new + {nm}"] = G(bo), G(bn)
                print(f"  exact old/new + {nm}", r[f"exact old + {nm}"], r[f"exact new + {nm}"], flush=True)
        d = diff_frames(B["old (re-detected)"], B["new (BALL_EDGE=1)"]); r["pick_differs_frames"] = len(d)
        segs = []
        for f in d:
            if segs and f - segs[-1][1] <= 3: segs[-1][1] = f
            else: segs.append([f, f])
        r["pick_differs_spells"] = segs; picks[key] = {nm: {str(k): [round(v[0], 1), round(v[1], 1)] for k, v in b.items()} for nm, b in B.items() if nm != "app inputs"}
        res[key] = r
        json.dump(res, open(f"{out}/grade.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    json.dump(picks, open(f"{out}/picks.json", "w"))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:3])
