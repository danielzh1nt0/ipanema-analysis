"""N0 19 Oct: old (first match) vs new (nearest match, one bonus per finder guess) RF-DETR + WASB fusion on the SFK-BP
test clip, through the app's picker, on every raw finder cache in the repo (3 Kaggle jobs' old/new finders + the clicks
finder), always with the WASB file the app uses (blobs, thr 0.05). Keys: 34 checked balls, B4 (285), who-has-the-ball
(99, results/review/who_answers.json, pipeline state), 28 graded passes (real kept / fake kept).
    PYTHONPATH=. python tools/fuse_check.py  -> results/qa/n0-19e/fuse_check.json"""
import sys, os, json, glob, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, analytics as AN
q = lambda *a: None; CLIP = "SFKBP1109_s1200"; OUT = "results/qa/n0-19e"
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
P_ = pickle.load(open(f"results/volume/cache/{CLIP}/picker_inputs.pkl", "rb")); fps, H, L, W = P_["fps"], P_["H"], P_["L"], P_["W"]
per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
sgt = {int(k): v[:2] for k, v in json.load(open(f"results/volume/reference/{CLIP}/ball_gt.json")).items() if v}
b4 = [m for m in json.load(open("results/ball/b4/key.json"))["moments"] if m["verdict"] == "ball"]
WHO = [a for a in json.load(open("results/review/who_answers.json"))["moments"] if a["truth"] in ("dark", "white", "loose")]
PA = json.load(open("results/review/passcheck_answers.json")); fake = set(PA["fake"]); unsure = set(PA["unsure"]); graded = [i for i in range(PA["graded_first_n"]) if i not in unsure]
exp = json.load(open(f"results/volume/runs/matches/{CLIP}/stats.json"))["passes"]

def load(p):
    o = pickle.load(open(p, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}

def run(cands):
    ball = BL.bridge(BL.pick_v2(cands, H, L, W, per=per, fps=fps, log=q), fps)
    keys = {"sfk34": int(sum(on(ball, f, *g) for f, g in sgt.items())), "b4": int(sum(on(ball, m["frame"], m["x"], m["y"]) for m in b4))}
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q); ar, _ = P.direction(state, ballm, log=q)
    keys["who99"] = int(sum(int(state[a["frame"]]) == {"dark": 0, "white": 1, "loose": 2}[a["truth"]] for a in WHO if a["frame"] < len(state)))
    tvs = P.turnovers(per, frames_, state, ballm, fps, ar, 2.0, 5.0, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
    ln = AN.lanes(per, frames_, ar, 1.5, 30.0); ps, _ = AN.passes(per, frames_, tvs, ln, ar, fps)
    keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
    return {**keys, "passes": len(ps), "real_kept": sum(k for i, k in zip(graded, keep) if i not in fake), "fake_kept": sum(k for i, k in zip(graded, keep) if i in fake), "turnovers": len(tvs)}

def changed(a, b):
    """frames whose fused guess list differs"""
    return sum(1 for k in set(a) | set(b) if sorted(a.get(k, [])) != sorted(b.get(k, [])))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    wb = load(f"results/volume/cache/{CLIP}/ball_cands_wasb_1790008894_t2x2_thr0.05.pkl")
    finders = {f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)[6:9]}": p for p in sorted(glob.glob(f"results/kaggle/ballfinder_*/cands_*_{CLIP}.pkl"))}
    finders["clicks_round_20260928"] = f"results/volume/cache/{CLIP}/ball_cands_clicks_round_20260928_0108.pkl"
    out = {"app inputs (already fused, for reference)": run(P_["cands"])}; print("app inputs", out["app inputs (already fused, for reference)"], flush=True)
    for name, p in finders.items():
        rf = load(p); fo = BL.fuse_candidates(rf, wb, nearest=False); fn = BL.fuse_candidates(rf, wb, nearest=True)
        ro, rn = run(fo), run(fn); out[name] = {"old": ro, "new": rn, "frames_changed": changed(fo, fn), "frames": len(fo)}
        print(f"{name:32s} frames changed {out[name]['frames_changed']}/{len(fo)} | old {ro} | new {rn}", flush=True)
    json.dump(out, open(f"{OUT}/fuse_check.json", "w"), indent=1)

def ab_moments(seed=19, per_spell=3):
    """blind A/B moments where the app finder's old and new fusion picks differ (> 30 px): up to per_spell frames per spell,
    side A/B randomly old or new; the key (which side is new) goes to a separate file, opened only after grading"""
    wb = load(f"results/volume/cache/{CLIP}/ball_cands_wasb_1790008894_t2x2_thr0.05.pkl")
    rf = load("results/kaggle/ballfinder_feet/cands_old_SFKBP1109_s1200.pkl")      # same-job copy of the app finder (B7)
    bo = BL.bridge(BL.pick_v2(BL.fuse_candidates(rf, wb, nearest=False), H, L, W, per=per, fps=fps, log=q), fps)
    bn = BL.bridge(BL.pick_v2(BL.fuse_candidates(rf, wb, nearest=True), H, L, W, per=per, fps=fps, log=q), fps)
    diff = [k for k in sorted(set(bo) & set(bn)) if np.hypot(bo[k][0] - bn[k][0], bo[k][1] - bn[k][1]) > 30]
    spells = []
    for k in diff:
        if spells and k - spells[-1][-1] <= 3: spells[-1].append(k)
        else: spells.append([k])
    rng = np.random.default_rng(seed); moments, key = [], {}
    for s in spells:
        pick = sorted(set(s[int(round(j))] for j in np.linspace(0, len(s) - 1, min(per_spell, len(s)))))
        for f in pick:
            i = len(key); new_is_b = bool(rng.random() < 0.5); key[i] = {"frame": f, "t": round(f / fps, 2), "new_side": "B" if new_is_b else "A"}
            a, b = (bo[f], bn[f]) if new_is_b else (bn[f], bo[f])
            moments += [{"id": i, "frame": f, "x": round(a[0], 1), "y": round(a[1], 1), "pick": "old"},   # 'old' folder = side A
                        {"id": i, "frame": f, "x": round(b[0], 1), "y": round(b[1], 1), "pick": "new"}]   # 'new' folder = side B
    return {"clip": CLIP, "note": "sheet folder 'old' = side A, 'new' = side B (blind, see ab_key.json)", "moments": moments}, key, len(diff), len(spells)
