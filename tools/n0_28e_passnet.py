"""N0-28e: is the app's pass network (stats.metrics.pass_network: who passes to whom, edges with counts, nodes = where a
player played / received) right, and is the pass length on every pass event ('9.0 m, gain 6.1 m') right? Never checked
on pro data. Our app copies have track pieces, not players (M2), so this measures what the network will show ONCE M2
gives one id per player: Metrica's ids are perfect, only the ball and the player view are degraded.

Truth = Metrica's PASS events (completed; From / To players; Start / End = the BALL's spots).
  edge count  = passes From -> To per team;  node = mean of the player's pass start spots and reception end spots.
Ours = the pipeline on Metrica's positions (tools/metricalab.ours -> analytics.passes -> metrics.pass_network), perfect
ball + our 'streak' / 'flicker' ball noise (2 seeds), 4 halves = 20 runs.
Scores per team-half:
  weighted overlap = sum min(ours, truth) / sum max(ours, truth) over all ordered pairs (1 = identical network)
  top-10 overlap   = of the truth's 10 busiest pairs, how many are in our 10 busiest
  spearman         = rank agreement of pair counts over all pairs with >= 1 pass in either
  busiest pair right, node error (median m, players in both), passes per edge vs truth
Pass length (matched passes: same team, completed both, passer's last touch within 1 s of the PASS start):
  ours (player to player) vs truth (ball start to end): median error, median |error|, share within 5 m, long (>= 30 m) agreement.

    PYTHONPATH=. python tools/n0_28e_passnet.py <metrica sample-data/data dir> [--out results/qa/n0-28e/metrica_passnet.json]
"""
import sys, os, csv, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN, metrics as MT

L, W, FPS = 105.0, 68.0, 25.0
LONG_M = 30.0


def _pid(team, s): return (100 if team == "Home" else 200) + int(s[6:])


def truth_passes(path, per):
    out = []
    for r in csv.DictReader(open(path)):
        if r["Type"] != "PASS" or r["Team"] not in ("Home", "Away") or int(r["Period"]) != per: continue
        if not (r["From"].startswith("Player") and r["To"].startswith("Player")): continue
        f = lambda s: None if s in ("", "NaN") else float(s)
        x0, y0, x1, y1 = f(r["Start X"]), f(r["Start Y"]), f(r["End X"]), f(r["End Y"])
        if None in (x0, y0, x1, y1): continue
        a, b = np.array([x0 * L, y0 * W]), np.array([x1 * L, y1 * W])
        out.append({"team": "A" if r["Team"] == "Home" else "B", "from": _pid(r["Team"], r["From"]), "to": _pid(r["Team"], r["To"]),
                    "f0": int(r["Start Frame"]), "from_m": a, "to_m": b, "length_m": float(np.linalg.norm(b - a)), "completed": True})
    return out


def network(ps):
    """{team: ({(from, to): count}, {player: node xy})} - edges counted like metrics.pass_network, nodes from the spots"""
    out = {}
    for tm in ("A", "B"):
        e, sp = {}, {}
        for p in ps:
            if p["team"] != tm or not p["completed"]: continue
            k = (int(p["from"]), int(p["to"])); e[k] = e.get(k, 0) + 1
            sp.setdefault(int(p["from"]), []).append(np.asarray(p["from_m"], float)); sp.setdefault(int(p["to"]), []).append(np.asarray(p["to_m"], float))
        out[tm] = (e, {i: np.mean(v, axis=0) for i, v in sp.items()})
    return out


def from_export(net):
    """metrics.pass_network output -> same shape as network()"""
    out = {}
    for tm, v in net.items():
        out[tm] = ({(int(e["from"]), int(e["to"])): e["count"] for e in v["edges"]}, {int(i): np.array([n["x"], n["y"]]) for i, n in v["nodes"].items()})
    return out


def spearman(a, b):
    if len(a) < 3: return None
    def avg(x):                                                                # ranks, ties averaged
        x = np.asarray(x, float); r = np.empty(len(x)); o = np.argsort(x, kind="mergesort"); xs = x[o]; i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and xs[j + 1] == xs[i]: j += 1
            r[o[i:j + 1]] = (i + j) / 2; i = j + 1
        return r
    ra, rb = avg(a), avg(b)
    if ra.std() == 0 or rb.std() == 0: return None
    return float(np.corrcoef(ra, rb)[0, 1])


def volume_null(te):
    """the strongest 'no pair information' network: each pair gets passes_from(i) * passes_to(j) / total (who is busy, not
    who passes to whom) - an overlap no better than this means the pair counts add nothing beyond player volume"""
    out_, in_ = {}, {}
    for (i, j), c in te.items(): out_[i] = out_.get(i, 0) + c; in_[j] = in_.get(j, 0) + c
    tot = max(1, sum(te.values()))
    return {(i, j): out_[i] * in_[j] / tot for i in out_ for j in in_ if i != j}


def overlap(te, oe):
    keys = set(te) | set(oe)
    return sum(min(te.get(k, 0), oe.get(k, 0)) for k in keys) / max(1e-9, sum(max(te.get(k, 0), oe.get(k, 0)) for k in keys))


def compare(T, O, top=10):
    te, tn = T; oe, on = O
    keys = sorted(set(te) | set(oe)); t = [te.get(k, 0) for k in keys]; o = [oe.get(k, 0) for k in keys]
    wo = sum(min(a, b) for a, b in zip(t, o)) / max(1, sum(max(a, b) for a, b in zip(t, o)))
    tt = [k for k, _ in sorted(te.items(), key=lambda kv: -kv[1])[:top]]; ot = {k for k, _ in sorted(oe.items(), key=lambda kv: -kv[1])[:top]}
    bt = max(te.values()) if te else 0; busiest = {k for k, v in te.items() if v == bt}
    ob = max(oe, key=oe.get) if oe else None
    ne = [float(np.linalg.norm(tn[i] - on[i])) for i in tn if i in on]
    return {"pairs_truth": len(te), "pairs_ours": len(oe), "passes_truth": sum(te.values()), "passes_ours": sum(oe.values()),
            "weighted_overlap": round(wo, 3), "null_overlap": round(overlap(te, volume_null(te)), 3), "ours_null_overlap": round(overlap(te, volume_null(oe)), 3), "top10_overlap": sum(k in ot for k in tt), "top10_n": len(tt),
            "spearman": None if spearman(t, o) is None else round(spearman(t, o), 3), "busiest_right": ob in busiest,
            "node_err_median_m": round(float(np.median(ne)), 1) if ne else None, "nodes_both": len(ne)}


def match_lengths(T, ours, f0, tol_s=1.0):
    used = set(); pairs = []
    for t in sorted(T, key=lambda p: p["f0"]):
        best = None
        for j, u in enumerate(ours):
            if j in used or u["team"] != t["team"] or not u["completed"]: continue
            dt = abs(f0 + u["t"] * FPS - t["f0"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is not None: used.add(best[1]); pairs.append((t, ours[best[1]]))
    if not pairs: return {"n": 0}
    tl = np.array([a["length_m"] for a, _ in pairs]); ol = np.array([b["length_m"] for _, b in pairs]); err = ol - tl
    lt, lo = tl >= LONG_M, ol >= LONG_M
    return {"n": len(pairs), "truth_median_m": round(float(np.median(tl)), 1), "ours_median_m": round(float(np.median(ol)), 1),
            "err_median_m": round(float(np.median(err)), 2), "abs_err_median_m": round(float(np.median(np.abs(err))), 2),
            "within_5m_pct": round(100 * float((np.abs(err) <= 5).mean()), 1), "long_truth": int(lt.sum()), "long_ours": int(lo.sum()), "long_both": int((lt & lo).sum()),
            "same_pair_pct": round(100 * float(np.mean([(a["from"], a["to"]) == (int(b["from"]), int(b["to"])) for a, b in pairs])), 1)}


def capture(fr, per, noisy, seed):
    """run the pipeline's passes on Metrica (metricalab.ours), return (passes, attack_right)"""
    import metricalab as ML
    got = {}; orig = AN.passes
    def cap(*a, **k):
        r = orig(*a, **k); got["ps"] = r[0]; return r
    AN.passes = cap
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: AN.passes = orig
    return got["ps"], o["attack_right"]


def players_stub(ps):
    """minimal players_ for metrics.pass_network (only used for nodes='avg' or nodes with no spots)"""
    ids = {int(p["from"]) for p in ps} | {int(p["to"]) for p in ps}
    return [{"id": i, "avg_x_m": 0.0, "avg_y_m": 0.0} for i in ids]


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = f"{gd}/{g}_RawEventsData.csv"
        for per in halves:
            T = truth_passes(ev, per); f0 = min(k for k in fr if fr[k][0] == per); TN = network(T)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    ps, ar = capture(fr, per, noisy, seed)
                    net = MT.pass_network(ps, players_stub(ps))                 # exactly what the export writes
                    ON = from_export(net)
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed,
                           "net": {tm: compare(TN[tm], ON[tm]) for tm in ("A", "B")}, "length": match_lengths(T, ps, f0)}
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


def summarise(rows):
    out = {}
    for nz in sorted({r["noise"] for r in rows}):
        R = [r for r in rows if r["noise"] == nz]; N = [r["net"][tm] for r in R for tm in ("A", "B")]
        f = lambda k: [n[k] for n in N if n[k] is not None]
        Lr = [r["length"] for r in R if r["length"].get("n")]
        out[nz] = {"team_halves": len(N), "weighted_overlap_median": round(float(np.median(f("weighted_overlap"))), 3),
                   "weighted_overlap_min": round(float(np.min(f("weighted_overlap"))), 3),
                   "null_overlap_median": round(float(np.median(f("null_overlap"))), 3), "ours_volume_only_median": round(float(np.median(f("ours_null_overlap"))), 3),
                   "top10_overlap_median": float(np.median(f("top10_overlap"))), "spearman_median": round(float(np.median(f("spearman"))), 3),
                   "busiest_right": f"{sum(n['busiest_right'] for n in N)}/{len(N)}", "node_err_median_m": round(float(np.median(f("node_err_median_m"))), 1),
                   "passes_ratio_median": round(float(np.median([n["passes_ours"] / max(1, n["passes_truth"]) for n in N])), 2),
                   "pairs_ratio_median": round(float(np.median([n["pairs_ours"] / max(1, n["pairs_truth"]) for n in N])), 2),
                   "length_err_median_m": round(float(np.median([x["err_median_m"] for x in Lr])), 2),
                   "length_abs_err_median_m": round(float(np.median([x["abs_err_median_m"] for x in Lr])), 2),
                   "length_within_5m_pct": round(float(np.median([x["within_5m_pct"] for x in Lr])), 1),
                   "long_found_pct": round(100 * sum(x["long_both"] for x in Lr) / max(1, sum(x["long_truth"] for x in Lr)), 1),
                   "long_real_pct": round(100 * sum(x["long_both"] for x in Lr) / max(1, sum(x["long_ours"] for x in Lr)), 1),
                   "matched_same_pair_pct": round(float(np.median([x["same_pair_pct"] for x in Lr])), 1)}
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-28e/metrica_passnet.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); s = summarise(rows); print(json.dumps(s, indent=1))
    json.dump({"summary": s, "runs": rows}, open(out, "w"), indent=1)
