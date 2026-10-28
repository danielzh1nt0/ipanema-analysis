"""N0-28f: are the app's per-player positions right when our camera only shows part of the pitch?

What the app reads (stats.json players[]): avg_x_m / avg_y_m (average position, also the pass network's fallback node),
pct_def_third / pct_mid_third / pct_att_third (time in each third), and stats.heatmaps[id] (a 12 x 8 grid per track).
analytics.stats takes them from the frames where the player is IN THE PICTURE. Our follow-cam shows ~30% of the pitch
around the ball, so a player is mostly measured when the ball is near him. Never checked against a truth.

Truth = Metrica's free pro tracking (outfield players, every team attacking right, frames with a ball). Each moment is
seen through a REAL camera view of our app clips slid onto Metrica's ball, with our track breaks and jitter (the
N0-22e / N0-25b machinery). Keepers left out (as in the team checks).

Rules (per player):
  seen     - today's rule, but with one id per player for the whole half (what M2 will give)
  piece    - today's app as it is: every broken track piece its own 'player' (scored against its real player's truth)
  piecehold20 - the same pieces, each piece's last spot held 20 s (what hold20 would do to today's pieces)
  hold<h>  - seen + the player's last seen spot kept for h s after he leaves the picture (the held_points idea of
             N0-22e / N0-25b, per player; edge rule: no held spot within 1.5 m of a line; cleared at a period change)
Scores per player-half: pos = distance of the average position from the truth (m), dx = x error (+ = too far forward),
thirds = mean abs error of the 3 shares (pts), tv = heat map (12 x 8) in the wrong place (%); per team-half: rank =
Spearman of the players' average x (who is deepest / highest) vs the truth.

Usage: python tools/n0_28f_playerpos.py <metrica sample-data/data dir> [--out results/qa/n0-28f/metrica_playerpos.json]
       python tools/n0_28f_playerpos.py --dry"""
import sys, os, json, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import n0_22e_shape_view as SV
from n0_28e_passnet import spearman

GRID = (12, 8)
L, W = 105.0, 68.0
EDGE_M = 1.5
HOLDS = (5, 10, 20, 60)
CLIPS = ["results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"]


def build(frames, views, Lv, Wv, seed=0, step=5, breaks=None, jit=None, every=2):
    """-> list of sampled moments: {period, truth: {pid: xy}, rows: [(piece_id, pid, team, xy)]}, fps. Oriented: each team
    attacks right. pid = period * 1000 + Metrica id (a player per half)."""
    rng = np.random.default_rng(seed); ks = sorted(frames)[::step]; fps = 25.0 / step
    meanx = {}
    for k in ks[::4]:
        p, ps, _ = frames[k]
        for i, x, y in ps: meanx.setdefault((p, i >= 200), {}).setdefault(i, []).append(x)
    gk, right = {}, {}
    for (p, away), d in meanx.items():
        m = {i: np.mean(v) for i, v in d.items() if len(v) > 10}; deep_left = np.mean(list(m.values())) < L / 2
        gk[(p, away)] = min(m, key=m.get) if deep_left else max(m, key=m.get); right[(p, away)] = bool(deep_left)
    allids = sorted({i for k in ks for i, _, _ in frames[k][1]})
    plan = {i: SV.detection_plan(len(ks), fps, rng, *(breaks or (None, None))) for i in allids}
    out = []; vi = 0
    for j, k in enumerate(ks):
        p, ps, b = frames[k]
        if j % every == 0: vi += 1
        if b is None: continue
        m, cx = views[vi % len(views)]; inside = SV.place(m, cx, b[0], Lv, Wv, L, W)
        tr, rows = {}, []
        for i, x, y in ps:
            key = (p, i >= 200)
            if key not in gk or i == gk[key]: continue
            tm = "B" if i >= 200 else "A"; ox = x if right[key] else L - x; pid = p * 1000 + i
            tr[pid] = (tm, np.array([ox, y]))
            c = plan[i][j]
            if c >= 0 and inside(np.array([x]), np.array([y]))[0]:
                e = jit[rng.integers(0, len(jit))] if jit is not None else 0.0
                rows.append(((pid * 1000) + c, pid, tm, np.array([ox, y]) + e))
        out.append({"period": p, "truth": tr, "rows": rows})
    return out, fps


def _off(m): return m[0] < EDGE_M or m[0] > L - EDGE_M or m[1] < EDGE_M or m[1] > W - EDGE_M


def positions(moments, fps, holds=HOLDS):
    """-> {rule: {id: [xy, ...]}} for truth / seen / piece / hold<h>; piece ids map to their player via id // 1000"""
    P = {r: collections.defaultdict(list) for r in ["truth", "seen", "piece", "piecehold20"] + [f"hold{h}" for h in holds]}
    last = {h: {} for h in holds}; plast = {}; prev_p = None
    for j, mo in enumerate(moments):
        if mo["period"] != prev_p:
            for h in holds: last[h].clear()
            plast.clear(); prev_p = mo["period"]
        for pid, (_, xy) in mo["truth"].items(): P["truth"][pid].append(xy)
        seen = set()
        for piece, pid, tm, xy in mo["rows"]:
            P["seen"][pid].append(xy); P["piece"][piece].append(xy); seen.add(pid)
        seenp = {r[0] for r in mo["rows"]}
        for piece, pid, tm, xy in mo["rows"]: plast[piece] = (xy, j)
        for q in [q for q, v in plast.items() if q not in seenp and (j - v[1] > 20 * fps + 1e-9 or _off(v[0]))]: del plast[q]
        for q, (xy, _) in plast.items(): P["piecehold20"][q].append(xy)
        for h in holds:
            lh = last[h]
            for piece, pid, tm, xy in mo["rows"]: lh[pid] = (xy, j)
            for pid in [q for q, v in lh.items() if q not in seen and (j - v[1] > h * fps + 1e-9 or _off(v[0]))]: del lh[pid]
            for pid, (xy, _) in lh.items(): P[f"hold{h}"][pid].append(xy)
    return P


def player_stats(pts):
    """the analytics.stats fields from a list of (oriented) spots"""
    A = np.asarray(pts, float); xs = A[:, 0]
    g = np.zeros(GRID); np.add.at(g, (np.clip((A[:, 0] / L * GRID[0]).astype(int), 0, GRID[0] - 1), np.clip((A[:, 1] / W * GRID[1]).astype(int), 0, GRID[1] - 1)), 1)
    return {"avg": A.mean(axis=0), "thirds": np.array([(xs < L / 3).mean(), ((xs >= L / 3) & (xs < 2 * L / 3)).mean(), (xs >= 2 * L / 3).mean()]) * 100, "grid": g / g.sum(), "n": len(A)}


def score(o, t):
    d = o["avg"] - t["avg"]
    return {"pos": float(np.hypot(*d)), "dx": float(d[0]), "dy": float(d[1]), "thirds": float(np.abs(o["thirds"] - t["thirds"]).mean()), "tv": float(0.5 * np.abs(o["grid"] - t["grid"]).sum() * 100)}


def evaluate(moments, fps, min_s=2.0):
    """-> {rule: {pos, dx, thirds, tv (medians over players), pos_p90, rank (median over team-halves), n}}, per-player rows"""
    P = positions(moments, fps); T = {pid: player_stats(v) for pid, v in P["truth"].items()}
    team = {}
    for mo in moments:
        for pid, (tm, _) in mo["truth"].items(): team[pid] = (mo["period"], tm)
    res, rows = {}, {}
    for r, d in P.items():
        if r == "truth": continue
        sc, w = [], []
        for i, v in d.items():
            pid = i // 1000 if r.startswith("piece") else i
            if len(v) / fps < min_s or pid not in T: continue
            sc.append(score(player_stats(v), T[pid])); w.append(len(v))
        rk = []
        if not r.startswith("piece"):
            for key in set(team.values()):
                ids = [pid for pid, k in team.items() if k == key and pid in d and len(d[pid]) / fps >= min_s]
                s = spearman([T[i]["avg"][0] for i in ids], [np.mean(np.asarray(d[i])[:, 0]) for i in ids])
                if s is not None: rk.append(s)
        if not sc: continue
        wa = np.asarray(w, float)
        res[r] = {q: round(float(np.median([s[q] for s in sc])), 1) for q in ("pos", "dx", "dy", "thirds", "tv")}
        res[r] |= {"pos_p90": round(float(np.percentile([s["pos"] for s in sc], 90)), 1), "pos_timeweighted": round(float(np.average([s["pos"] for s in sc], weights=wa)), 1),
                   "rank": round(float(np.median(rk)), 2) if rk else None, "n": len(sc)}
        rows[r] = sc
    seen_share = [len(P["seen"].get(pid, [])) / len(v) for pid, v in P["truth"].items()]
    res["seen_share_median"] = round(float(np.median(seen_share)), 2)
    return res, rows


def dry():
    """stand-ins: 10 players spread 10-90 m, the ball swings 20-85 m, the camera shows 45 m around it -> the far players
    are measured only when the ball comes to them, so their average is pulled toward the ball's side; holding helps"""
    rng = np.random.default_rng(1); fr = {}
    for k in range(6000):
        bx = 52.5 + 32 * np.sin(k / 300)
        ps = [(100 + i, 8 + 9 * i + 4 * np.sin(k / 300) + rng.normal(0, .3), 6 + 6 * i) for i in range(11)] + [(200 + i, 97 - 9 * i, 6 + 6 * i) for i in range(11)]
        fr[k] = (1, ps, (bx, 34.0))
    m = np.zeros((106, 33), bool); m[:45] = True
    mo, fps = build(fr, [(m, 0.21)], 105.0, 64.0, step=5, breaks=(np.array([3.0, 6.0]), np.array([1.0, 3.0])))
    return evaluate(mo, fps)[0]


def main(a):
    if "--dry" in a: print(json.dumps(dry(), indent=1)); return
    import n0_22c_metrica as C
    out_p = a[a.index("--out") + 1] if "--out" in a else None; d = [x for x in a if not x.startswith("--") and x != out_p][0]
    runs, gaps, jit = C.our_breaks(CLIPS); res = {}; ex = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr = SV.load_metrica(d, g)
        for c in CLIPS:
            v, Lv, Wv = SV.view_masks(c)
            for seed in (0, 1):
                mo, fps = build(fr, v, Lv, Wv, breaks=(runs, gaps), jit=jit, seed=seed)
                nm = f"game{g[-1]}_{c.split('/')[-2][:9]}_seed{seed}"; res[nm], rows = evaluate(mo, fps)
                if nm == "game2_SFKBP1109_seed0":
                    P = positions(mo, fps)
                    ex = {r: {str(i): np.mean(np.asarray(P[r][i]), axis=0).round(1).tolist() for i in P[r] if i // 1000 == 1 and len(P[r][i]) / fps >= 2} for r in ("truth", "seen", "hold10", "hold20")}
                print(nm, " | ".join(f"{r}: {s['pos']} m / {s['thirds']} pts / rank {s['rank']}" for r, s in res[nm].items() if isinstance(s, dict)), flush=True)
    rules = [r for r in next(iter(res.values())) if isinstance(res[next(iter(res))][r], dict)]
    summ = {r: {q: round(float(np.mean([res[n][r][q] for n in res if res[n][r][q] is not None])), 2 if q == "rank" else 1) for q in ("pos", "dx", "dy", "thirds", "tv", "pos_p90", "pos_timeweighted", "rank") if any(res[n][r][q] is not None for n in res)}
            | {"pos_worst_run": max(res[n][r]["pos"] for n in res), "better_than_seen_runs": sum(res[n][r]["pos"] < res[n]["seen"]["pos"] for n in res) if r != "seen" else None,
               "heat_better_than_seen_runs": sum(res[n][r]["tv"] < res[n]["seen"]["tv"] for n in res) if r != "seen" else None,
               "thirds_better_than_seen_runs": sum(res[n][r]["thirds"] < res[n]["seen"]["thirds"] for n in res) if r != "seen" else None} for r in rules}
    summ["seen_share_median"] = round(float(np.mean([res[n]["seen_share_median"] for n in res])), 2)
    print(json.dumps(summ, indent=1))
    if out_p:
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        json.dump({"summary": summ, "runs": res, "example_game2_half1_SFKBP_view": ex}, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
