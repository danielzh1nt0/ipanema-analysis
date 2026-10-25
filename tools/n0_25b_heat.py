"""N0-25b: is the team heat map right when our camera only shows part of the pitch?

The UI contract (results/app/ui_contract_2026-10-02.md) says the app builds the team heat map (SHOW, Beta) from the
export in one of two ways: all players[].m of a team in the frames, or the sum of stats.heatmaps over the team's
tracks (each track's grid is normalised to 1). Neither was ever checked against a truth.

Truth = Metrica's free pro tracking (all 10 outfield players per team, 25 fps), oriented so every team attacks to the
right. Each Metrica moment is seen through a REAL camera view from our app clips, slid so its centre sits on the ball,
with our track breaks and jitter (the N0-22e machinery: tools/n0_22e_shape_view.py). Keepers left out.

Rules compared (12 x 8 grid, like stats.grid):
  frames    - every seen player in every frame (= sum of per-track grids weighted by track length)
  tracksum  - sum of stats.heatmaps over the team's tracks (each >= 2 s, each normalised to 1: a 2-s track weighs as
              much as a 2-min one)
  hold<h>   - seen players + each track's last seen spot for h s after it leaves the picture (analytics.held_points,
              the shape rule of N0-22e, edge rule on)
Scores per team-half: TV = half the sum of |ours - truth| over the cells (0 = same map, 100 = no overlap), thirds =
mean abs error of the own / middle / attacking third shares (pts), x = error of the mean x (m, + = too far forward).

Usage: python tools/n0_25b_heat.py /tmp/metrica/data [--out results/qa/n0-25b/metrica.json]
       python tools/n0_25b_heat.py --dry"""
import sys, os, json, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN
import n0_22e_shape_view as SV

GRID = (12, 8)
CLIPS = ["results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"]


def cell_counts(pts, L, W, g=None):
    g = np.zeros(GRID) if g is None else g
    if pts is None or len(pts) == 0: return g
    pts = np.asarray(pts, float)
    i = np.clip((pts[:, 0] / L * GRID[0]).astype(int), 0, GRID[0] - 1); j = np.clip((pts[:, 1] / W * GRID[1]).astype(int), 0, GRID[1] - 1)
    np.add.at(g, (i, j), 1)
    return g


def scores(g, truth):
    p = g / max(1e-9, g.sum()); q = truth / max(1e-9, truth.sum())
    tv = 0.5 * np.abs(p - q).sum() * 100
    th = lambda m: np.array([m[:4].sum(), m[4:8].sum(), m[8:].sum()]) * 100
    xc = (np.arange(GRID[0]) + 0.5) / GRID[0]
    return {"tv": round(float(tv), 1), "thirds": round(float(np.abs(th(p) - th(q)).mean()), 1),
            "x": round(float(((p.sum(1) * xc).sum() - (q.sum(1) * xc).sum()) * 105.0), 1), "thirds_ours": [round(float(v), 1) for v in th(p)], "thirds_truth": [round(float(v), 1) for v in th(q)]}


def build(frames, views, Lv, Wv, Lm=105.0, Wm=68.0, seed=0, step=5, breaks=None, jit=None, every=2):
    """pipeline rows per sampled frame (oriented: each team attacks right), the truth points, fps"""
    rng = np.random.default_rng(seed); ks = sorted(frames)[::step]; fps = 25.0 / step
    meanx = {}
    for k in ks[::4]:
        p, ps, _ = frames[k]
        for i, x, y in ps: meanx.setdefault((p, i >= 200), {}).setdefault(i, []).append(x)
    gk, right = {}, {}
    for (p, away), d in meanx.items():
        m = {i: np.mean(v) for i, v in d.items() if len(v) > 10}; deep_left = np.mean(list(m.values())) < Lm / 2
        gk[(p, away)] = min(m, key=m.get) if deep_left else max(m, key=m.get); right[(p, away)] = bool(deep_left)
    allids = sorted({i for k in ks for i, _, _ in frames[k][1]})
    plan = {i: SV.detection_plan(len(ks), fps, rng, *(breaks or (None, None))) for i in allids}
    per = {}; truth = []; vi = 0; prev_p = None; vis = []
    cx_ = (np.arange(GRID[0]) + 0.5) / GRID[0] * Lm; cy_ = (np.arange(GRID[1]) + 0.5) / GRID[1] * Wm; CX, CY = np.meshgrid(cx_, cy_, indexing="ij")
    for j, k in enumerate(ks):
        p, ps, b = frames[k]; rows = []; tr = {}
        if j % every == 0: vi += 1
        if p != prev_p: per[len(per)] = []; truth.append(None); vis.append(None); prev_p = p
        vv = {}
        if b is not None:
            m, cx = views[vi % len(views)]; inside = SV.place(m, cx, b[0], Lv, Wv, Lm, Wm)
            for away, tm in ((False, "A"), (True, "B")):
                key = (p, away)
                if key not in gk: continue
                o = (lambda x: x) if right[key] else (lambda x: Lm - x)
                vv[tm] = inside(o(CX).ravel(), CY.ravel()).reshape(GRID)   # oriented cell centres in view
                team = [(i, x, y) for i, x, y in ps if (i >= 200) == away and i != gk[key]]
                tr[tm] = np.array([(o(x), y) for _, x, y in team])
                for i, x, y in team:
                    c = plan[i][j]
                    if c >= 0 and inside(np.array([x]), np.array([y]))[0]:
                        e = jit[rng.integers(0, len(jit))] if jit is not None else 0.0
                        rows.append([(p * 100 + i) * 1000 + c, tm, np.array([o(x), y]) + e, None, None, False])
        per[len(per)] = rows; truth.append(tr if b is not None else None); vis.append(vv or None)
    build.vis = vis
    return per, truth, fps


def heat_rules(per, truth, fps, Lm=105.0, Wm=68.0, holds=(5, 10, 20)):
    """-> {team: {rule: grid}} incl. 'truth'"""
    G = {tm: collections.defaultdict(lambda: np.zeros(GRID)) for tm in "AB"}
    tracks = collections.defaultdict(list)
    for k in range(len(per)):
        if truth[k]:
            for tm in "AB":
                if tm in truth[k]: cell_counts(truth[k][tm], Lm, Wm, G[tm]["truth"])
        for r in per[k]:
            cell_counts([r[2]], Lm, Wm, G[r[1]]["frames"]); tracks[r[0]].append((r[1], r[2]))
    for tid, v in tracks.items():
        if len(v) / fps < 2.0: continue
        g = cell_counts([q[1] for q in v], Lm, Wm); G[v[0][0]]["tracksum"] += g / g.sum()
    vis = getattr(build, "vis", None)
    if vis is not None and len(vis) == len(per):
        for tm in "AB":
            ex = np.zeros(GRID)
            for k in range(len(per)):
                if vis[k] and tm in vis[k] and any(r[1] == tm for r in per[k]): ex += vis[k][tm]
            G[tm]["exposure"] = np.where(ex > 0, G[tm]["frames"] / np.maximum(ex, 1), 0)
    for h in holds:
        held = AN.held_points(per, fps, h, L=Lm, W=Wm)
        for k in range(len(per)):
            for tm in "AB":
                if held[k][tm] is not None: cell_counts(held[k][tm], Lm, Wm, G[tm][f"hold{h}"])
    return G


def team_heat(per, fps, L, W, hold_s):
    """the shipped export rule (analytics.team_heatmaps) for the pipeline check"""
    return AN.team_heatmaps(per, fps, L, W, hold_s=hold_s)


def dry():
    """stand-ins: a team spread over the whole pitch, a view showing only the half around the ball -> the seen map leans
    to the ball; holding spots brings it back"""
    rng = np.random.default_rng(1); fr = {}
    for k in range(3000):
        bx = 30 + 45 * (0.5 + 0.5 * np.sin(k / 200))
        ps = [(100 + i, 10 + 8 * i + 6 * np.sin(k / 200) + rng.normal(0, .5), 6 + 6 * i) for i in range(11)] + [(200 + i, 95 - 8 * i, 6 + 6 * i) for i in range(11)]
        fr[k] = (1, ps, (bx, 34.0))
    m = np.zeros((106, 33), bool); m[:45] = True
    per, truth, fps = build(fr, [(m, 0.21)], 105.0, 64.0, step=5, breaks=(np.array([2.0, 4.0]), np.array([1.0, 3.0])))
    G = heat_rules(per, truth, fps)
    return {tm: {r: scores(g, G[tm]["truth"]) for r, g in G[tm].items() if r != "truth"} for tm in "AB"}


def main(a):
    if "--dry" in a: print(json.dumps(dry(), indent=1)); return
    import n0_22c_metrica as C
    out_p = a[a.index("--out") + 1] if "--out" in a else None; d = [x for x in a if not x.startswith("--") and x != out_p][0]
    runs, gaps, jit = C.our_breaks(CLIPS); res = {}; grids = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr = SV.load_metrica(d, g)
        for c in CLIPS:
            v, Lv, Wv = SV.view_masks(c)
            for seed in (0, 1):
                per, truth, fps = build(fr, v, Lv, Wv, breaks=(runs, gaps), jit=jit, seed=seed)
                G = heat_rules(per, truth, fps)
                if hasattr(AN, "team_heatmaps"):
                    th = team_heat(per, fps, 105.0, 68.0, None)
                    for tm in "AB": G[tm]["shipped"] = np.array(th[tm]["grid"]) if th[tm] else np.zeros(GRID)
                nm = f"game{g[-1]}_{c.split('/')[-2][:9]}_seed{seed}"
                res[nm] = {tm: {r: scores(gr, G[tm]["truth"]) for r, gr in G[tm].items() if r != "truth"} for tm in "AB"}
                if seed == 0: grids[nm] = {tm: {r: (gr / max(1e-9, gr.sum())).round(4).tolist() for r, gr in G[tm].items()} for tm in "AB"}
                print(nm, " | ".join(f"{r}: " + "/".join(f"{res[nm][tm][r]['tv']}" for tm in "AB") for r in res[nm]["A"]))
    rules = list(next(iter(res.values()))["A"])
    summ = {r: {q: round(float(np.mean([res[n][tm][r][q] for n in res for tm in "AB"])), 1) for q in ("tv", "thirds")} | {"x_mean": round(float(np.mean([res[n][tm][r]["x"] for n in res for tm in "AB"])), 1), "x_absmax": round(float(np.max([abs(res[n][tm][r]["x"]) for n in res for tm in "AB"])), 1)} for r in rules}
    print(json.dumps(summ, indent=1))
    if out_p:
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        json.dump({"summary": summ, "runs": res}, open(out_p, "w"), indent=1); json.dump(grids, open(out_p.replace(".json", "_grids.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
