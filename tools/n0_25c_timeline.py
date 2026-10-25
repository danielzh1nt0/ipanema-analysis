"""N0-25c: the app's 'shape over time' (stats.metrics.shape_timeline: one {t, length, width, line_height, n} per second
per team; read by the Phases ribbon / shape-over-time chart) checked on Metrica pro data for the first time.

Truth = Metrica's free pro tracking (all outfield players). Each Metrica frame is seen through a REAL camera view of our
app clips + our track breaks + our feet jitter (the same set-up as N0-22e, tools/n0_22e_shape_view.py), so the
pipeline rows look like ours. Then the SHIPPED timeline code is run on those rows and every second is compared with
the truth at that moment.

Rules compared (per team-second):
  old     - metrics.shape_timeline(analytics.shapes(per)) with IPANEMA default off: seen players only, >= 3 of them,
            one frame per second (what the app gets today)
  new     - metrics.shape_timeline(..., per=per, L, W): length / width from analytics.held_points (last seen spot kept
            5 s, like the Shape tab's medians since N0-22e), >= 5 points; line height from the seen players (N0-22e:
            holding makes the line worse)
Reported: median abs error per second (m), median ratio to the truth, share of seconds with a value, and how far the
timeline's median sits from the team card median (stats.teams[], which the Shape tab shows).
Usage: python tools/n0_25c_timeline.py /tmp/metrica/data [--out results/qa/n0-25c/metrica_timeline.json]
       python tools/n0_25c_timeline.py --dry"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import n0_22e_shape_view as SV
from ipanema import analytics as AN, metrics as ME

CLIPS = ["results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"]


def build(frames, views, Lv, Wv, Lm=105.0, Wm=68.0, seed=0, step=5, breaks=None, jit=None, every=2):
    """pipeline rows per sampled frame (as SV.pipeline_check) + the truth per frame. -> per, fps, attack_right per
    frame for A, truth {k: {tm: (length, width, line)}}"""
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
    per = {}; truth = {}; dirA = {}; vi = 0; prev_p = None
    for j, k in enumerate(ks):
        p, ps, b = frames[k]; rows = []
        if j % every == 0: vi += 1
        if p != prev_p:
            for _ in range(int(fps)): per[len(per)] = []              # a blank second between halves
            prev_p = p
        kk = len(per); tr = {}
        if b is not None:
            m, cx = views[vi % len(views)]; inside = SV.place(m, cx, b[0], Lv, Wv, Lm, Wm)
            for away, tm in ((False, "A"), (True, "B")):
                key = (p, away)
                if key not in gk: continue
                team = [(i, x, y) for i, x, y in ps if (i >= 200) == away]
                out = np.array([(x, y) for i, x, y in team if i != gk[key]])
                if len(out) >= 8: tr[tm] = SV.shape(out, right[key], Lm)
                for i, x, y in team:
                    c = plan[i][j]
                    if c >= 0 and inside(np.array([x]), np.array([y]))[0]:
                        e = jit[rng.integers(0, len(jit))] if jit is not None else 0.0
                        rows.append([i * 1000 + c, tm, np.array([x, y]) + e, None, None, i == gk[key]])
        per[kk] = rows; truth[kk] = tr; dirA[kk] = right.get((p, False), True)
    return per, fps, dirA, truth


def score(per, fps, dirA, truth, Lm=105.0, Wm=68.0):
    """run old and new timelines half by half (attack direction fixed within a half) -> per rule, per quantity:
    errors, ratios, coverage; plus each rule's timeline median vs the team-card median (analytics.stats rule)"""
    ks = sorted(per); halves = []; cur = []
    for k in ks:
        if not per[k] and not truth.get(k):
            if cur: halves.append(cur); cur = []
        else: cur.append(k)
    if cur: halves.append(cur)
    acc = {r: {q: {"err": [], "ratio": [], "have": 0, "n": 0, "vals": []} for q in ("length", "width", "line")} for r in ("old", "new")}
    card = {"length": [], "width": [], "line": []}
    for hk in halves:
        sub = {i: per[k] for i, k in enumerate(hk)}; ar_A = dirA[hk[0]]; ar = {"A": ar_A, "B": not ar_A}
        shapes_ = AN.shapes(sub, Lm, fps)
        tl = {"old": ME.shape_timeline(shapes_, fps, ar, Lm, per=None),
              "new": ME.shape_timeline(shapes_, fps, ar, Lm, per=sub, W=Wm)}
        # team-card medians the Shape tab shows (stats() rule: held length/width, seen line, >= 5)
        held = AN.held_points(sub, fps, L=Lm, W=Wm)
        for tm in ("A", "B"):
            for i in range(len(sub)):
                pts = np.array([r[2] for r in sub[i] if r[1] == tm and not r[5]])
                if len(pts) >= 5: card["line"].append(pts[:, 0].min() if ar[tm] else Lm - pts[:, 0].max())
                hp = held[i][tm]
                if hp is not None and len(hp) >= 5: card["length"].append(np.ptp(hp[:, 0])); card["width"].append(np.ptp(hp[:, 1]))
        for r in ("old", "new"):
            for tm in ("A", "B"):
                for e in tl[r][tm]:
                    i = int(round(e["t"] * fps)); k = hk[min(i, len(hk) - 1)]; t = truth.get(k, {}).get(tm)
                    if t is None: continue
                    for q, tv, key in zip(("length", "width", "line"), t, ("length", "width", "line_height")):
                        a = acc[r][q]; a["n"] += 1; v = e[key]
                        if v is None: continue
                        a["have"] += 1; a["err"].append(abs(v - tv)); a["ratio"].append(v / tv if tv > 1e-6 else np.nan); a["vals"].append(v)
    out = {}
    for r, d in acc.items():
        out[r] = {}
        for q, a in d.items():
            out[r][q] = {"err_m": round(float(np.median(a["err"])), 1) if a["err"] else None,
                         "ratio": round(float(np.nanmedian(a["ratio"])), 2) if a["ratio"] else None,
                         "covered": round(a["have"] / a["n"], 2) if a["n"] else None,
                         "median": round(float(np.median(a["vals"])), 1) if a["vals"] else None}
    out["card_median"] = {q: round(float(np.median(v)), 1) if v else None for q, v in card.items()}
    tv = {q: [] for q in ("length", "width", "line")}
    for k, d in truth.items():
        for tm, t in d.items():
            for q, v in zip(("length", "width", "line"), t): tv[q].append(v)
    out["truth_median"] = {q: round(float(np.median(v)), 1) if v else None for q, v in tv.items()}
    return out


def dry():
    """stand-ins: a team spread 20-80 m, a camera that shows the left half -> seen length short, held closer"""
    rng = np.random.default_rng(1); fr = {}
    for k in range(3000):
        bx = 30 + 45 * (0.5 + 0.5 * np.sin(k / 400))
        ps = [(100 + i, 20 + 6 * i + rng.normal(0, 1), 5 + 6 * i) for i in range(11)] + [(200 + i, 25 + 6 * i, 4 + 6 * i) for i in range(11)]
        fr[k] = (1, ps, (bx, 34.0))
    m = np.zeros((106, 33), bool); m[:40] = True
    per, fps, dirA, truth = build(fr, [(m, 0.19)], 105.0, 64.0)
    return score(per, fps, dirA, truth)


def main(a):
    if "--dry" in a: print(json.dumps(dry(), indent=1)); return
    import n0_22c_metrica as C
    out_p = a[a.index("--out") + 1] if "--out" in a else None
    d = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--out")][0]
    runs, gaps, jit = C.our_breaks(CLIPS); res = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr = SV.load_metrica(d, g)
        for c in CLIPS:
            v, Lv, Wv = SV.view_masks(c)
            for seed in (0, 1):
                per, fps, dirA, truth = build(fr, v, Lv, Wv, seed=seed, breaks=(runs, gaps), jit=jit)
                r = score(per, fps, dirA, truth); nm = f"game{g[-1]}_{c.split('/')[-2][:9]}_seed{seed}"; res[nm] = r
                print(nm, "truth", r["truth_median"], "card", r["card_median"])
                for rule in ("old", "new"): print("   ", rule, " ".join(f"{q}: err {r[rule][q]['err_m']} m x{r[rule][q]['ratio']} cov {r[rule][q]['covered']} med {r[rule][q]['median']}" for q in ("length", "width", "line")))
    if out_p: os.makedirs(os.path.dirname(out_p), exist_ok=True); json.dump(res, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
