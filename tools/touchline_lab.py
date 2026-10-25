"""TOUCHLINE (25 Oct): drop people just behind the far touchline (spectators, bench staff, a bag) that the tracker
keeps as players, without losing real wingers / throw-ins. Rule under test: tracking.touchline_rows (per stretch of a
track: lasts >= min_s, never deeper than max_m from a side touchline, median smoothed speed <= max_v).

Graded on
  1. the 36 people graded blind in N0-25b (results/qa/n0-25b/grades.json; 4 tiles each, P / N / ?) on the app clips'
     picker inputs, after the 9 Oct track rule that the current clean() already has (these exports predate it);
  2. Metrica pro tracking (every row is a real player): rows dropped, as the truth and cut like our tracks
     (n0_22c_metrica.degrade: our seen runs / gaps / jitter);
  3. rows dropped on our two clips.
    PYTHONPATH=. python tools/touchline_lab.py /tmp/metrica/data [--out results/qa/touchline/lab.json]
    PYTHONPATH=. python tools/touchline_lab.py --dry"""
import sys, os, json, pickle, itertools, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import tracking as TR
CL = {"sfk": "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl", "aik": "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"}
CACHE = {}
GRID = {"min_s": [2.0, 3.0, 4.0, 6.0], "max_m": [2.5, 3.5, 4.5], "max_v": [0.6, 0.8, 1.0, 1.3]}


def rule_9oct(per, L, W, fps, line_m=1.0):
    """the track rule clean() has had since 9 Oct (>= 1 s, >= 70% of rows within 1 m of a line)"""
    life = collections.defaultdict(list)
    for k in per:
        for r in per[k]: life[r[0]].append(r[2])
    return {t for t, pts in life.items() if len(pts) >= int(fps) and sum(1 for p in pts if min(p[0], L - p[0], p[1], W - p[1]) < line_m) >= 0.7 * len(pts)}


def load_clips():
    out = {}
    for s, p in CL.items():
        d = pickle.load(open(p, "rb")); per = d["per"]; old = rule_9oct(per, d["L"], d["W"], d["fps"])
        per2 = {k: [r for r in per[k] if r[0] not in old] for k in per}
        out[s] = {"per": per2, "W": d["W"], "fps": d["fps"], "old_tracks": old, "rows": sum(len(v) for v in per.values()),
                  "rows_after_9oct": sum(len(v) for v in per2.values())}
    return out


def tiles():
    key = json.load(open("results/qa/n0-25b/moments_key.json")); gr = json.load(open("results/qa/n0-25b/grades.json")); out = []
    for s in CL:
        for m in json.load(open(f"results/qa/n0-25b/moments_{s}.json"))["moments"]:
            v = key[m["id"]]
            for tl, g in zip(m["tiles"], gr[m["id"]][:4]):
                if g in "PN": out.append({"clip": s, "id": m["id"], "frame": tl["frame"], "track": v["track"], "label": g})
    return out


def grade(clips, T, prm):
    res = {"N_dropped": 0, "N": 0, "P_dropped": 0, "P": 0, "rows_dropped": {}}; drops = {}
    for s, c in clips.items():
        st = c.setdefault("st", TR.touchline_stretches(c["per"], c["W"], c["fps"], max_m=max(GRID["max_m"])))
        dr = TR.touchline_rows(c["per"], c["W"], c["fps"], stretches=st, **prm); drops[s] = dr
        res["rows_dropped"][s] = round(100 * len(dr) / max(1, c["rows_after_9oct"]), 2)
    for t in T:
        gone = t["track"] in clips[t["clip"]]["old_tracks"] or (t["frame"], t["track"]) in drops[t["clip"]]
        res[t["label"]] += 1; res[t["label"] + "_dropped"] += int(gone)
    return res, drops


def metrica_per(players, ks_q=None):
    per = collections.defaultdict(list)
    for pid, P in players.items():
        ks, Q = ks_q[pid] if ks_q else (np.arange(len(P)), P)
        for k, q in zip(ks, Q): per[int(k)].append([pid, "A" if pid < 100 else "B", np.asarray(q, float)])
    return dict(per)


def metrica_sets(data_dir, minutes=45.0, seeds=2):
    import n0_22c_metrica as C, metricalab as ML
    runs, gaps, jit = C.our_breaks(list(CL.values())); sets = {}
    for game in (1, 2):
        players, fps = C.metrica_players(data_dir, minutes, game)
        sets[f"g{game}_truth"] = (metrica_per(players), fps)
        print("metrica game", game, "players", len(players), flush=True)
        for sd in range(seeds):
            rng = np.random.default_rng(sd); kq = {pid: C.degrade(P, fps, runs, gaps, jit, rng) for pid, P in players.items()}
            sets[f"g{game}_ours_s{sd}"] = (metrica_per(players, kq), fps)
    return sets, ML.W


def metrica_grade(sets, W, prm):
    out = {}
    for name, (per, fps) in sets.items():
        if name not in CACHE:
            CACHE[name] = (sum(len(v) for v in per.values()), sum(1 for v in per.values() for r in v if min(r[2][1], W - r[2][1]) < 3.5),
                           TR.touchline_stretches(per, W, fps, max_m=max(GRID["max_m"])))
        n, edge, st = CACHE[name]
        dr = TR.touchline_rows(per, W, fps, stretches=st, **prm)
        out[name] = {"rows_dropped_pct": round(100 * len(dr) / max(1, n), 3), "edge_rows_dropped_pct": round(100 * len(dr) / max(1, edge), 2)}
    return out


def main(a):
    out_path = a[a.index("--out") + 1] if "--out" in a else None
    if "--dry" in a:
        fps, W = 25.0, 68.0; t = np.arange(int(60 * fps))
        per = {k: [[1, "A", np.array([50.0 + 0.01 * (k % 7), 67.5])], [2, "A", np.array([20 + 0.15 * k, 30.0])]] for k in t}
        dr = TR.touchline_rows(per, W, fps); print("dry: dropped", len(dr), "spectator rows", sum(1 for _, i in dr if i == 1)); return 0
    clips = load_clips(); T = tiles(); sets, MW = metrica_sets(a[0]); rows = []
    for v in itertools.product(*GRID.values()):
        prm = dict(zip(GRID, v)); g, _ = grade(clips, T, prm); m = metrica_grade(sets, MW, prm)
        rows.append({"prm": prm, **g, "metrica": m, "metrica_worst_pct": max(x["rows_dropped_pct"] for x in m.values()),
                     "metrica_worst_edge_pct": max(x["edge_rows_dropped_pct"] for x in m.values())})
        print(prm, f"N {g['N_dropped']}/{g['N']}  P lost {g['P_dropped']}/{g['P']}  clips {g['rows_dropped']}  metrica worst {rows[-1]['metrica_worst_pct']}% (edge {rows[-1]['metrica_worst_edge_pct']}%)", flush=True)
    base = {"N_dropped_9oct_only": sum(1 for t in T if t["label"] == "N" and t["track"] in clips[t["clip"]]["old_tracks"]),
            "P_dropped_9oct_only": sum(1 for t in T if t["label"] == "P" and t["track"] in clips[t["clip"]]["old_tracks"]),
            "clips": {s: {"rows": c["rows"], "rows_after_9oct": c["rows_after_9oct"], "tracks_9oct": len(c["old_tracks"])} for s, c in clips.items()}}
    print("9 Oct rule alone:", base)
    if out_path: os.makedirs(os.path.dirname(out_path), exist_ok=True); json.dump({"baseline_9oct": base, "grid": rows}, open(out_path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
