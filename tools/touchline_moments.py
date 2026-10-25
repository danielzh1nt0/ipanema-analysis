"""TOUCHLINE: a fresh blind sample for the new touchline rule (tracking.touchline_rows, 6 s / 3.5 m / 1.0 m/s).
The app clips have only 9 stretches the rule drops and all 9 are people already graded in N0-25b, so the fresh sample
comes from the full-match exports in the repo (results/volume/runs/matches/<m>/frames_*.json, 10 rows/s, 'observed'
rows only, after the 9 Oct track rule): Vallentuna (a third ground), SFK-BP full (minus the clip's 1200-1500 s) and
AIK full (900-1200 s, 1500-1800 s). Per match random stretches the rule DROPS + some it KEEPS although they live by the
touchline (>= 6 s, never deeper than 3.5 m, moving faster than 1.0 m/s); shuffled. Writes moments files for
tools/touchline_frames.py and the key results/qa/touchline/moments_key.json (opened only after grading).
    PYTHONPATH=. python tools/touchline_moments.py"""
import sys, os, json, glob, random, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import tracking as TR
import touchline_lab as TL
PRM = {"min_s": 6.0, "max_m": 3.5, "max_v": 1.0}
MATCHES = {"val": ("p15u-vs-vallentuna-2026-10-03-6cce", None, 8, 4), "sfk": ("SFKBP1109", (1200, 1500), 6, 3),
           "aik": ("p15u-vs-aik-2026-09-21-bd09", None, 6, 3)}


def load_frames(m, skip=None):
    """frame files -> per {k: [[id, team, m, px]]} at 10 rows/s, k = round(t * 10); plus pitch L, W"""
    md = json.load(open(f"results/volume/runs/matches/{m}/match_data.json")); L, W = md["pitch"]["length"], md["pitch"]["width"]
    per = collections.defaultdict(list)
    for f in sorted(glob.glob(f"results/volume/runs/matches/{m}/frames_*.json")):
        for fr in json.load(open(f))["frames"]:
            if skip and skip[0] <= fr["t"] < skip[1]: continue
            k = int(round(fr["t"] * 10))
            for p in fr["players"]:
                if p.get("state", "observed") == "observed" and not p.get("gk"): per[k].append([p["id"], p["team"], np.array(p["m"], float), np.array(p["px"], float)])
    return dict(per), L, W, 10.0


def main(out="results/qa/touchline"):
    os.makedirs(out, exist_ok=True); rng = random.Random(1025); key = {}; pools = {}
    for s, (m, skip, nd, nk) in MATCHES.items():
        per, L, W, fps = load_frames(m, skip); old = TL.rule_9oct(per, L, W, fps)
        per = {k: [r for r in v if r[0] not in old] for k, v in per.items()}
        pos = {(k, r[0]): r for k, v in per.items() for r in v}
        st = [x for x in TR.touchline_stretches(per, W, fps, max_m=PRM["max_m"]) if x[2] >= PRM["min_s"]]
        drop = [x for x in st if x[4] <= PRM["max_v"]]; keep = [x for x in st if x[4] > PRM["max_v"]]
        rows = sum(len(v) for v in per.values()); dr = sum(len(x[1]) for x in drop)
        pools[s] = {"match": m, "rows_after_9oct": rows, "tracks_9oct": len(old), "stretches_dropped": len(drop), "rows_dropped": dr,
                    "rows_dropped_pct": round(100 * dr / max(1, rows), 2), "kept_near_line": len(keep)}
        pick = [(x, "dropped") for x in rng.sample(drop, min(nd, len(drop)))] + [(x, "kept") for x in rng.sample(keep, min(nk, len(keep)))]
        rng.shuffle(pick); moms = []
        for j, ((t, ks, dur, deep, v), kind) in enumerate(pick):
            sel = [int(ks[int(i)]) for i in np.linspace(0, len(ks) - 1, 4)]; mid = f"{s}{j:02d}"; rr = [pos[(k, t)] for k in sel]
            moms.append({"id": mid, "clip": m, "fps_src": "time", "t": round(sel[0] / fps, 2),
                         "tiles": [{"t": round(k / fps, 2), "ball_px": [float(r[3][0]), float(r[3][1])]} for k, r in zip(sel, rr)]})
            key[mid] = {"track": int(t), "kind": kind, "dur_s": round(float(dur), 1), "deep_m": round(deep, 1), "speed": round(v, 2),
                        "y_m": [round(float(r[2][1]), 1) for r in rr], "x_m": [round(float(r[2][0]), 1) for r in rr]}
        json.dump({"clip": m, "moments": moms}, open(f"{out}/moments_{s}.json", "w"), indent=1)
        print(s, pools[s], flush=True)
    json.dump(key, open(f"{out}/moments_key.json", "w"), indent=1); json.dump(pools, open(f"{out}/full_matches.json", "w"), indent=1)


if __name__ == "__main__":
    main()
