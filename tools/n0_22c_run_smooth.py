"""N0-22c: per-player distance / high-intensity / top speed in analytics.stats, old (one rolling mean across track gaps)
vs new (smoothed per seen run + gaps up to 2.5 s as a straight line when <= 7 m/s), on the exact picker inputs of the app clips (players after cleaning + gap filling).
Usage: python tools/n0_22c_run_smooth.py <picker_inputs.pkl> ... [--out results/qa/n0-22c/summary.json] [--dry]
--dry: a stand-in scene (no files needed), for the test suite."""
import sys, os, json, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN


def player_stats(per, fps, runs):
    n = len(per)
    per = {k: [[r[0], r[1], np.asarray(r[2], float), np.asarray(r[3], float), None, False] for r in per.get(k, [])] for k in range(n)}
    tracks = collections.defaultdict(dict)
    for k in range(n):
        for r in per[k]: tracks[r[0]][k] = r[2]
    frames_ = [{"carrier": None, "team": None, "pressure_m": None, "near_opps": 0} for _ in range(n)]
    keep = (AN.RUN_SMOOTH, AN.GAP_LINE_S, AN.GAP_MAX_MS); AN.RUN_SMOOTH, AN.GAP_LINE_S, AN.GAP_MAX_MS = (True, 2.5, 7.0) if runs else (False, 0.0, 7.0)
    try: st = AN.stats(per, frames_, [], [], tracks, np.full(n, 2), fps, 105.0, 68.0, {"A": True, "B": False})
    finally: AN.RUN_SMOOTH, AN.GAP_LINE_S, AN.GAP_MAX_MS = keep
    return {p["id"]: p for p in st["players"]}, tracks


def compare(per, fps, name):
    old, tracks = player_stats(per, fps, False); new, _ = player_stats(per, fps, True)
    gaps = {t: int((np.diff(sorted(v)) > 3).sum()) for t, v in tracks.items()}
    out = {"clip": name, "players": len(new), "with_gap": sum(1 for t in new if gaps[t]),
           "minutes": round(len(per) / fps / 60, 1)}
    for key in ("distance_m", "high_intensity_m"):
        for tm in ("A", "B"):
            out[f"{key}_{tm}_old"] = round(sum(p[key] for p in old.values() if p["team"] == tm)); out[f"{key}_{tm}_new"] = round(sum(p[key] for p in new.values() if p["team"] == tm))
    to = np.array([old[t]["top_speed_ms"] for t in new]); tn = np.array([new[t]["top_speed_ms"] for t in new])
    out["top_speed_median_old"], out["top_speed_median_new"] = round(float(np.median(to)), 2), round(float(np.median(tn)), 2)
    out["players_top_over_7_old"], out["players_top_over_7_new"] = int((to > 7).sum()), int((tn > 7).sum())
    ch = sorted(new, key=lambda t: -(old[t]["high_intensity_m"] - new[t]["high_intensity_m"]))[:12]
    out["biggest_changes"] = [{"id": int(t), "team": new[t]["team"], "gaps": gaps[t], "dist_old": old[t]["distance_m"], "dist_new": new[t]["distance_m"],
                               "hi_old": old[t]["high_intensity_m"], "hi_new": new[t]["high_intensity_m"], "top_old": old[t]["top_speed_ms"], "top_new": new[t]["top_speed_ms"]} for t in ch]
    return out


def stand_in(fps=30.0):
    n = int(8 * fps); per = {k: [] for k in range(n)}
    for k in range(n):
        if k < 90 or k >= 150: per[k].append([1, "A", [10.0 if k < 90 else 15.0, 30.0], [0, 0]])          # still, 2 s gap, still 5 m on
        per[k].append([2, "B", [20.0 + 4.0 * k / fps, 40.0], [0, 0]])                                     # runs 4 m/s, no gap
    return per, fps


if __name__ == "__main__":
    a = sys.argv[1:]; out_path = None
    if "--out" in a: i = a.index("--out"); out_path = a[i + 1]; del a[i:i + 2]
    res = []
    if "--dry" in a:
        per, fps = stand_in(); res.append(compare(per, fps, "stand-in"))
    for p in a:
        if p.startswith("--"): continue
        d = pickle.load(open(p, "rb")); res.append(compare(d["per"], d["fps"], os.path.basename(os.path.dirname(p))))
    for r in res: print(json.dumps({k: v for k, v in r.items() if k != "biggest_changes"}))
    if out_path: os.makedirs(os.path.dirname(out_path), exist_ok=True); json.dump(res, open(out_path, "w"), indent=1)
