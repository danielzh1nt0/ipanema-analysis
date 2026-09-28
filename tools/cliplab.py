"""28 Sep: re-run the stats on the SFK-BP clip locally from its saved positions (match_data.json) - no cloud, no cost.
Compares old vs new pass rule, ball-based vs movement-based stoppages, and scores stoppages against Veo's list."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eventlab as EL
from ipanema import possession as P, analytics as AN

def run(path="results/volume/runs/matches/SFKBP1109_s1200/match_data.json", t0=1200, t1=1500, fill=False):
    d, per, ballm, conf, seen, fps, L, W = EL.load(path)
    if fill:
        from ipanema import tracking as TR
        Hs = {k: (np.array(f["pitch_lines"]).reshape(3, 3) if f.get("pitch_lines") else None) for k, f in enumerate(d["frames"])}
        per, _ = TR.fill_gaps(per, fps, 1.0, H=Hs)
    H = {k: np.eye(3) for k in per}
    frames_, bm = P.carriers(per, {k: v for k, v in ballm.items()}, H)
    state, bspeed = P.viterbi(per, P.clean_ball(bm, L, W), fps, L, W)
    ar = d.get("attack_right") or {"A": True, "B": False}
    tvs = P.turnovers(per, frames_, state, bm, fps, ar)
    old, _ = AN.passes(per, frames_, tvs, {}, ar, fps, min_touch_s=0.0, floor_s=0.0, sandwich=False)
    new, _ = AN.passes(per, frames_, tvs, {}, ar, fps)
    rst = P.restarts(state, bm, fps, L, W); mot = P.stoppages_from_motion(per, fps, 1.4, 4.0)
    truth = EL.veo_truth("reference/veo_events_SFKBP1109.txt", t0, t1)
    st = np.asarray(state); n = len(st); dm = np.zeros(n, bool)
    for a, b in mot: dm[a:b + 1] = True
    def veo_hits(times):
        return sum(any(t["t0"] - 5 <= x < t["t1"] + 5 for x in times) for t in truth)
    rep = {"frames": n, "players_per_frame_median": float(np.median([len(per[k]) for k in per])),
           "ball_known_pct": round(100 * len(bm) / n),
           "passes_old": {t: sum(p["completed"] for p in old if p["team"] == t) for t in "AB"},
           "passes_new": {t: sum(p["completed"] for p in new if p["team"] == t) for t in "AB"},
           "possession_pct": {t: round(100 * (st == i).sum() / max(1, np.isin(st, [0, 1]).sum())) for i, t in enumerate("AB")},
           "veo_restarts": len(truth), "veo_by_minute": [(t["minute"], t["kind"]) for t in truth],
           "ball_rule": {"stoppages": len(rst), "dead_pct": round(100 * (st == 3).mean()), "veo_minutes_covered": veo_hits([r["t"] for r in rst])},
           "movement_rule": {"stoppages": len(mot), "dead_pct": round(100 * dm.mean()), "veo_minutes_covered": veo_hits([a / fps for a, b in mot]),
                             "intervals_s": [(round(a / fps, 1), round(b / fps, 1)) for a, b in mot]}}
    return rep

if __name__ == "__main__":
    fill = "--fill" in sys.argv; r = run(fill=fill); print(json.dumps({k: v for k, v in r.items() if k != "veo_by_minute"}, indent=1))
    os.makedirs("results/metrica", exist_ok=True); json.dump(r, open(f"results/metrica/clip_offline{'_filled' if fill else ''}.json", "w"), indent=1)
