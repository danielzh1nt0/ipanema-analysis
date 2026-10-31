"""N0-31e: final-third entries / field tilt with off-pitch balls skipped (metrics.field_tilt_and_entries W=, onpitch=)
vs the old rule, on Metrica's pro data (truth as in N0-24c: Metrica events + clean ball) and our ball-noise models.

    PYTHONPATH=. python tools/n0_31e_entries.py <metrica sample-data/data dir> [--out results/qa/n0-31e/metrica_entries.json]
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, metrics as M
from n0_24b_hightov import truth_direction
from n0_24c_entries import truth_owner, truth_entries, match, L, FPS

W = 68.0


def ours_state(fr, per, noisy, seed):
    import metricalab as ML
    got = {}; o_ps, o_car = P.pipeline_state, P.carriers
    def ps_(*a, **k): r = o_ps(*a, **k); got["state"] = r[0]; got["bspeed"] = r[1]; return r
    def car_(*a, **k): r = o_car(*a, **k); got["ballm"] = r[1]; return r
    P.pipeline_state, P.carriers = ps_, car_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.pipeline_state, P.carriers = o_ps, o_car
    return got, o["attack_right"]


def truth_tilt(own, fr, ar):
    out = {}
    for tm in ("A", "B"):
        ks = [k for k in own if own[k] == tm and fr.get(k, (0, 0, None))[2] is not None]
        fin = sum(1 for k in ks if (fr[k][2][0] if ar[tm] else L - fr[k][2][0]) >= 2 * L / 3)
        out[tm] = round(100 * fin / len(ks)) if ks else None
    return out


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        ev = list(csv.DictReader(open(f"{gd}/{g}_RawEventsData.csv")))
        for per in halves:
            k0 = min(k for k in fr if fr[k][0] == per); ar_t = truth_direction(fr, per)
            own, passes = truth_owner(ev, fr, per); te = truth_entries(own, passes, fr, ar_t); tt = truth_tilt(own, fr, ar_t)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    got, ar = ours_state(fr, per, noisy, seed)
                    bm = got["ballm"]; off = sum(1 for b in bm.values() if b is not None and not (-2 <= b[0] <= L + 2 and -2 <= b[1] <= W + 2))
                    for rule, kw in (("old", {"onpitch": False}), ("new", {"W": W, "onpitch": True})):
                        f = M.field_tilt_and_entries(got["state"], bm, got["bspeed"], FPS, L, ar, *M.ENTRY_REARM, hold_s=M.ENTRY_HOLD_S, **kw)
                        ours = {tm: f[tm]["final_third_entries"] for tm in ("A", "B")}
                        ys = [e["y_m"] for tm in ours for e in ours[tm]]
                        row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "rule": rule, "direction_same": ar == ar_t,
                               "ball_frames": len(bm), "ball_off_pitch_frames": off,
                               "tilt": {tm: [tt[tm], f[tm]["field_tilt_pct"]] for tm in ("A", "B")},
                               "entries_off_pitch": sum(1 for y in ys if not (-2 <= y <= W + 2)),
                               "matched": match(te, ours, k0)}
                        rows.append(row); print(json.dumps(row), flush=True)
    return rows


def summary(rows):
    out = {}
    for noise in ("False", "streak", "flicker"):
        for rule in ("old", "new"):
            R = [r for r in rows if r["noise"] == noise and r["rule"] == rule]
            if not R: continue
            t = sum(r["matched"]["truth"] for r in R); o = sum(r["matched"]["ours"] for r in R); fd = sum(r["matched"]["found"] for r in R); rl = sum(r["matched"]["ours_real"] for r in R)
            te = [abs(r["tilt"][tm][0] - r["tilt"][tm][1]) for r in R for tm in ("A", "B") if r["tilt"][tm][0] is not None and r["tilt"][tm][1] is not None]
            out[f"{noise}/{rule}"] = {"found_pct": round(100 * fd / max(1, t)),
                                      "ours_real_pct": round(100 * rl / max(1, o)), "count_vs_truth": round(o / max(1, t), 2),
                                      "entries_off_pitch": sum(r["entries_off_pitch"] for r in R), "tilt_off_pts_median": sorted(te)[len(te) // 2] if te else None,
                                      "ball_off_pitch_share": round(sum(r["ball_off_pitch_frames"] for r in R) / max(1, sum(r["ball_frames"] for r in R)), 4)}
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-31e/metrica_entries.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); s = summary(rows); print(json.dumps(s, indent=1))
    json.dump({"summary": s, "rows": rows}, open(out, "w"), indent=1)
