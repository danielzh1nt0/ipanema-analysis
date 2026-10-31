"""N0-31e: final-third entries / field tilt on the repo's app match copies, old rule vs off-pitch balls skipped (N0-31e),
rebuilt from the exported frame files (raw possession + ball metres per kept frame; the copies are thinned, so 'old'
is a rebuild - its count is reported next to the stored one). Writes the entries only the old rule counts
(results/qa/n0-31e/dropped.json) for blind picture sheets on the free runner (tools/n0_31e_frames.py).

    python tools/n0_31e_clips.py [--out results/qa/n0-31e/clips.json]
"""
import sys, os, json, glob, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import metrics as M
from n0_30d_clips import load

IDX = {"A": 0, "B": 1, "loose": 2, "dead": 3}


def inputs(d, fr):
    ts = np.array([f["t"] for f in fr]); fps = 1.0 / max(float(np.median(np.diff(ts))), 1e-3)
    state = np.array([IDX.get(f.get("possession") or ("dead" if f.get("phase") == "dead" else "loose"), 2) for f in fr])
    ballm = {k: np.array(f["ball"]["m"], float) for k, f in enumerate(fr) if f.get("ball") and f["ball"].get("m") is not None}
    bspeed = {}
    ks = sorted(ballm)
    for a, b in zip(ks, ks[1:]):
        if b - a <= 3: bspeed[b] = float(np.linalg.norm(ballm[b] - ballm[a])) * fps / (b - a)
    return ts, fps, state, ballm, bspeed


def main(out):
    res, dropped = {}, []
    for p in sorted(glob.glob("results/volume/runs/matches/*/match_data.json")):
        mid = p.split("/")[-2]; d, fr = load(p)
        if not fr: continue
        st = json.load(open(p.replace("match_data.json", "stats.json"))) if os.path.exists(p.replace("match_data.json", "stats.json")) else {}
        L, W = d["pitch"]["length"], d["pitch"]["width"]; ar = d["attack_right"]
        ts, fps, state, ballm, bspeed = inputs(d, fr)
        kw = dict(hold_s=M.ENTRY_HOLD_S)
        old = M.field_tilt_and_entries(state, ballm, bspeed, fps, L, ar, *M.ENTRY_REARM, onpitch=False, **kw)
        new = M.field_tilt_and_entries(state, ballm, bspeed, fps, L, ar, *M.ENTRY_REARM, W=W, onpitch=True, **kw)
        stored = (st.get("metrics") or {}).get("field") or {}
        off = sum(1 for b in ballm.values() if not (-2 <= b[0] <= L + 2 and -2 <= b[1] <= W + 2))
        r = {"fps_kept": round(fps, 2), "ball_frames": len(ballm), "ball_off_pitch_share": round(off / max(1, len(ballm)), 3)}
        for tm in "AB":
            so = stored.get(tm) or {}
            ys_old = [e["y_m"] for e in old[tm]["final_third_entries"]]
            r[tm] = {"stored_entries": len(so.get("final_third_entries") or []), "stored_tilt": so.get("field_tilt_pct"),
                     "old_entries": old[tm]["entries_count"], "new_entries": new[tm]["entries_count"],
                     "old_tilt": old[tm]["field_tilt_pct"], "new_tilt": new[tm]["field_tilt_pct"],
                     "old_y_off_pitch": sum(1 for y in ys_old if not (-2 <= y <= W + 2)),
                     "new_y_off_pitch": sum(1 for e in new[tm]["final_third_entries"] if not (-2 <= e["y_m"] <= W + 2)),
                     "old_y_range": [min(ys_old), max(ys_old)] if ys_old else None}
            nt = {e["t"] for e in new[tm]["final_third_entries"]}
            for e in old[tm]["final_third_entries"]:
                if e["t"] in nt: continue
                k = int(round(e["t"] * fps)); f = fr[min(k, len(fr) - 1)]
                kk = min(range(len(ts)), key=lambda j: abs(ts[j] - e["t"])) if abs(ts[min(k, len(ts) - 1)] - e["t"]) > 0.05 else k
                f = fr[kk]
                dropped.append({"match": mid, "team": tm, "t": float(ts[kk]), "y_m": e["y_m"], "how": e["how"],
                                "ball_px": f["ball"]["px"] if f.get("ball") else None, "ball_m": f["ball"]["m"] if f.get("ball") else None,
                                "attacks": "right" if ar[tm] else "left"})
            r[tm]["only_old"] = sum(1 for x in dropped if x["match"] == mid and x["team"] == tm)
            r[tm]["only_new"] = len(nt - {e["t"] for e in old[tm]["final_third_entries"]})
        res[mid] = r; print(mid, json.dumps(r))
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1)
    # blind key: the sheets show the dropped entries mixed with an equal number of kept ones (seeded), rule not on the sheet
    rnd = random.Random(31); kept = []
    for p in sorted(glob.glob("results/volume/runs/matches/*/match_data.json")):
        mid = p.split("/")[-2]
        if mid not in res: continue
        d, fr = load(p); L, W = d["pitch"]["length"], d["pitch"]["width"]
        ts, fps, state, ballm, bspeed = inputs(d, fr)
        new = M.field_tilt_and_entries(state, ballm, bspeed, fps, L, d["attack_right"], *M.ENTRY_REARM, W=W, onpitch=True, hold_s=M.ENTRY_HOLD_S)
        n_drop = sum(1 for x in dropped if x["match"] == mid)
        pool = [(tm, e) for tm in "AB" for e in new[tm]["final_third_entries"]]
        for tm, e in rnd.sample(pool, min(len(pool), n_drop)):
            k = min(range(len(ts)), key=lambda j: abs(ts[j] - e["t"])); f = fr[k]
            kept.append({"match": mid, "team": tm, "t": float(ts[k]), "y_m": e["y_m"], "how": e["how"], "ball_px": f["ball"]["px"] if f.get("ball") else None,
                         "ball_m": f["ball"]["m"] if f.get("ball") else None, "attacks": "right" if d["attack_right"][tm] else "left"})
    allm = [dict(x, rule="dropped") for x in dropped] + [dict(x, rule="kept") for x in kept]
    rnd.shuffle(allm)
    for i, x in enumerate(allm): x["id"] = f"e{i:02d}"
    od = os.path.dirname(out)
    json.dump({x["id"]: {k: x[k] for k in ("match", "team", "t", "y_m", "rule")} for x in allm}, open(f"{od}/moments_key.json", "w"), indent=1)
    json.dump([{k: v for k, v in x.items() if k not in ("rule", "y_m", "ball_m")} for x in allm], open(f"{od}/moments.json", "w"), indent=1)
    print(len(dropped), "dropped,", len(kept), "kept controls")
    return res


if __name__ == "__main__":
    a = sys.argv[1:]; main(a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-31e/clips.json")
