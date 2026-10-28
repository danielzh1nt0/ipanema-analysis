"""N0-28f: what IPANEMA_PLAYER_HOLD_S would do to the app clips' per-track average positions (exact inputs,
results/volume/cache/<clip>/picker_inputs.pkl). Per track (>= 2 s seen): shift of the average spot and of the
attacking-third share, hold 0 vs 10 s. Writes results/qa/n0-28f/clips.json.
    PYTHONPATH=. python tools/n0_28f_clips.py"""
import sys, os, json, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN

CLIPS = {"SFKBP1109_s1200": "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl",
         "aik_s2520": "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"}


def main(hold=10.0):
    out = {}
    for nm, p in CLIPS.items():
        d = pickle.load(open(p, "rb")); per, fps, L, W = d["per"], d["fps"], float(d["L"]), float(d["W"])
        seen = AN.held_track_points(per, fps, 0.0, L=L, W=W); held = AN.held_track_points(per, fps, hold, L=L, W=W)
        sh, n_s, n_h = [], [], []
        for t, P in seen.items():
            if len(P) / fps < 2.0: continue
            H = held[t]; sh.append(float(np.linalg.norm(H.mean(0) - P.mean(0)))); n_s.append(len(P)); n_h.append(len(H))
        out[nm] = {"tracks_2s": len(sh), "tracks_all": len(seen), "seen_s_median": round(float(np.median(n_s)) / fps, 1),
                   "held_extra_s_median": round(float(np.median(np.array(n_h) - np.array(n_s))) / fps, 1),
                   "avg_spot_shift_median_m": round(float(np.median(sh)), 1), "avg_spot_shift_p90_m": round(float(np.percentile(sh, 90)), 1)}
        print(nm, out[nm], flush=True)
    os.makedirs("results/qa/n0-28f", exist_ok=True); json.dump(out, open("results/qa/n0-28f/clips.json", "w"), indent=1)


if __name__ == "__main__":
    main()
