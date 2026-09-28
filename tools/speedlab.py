import sys, os, numpy as np, collections, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.metricalab as ML, tools.turnoverlab as TL
from ipanema import possession as P, analytics as AN
out = []
for g in ("Sample_Game_1", "Sample_Game_2"):
    d = f"{sys.argv[1]}/{g}"
    home = ML.load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
    frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
    ev = ML.load_events(f"{d}/{g}_RawEventsData.csv"); byper = {}
    for k, (p, _, _) in frames.items(): byper.setdefault(p, []).append(k)
    T = ML.truth(ev, byper)
    for per in (1, 2):
        tl = [(e["f0"], "A" if e["team"] == "Home" else "B") for e in ev if e["period"] == per and e["type"] == "BALL LOST"]
        for noisy in (False, "streak"):
            for sw in (None, 0.4, 0.6):
                S = TL.prepare(frames, per, noisy, speed_win_s=sw); st = np.asarray(S["state"])
                live = np.isin(st, [0, 1]).sum(); poss = round(100 * (st == 0).sum() / max(1, live))
                row = {"g": g[-1], "half": per, "noise": noisy or "clean", "win": sw, "loose%": round(100 * (st == 2).mean()), "poss_home": poss, "poss_truth": T[per]["possession_pct"]["Home"]}
                for bef, aft in ((3.0, 3.0), (1.0, 1.0), (0.5, 0.5)):
                    tvs = P.turnovers(S["perd"], S["frames_"], S["state"], S["bm"], ML.FPS, S["ar"], min_before_s=bef, min_after_s=aft)
                    ol = [(S["k0"] + t["frame_lost"], t["lost_by"]) for t in tvs]; hit = TL.match(tl, ol)
                    ps, _ = AN.passes(S["perd"], S["frames_"], tvs, {}, S["ar"], ML.FPS, min_touch_s=0.6, floor_s=0.08, sandwich=True)
                    tp = sum(T[per]["passes"].values()); op = sum(p["completed"] for p in ps)
                    row[f"tv{bef}"] = f"{len(ol)}/{len(tl)} found {hit}"; row[f"pass{bef}"] = f"{round(100*(op-tp)/tp)}%"
                out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open("results/metrica/speedlab.json", "w"), indent=1)
