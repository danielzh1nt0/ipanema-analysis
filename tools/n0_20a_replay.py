"""N0-20a (20 Oct, local): what the app's AIK clip would show with its Veo shots/goals, from the run's exact picker inputs
(results/volume/cache/<clip>/picker_inputs.pkl): ball picked like the app (pick_v2 + bridge), Veo events via run.veo_source.
    PYTHONPATH=. python tools/n0_20a_replay.py"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, veo as VEO
from ipanema.run import veo_source, periods_direction, veo_in_window
from ipanema.config import Settings
CLIP = "p15u-vs-aik-2026-09-21-bd09_s2520"; OUT = "results/qa/n0-20a"
d = pickle.load(open(f"results/volume/cache/{CLIP}/picker_inputs.pkl", "rb")); S = Settings()
per = {k: [[r[0], r[1], np.asarray(r[2], float), np.asarray(r[3], float)] for r in v] for k, v in d["per"].items()}
fps, H, L, W, cands = d["fps"], d["H"], d["L"], d["W"], d["cands"]
ball = BL.bridge(BL.pick_v2(cands, H, L, W, per=per, fps=fps, log=lambda *a: None), fps)
frames_, ballm = P.carriers(per, ball, H, S.carrier_r, S.near_r)
ar = periods_direction("/nonexistent", CLIP); attack_right = {"A": bool(ar), "B": not ar}
log = []; src = veo_source("/nonexistent", CLIP, len(per) / fps, L, W, log=log.append)
vs, rej = VEO.build(src[0], fps, ballm, per, L, W, attack_right, [], log=log.append, known_teams=src[1], known_origins=src[2])
vs = [x for x in vs if veo_in_window(x["t"], len(per) / fps, None)]
res = {"clip": CLIP, "fps": fps, "frames": len(per), "L": L, "W": W, "attack_right": attack_right, "log": log,
       "before": {"shots": {"A": 0, "B": 0}, "goals": {"A": 0, "B": 0}, "note": "app summary (results/modal/<clip>.txt): no Veo file found for the clip id"},
       "after": {"shots": {t: sum(1 for s in vs if s["team"] == t) for t in ("A", "B")}, "goals": {t: sum(1 for s in vs if s["team"] == t and s["goal"]) for t in ("A", "B")},
                 "unknown_team": sum(1 for s in vs if s["team"] is None), "events": vs, "rejected": rej}}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/replay_aik_clip.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("attack_right", "log")}, indent=1)); print(json.dumps(res["after"], indent=1, default=float))
