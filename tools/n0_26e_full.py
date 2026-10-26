"""N0-26e (26 Oct, local, free): our shot detector on the 3 full matches in the repo (stats.json metrics.shots_detected,
written by the run with the rule of that day = old rule; the goal fix only changes outcomes) vs Veo's shot list in the
same file (metrics.shots, t = start of Veo's highlight clip), scored the old way (+-10 s) and with the clip window
(2 s before to 30 s after the clip start, veo.score_detector since N0-26e). Veo's list is 'shots on goal' only, so a
wide shot of ours counts as not real here: 'real' is a lower bound. -> results/qa/n0-26e/full_matches.json
    PYTHONPATH=. python tools/n0_26e_full.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import veo as VEO

MATCHES = ["SFKBP1109", "p15u-vs-aik-2026-09-21-bd09", "p15u-vs-vallentuna-2026-10-03-6cce"]


def old_score(d, v, tol=10.0):
    return {"found": sum(1 for a in v if any(abs(b - a) <= tol for b in d)), "real": sum(1 for b in d if any(abs(b - a) <= tol for a in v))}


def main():
    res = {}
    for m in MATCHES:
        mx = json.load(open(f"results/volume/runs/matches/{m}/stats.json"))["metrics"]
        d, v = mx.get("shots_detected", []), mx["shots"]
        res[m] = {"veo_shots": len(v), "ours": len(d), "old_window_10s": old_score([x["t"] for x in d], [x["t"] for x in v]),
                  "clip_window": VEO.score_detector(d, v), "ours_t": [x["t"] for x in d], "veo_t": [x["t"] for x in v]}
        print(m, json.dumps({k: res[m][k] for k in ("veo_shots", "ours", "old_window_10s", "clip_window")}))
    tot = {k: sum(r["clip_window"][k] for r in res.values()) for k in ("veo_shots", "found", "ours", "real")}
    res["total"] = tot; print("total", tot)
    os.makedirs("results/qa/n0-26e", exist_ok=True); json.dump(res, open("results/qa/n0-26e/full_matches.json", "w"), indent=1)


if __name__ == "__main__":
    main()
