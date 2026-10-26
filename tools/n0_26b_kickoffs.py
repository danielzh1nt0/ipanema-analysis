"""N0-26b (26 Oct, local, free): kick-off line-ups as a direction check (P.direction_from_kickoffs) on our exports, against
the by-eye direction in periods/<match>.json (attack_right_A; the dark team A, first period; later periods are mirrored in
a full match's frames, so one value covers the match) and the run's own attack_right.
    PYTHONPATH=. python tools/n0_26b_kickoffs.py [--out results/qa/n0-26b/exports.json]"""
import sys, os, json, gzip, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P

RUNS = ["results/kaggle/full_match_52d795-join", "results/kaggle/full_match_vjoin7", "results/kaggle/full_match_sjoin25"]


def load_export(d):
    """frames_lite -> (per {i: rows}, ballm {i: xy}, fps, ts, match, summary)"""
    s = json.load(open(f"{d}/summary.json")); fl = json.load(gzip.open(f"{d}/frames_lite.json.gz"))
    st = json.load(open(f"{d}/stats.json")); L, W = st["pitch"]["length"], st["pitch"]["width"]
    per, ballm, ts = {}, {}, []
    for i, f in enumerate(fl["frames"]):
        ts.append(f["t"])
        per[i] = [[p["id"], p["team"], np.asarray(p["m"], float), p.get("px"), None, bool(p.get("gk"))] for p in f["players"] if p.get("m")]
        b = f.get("ball")
        if b and b.get("m") is not None: ballm[i] = np.asarray(b["m"], float)
    return per, ballm, 1 / fl["stride_s"], ts, s["match_id"], s, L, W


def periods_file(match):
    m = match.split("_s")[0] if "_s" in match and not match.startswith("p15u") else match
    p = f"periods/{m}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-26b/exports.json"
    rep = {}
    for d in RUNS:
        per, ballm, fps, ts, match, s, L, W = load_export(d)
        sp = P.kickoff_spells(per, ballm, fps, L, W)
        ko = P.direction_from_kickoffs(per, ballm, fps, L, W, log=lambda *x: None)
        pf = periods_file(match)
        rep[os.path.basename(d)] = {
            "match": match, "pitch": [L, W], "run_attack_right_A": s["attack_right"]["A"],
            "periods_file_attack_right_A": pf.get("attack_right_A"),
            "kickoff_attack_right_A": None if ko is None else ko["A"],
            "spells": [{"t0": round(ts[x["k0"]], 1), "t1": round(ts[x["k1"]], 1), "dark_A_in": "left half" if x["a_left"] else "right half"} for x in sp]}
        r = rep[os.path.basename(d)]
        print(match, "run A->", r["run_attack_right_A"], "| file A->", r["periods_file_attack_right_A"], "| kick-offs A->", r["kickoff_attack_right_A"], len(sp), "spells", flush=True)
        for x in r["spells"]: print("   ", x)
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
