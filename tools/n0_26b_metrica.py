"""N0-26b (26 Oct, local, free): is 'the half a team lines up in at kick-offs' a safe direction rule? Checked on Metrica pro
data (Sample Games 1 and 2, 4 halves). Truth per half: the end each team's shots go to (event end x). Ours:
P.direction_from_kickoffs on (a) perfect positions, (b) what our Veo follow-cam gives: only players within 30 m (x) of the
ball's spot in the picture, ~65% of those found, 0.8 m feet jitter, ball missing 25% / a wrong pick 10% of frames, 3 seeds.
Also: every spell is matched to Metrica's KICK OFF events (within 10 s) - how many real kick-offs are found, how many
spells are elsewhere, and whether those vote the right way too.
    PYTHONPATH=. python tools/n0_26b_metrica.py <metrica sample-data/data dir> [--out results/qa/n0-26b/metrica.json]"""
import sys, os, json, csv, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P
import metricalab as ML

L, W, FPS = ML.L, ML.W, ML.FPS


def truth_dir(ev_rows, per):
    """Home attacks right in this half? from the mean end x of each team's shots (Metrica x 0..1)"""
    xs = {"Home": [], "Away": []}
    for r in ev_rows:
        if int(r["Period"]) == per and r["Type"] == "SHOT" and r["End X"] not in ("NaN", ""): xs[r["Team"]].append(float(r["End X"]))
    h = np.mean(xs["Home"]) if xs["Home"] else None; a = np.mean(xs["Away"]) if xs["Away"] else None
    if h is not None and a is not None: assert (h > 0.5) != (a > 0.5), (per, h, a)
    return bool(h > 0.5) if h is not None else bool(a < 0.5)


def kickoffs(ev_rows, per):
    return [int(r["Start Frame"]) for r in ev_rows if int(r["Period"]) == per and r["Type"] == "SET PIECE" and r["Subtype"].startswith("KICK OFF")]


def build(frames, per, noisy, seed):
    rng = random.Random(seed); ks = sorted(k for k in frames if frames[k][0] == per)
    P_, B_ = {}, {}
    for k in ks:
        _, ps, b = frames[k]
        rows = [[pid, "A" if pid < 200 else "B", np.array([x, y]), None, None, False] for pid, x, y in ps]
        bm = None if b is None else np.array(b, float)
        if noisy:
            cx = bm[0] if bm is not None else L / 2
            rows = [r for r in rows if abs(r[2][0] - cx) < 30 and rng.random() < 0.65]
            for r in rows: r[2] = r[2] + np.array([rng.gauss(0, 0.8), rng.gauss(0, 0.8)])
            if bm is not None:
                u = rng.random()
                bm = None if u < 0.25 else bm + (np.array([rng.uniform(-25, 25), rng.uniform(-15, 15)]) if u < 0.35 else np.array([rng.gauss(0, 0.7), rng.gauss(0, 0.7)]))
        P_[k] = rows
        if bm is not None: B_[k] = bm
    return P_, B_


def main():
    a = sys.argv[1:]; d = a[0]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-26b/metrica.json"
    rep = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        base = f"{d}/{g}/{g}"
        home = ML.load_tracking(f"{base}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{base}_RawTrackingData_Away_Team.csv", 200)
        frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        ev = list(csv.DictReader(open(f"{base}_RawEventsData.csv")))
        for per in (1, 2):
            tr = truth_dir(ev, per); kos = kickoffs(ev, per)
            for noisy, seeds in ((False, [0]), (True, [0, 1, 2])):
                for seed in seeds:
                    pp, bb = build(frames, per, noisy, seed)
                    sp = P.kickoff_spells(pp, bb, FPS, L, W)
                    res = P.direction_from_kickoffs(pp, bb, FPS, L, W, log=lambda *x: None)
                    at_ko = [s for s in sp if any(s["k0"] - 10 * FPS <= f <= s["k1"] + 10 * FPS for f in kos)]
                    kos_found = sum(1 for f in kos if any(s["k0"] - 10 * FPS <= f <= s["k1"] + 10 * FPS for s in sp))
                    right_votes = sum(1 for s in sp if s["a_left"] == tr)   # Home (A) in the left half <=> Home attacks right
                    key = f"{g} half {per} {'our noise seed ' + str(seed) if noisy else 'perfect'}"
                    rep[key] = {"truth_home_attacks_right": tr, "answer": None if res is None else res["A"], "right": None if res is None else res["A"] == tr,
                                "spells": len(sp), "spells_voting_right": right_votes, "kickoffs": len(kos), "kickoffs_found": kos_found,
                                "spells_away_from_kickoffs": len(sp) - len(at_ko),
                                "spells_away_right": sum(1 for s in sp if s not in at_ko and s["a_left"] == tr)}
                    print(key, json.dumps(rep[key]), flush=True)
    s = {"halves_answered_right": sum(1 for v in rep.values() if v["right"]), "halves_answered_wrong": sum(1 for v in rep.values() if v["right"] is False),
         "no_answer": sum(1 for v in rep.values() if v["right"] is None), "runs": len(rep),
         "spells": sum(v["spells"] for v in rep.values()), "spells_voting_right": sum(v["spells_voting_right"] for v in rep.values())}
    print("summary", json.dumps(s))
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump({"summary": s, "runs": rep}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
