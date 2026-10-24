"""N0-24e: are the app's restarts (stats.restarts[], HIDE) right? Checked on Metrica pro data.

Truth: Metrica's SET PIECE events (throw-ins, corners, free kicks, kick-offs, penalties) + goal kicks (PASS / BALL LOST
with subtype GOAL KICK), with frame, team, spot. Ours: the pipeline (metricalab.ours: perfect ball and our noise models)
with P.restarts captured. A truth restart is found when one of ours is within 5 s (metricalab.match_frames' window,
one-to-one). Kinds compared: throw-in / corner / goal kick (ours never calls a free kick or kick-off: 'stoppage').

Lab: P.GAP_DEAD_S / GAP_EDGE_M / GAP_SIDE (an unseen-ball gap next to a line = dead). Metrica blanks the ball while it is
out of play, so the old rule (dead = ball still with nobody near, or ball off the pitch) never sees those stoppages.

    PYTHONPATH=. python tools/n0_24e_restarts.py /tmp/metrica/data [--out results/qa/n0-24e/metrica_restarts.json]
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P

L, W, FPS = 105.0, 68.0, 25.0
TOL = 5 * FPS
KIND = {"THROW IN": "throw-in", "CORNER KICK": "corner", "GOAL KICK": "goal kick", "FREE KICK": "free kick",
        "FREE KICK-RETAKEN": "free kick", "KICK OFF": "kick-off", "KICK OFF-RETAKEN": "kick-off", "PENALTY": "penalty"}
OUT_OF_PLAY = ("throw-in", "corner", "goal kick")   # the ball went out: what a dead-ball rule can find


def truth_restarts(ev_rows, per):
    out = []
    for r in ev_rows:
        if int(r["Period"]) != per or r["Team"] not in ("Home", "Away"): continue
        if r["Type"] == "SET PIECE" or (r["Type"] in ("PASS", "BALL LOST") and r["Subtype"].startswith("GOAL KICK")):
            sub = "GOAL KICK" if r["Subtype"].startswith("GOAL KICK") else r["Subtype"]
            if sub not in KIND: continue
            out.append({"frame": int(r["Start Frame"]), "kind": KIND[sub], "team": "A" if r["Team"] == "Home" else "B"})
    # a goal kick is labelled twice when Metrica has SET PIECE + PASS GOAL KICK on the same frame: keep one
    seen, uniq = set(), []
    for t in sorted(out, key=lambda t: t["frame"]):
        key = (t["frame"] // 13, t["kind"])
        if key in seen: continue
        seen.add(key); uniq.append(t)
    return uniq


def match(truth, ours, tol=TOL):
    """one-to-one nearest pairs within tol -> list of (truth i, ours j)"""
    pairs = sorted((abs(o["frame"] - t["frame"]), i, j) for i, t in enumerate(truth) for j, o in enumerate(ours) if abs(o["frame"] - t["frame"]) <= tol)
    ut, uo, out = set(), set(), []
    for _, i, j in pairs:
        if i in ut or j in uo: continue
        ut.add(i); uo.add(j); out.append((i, j))
    return out


def ours_restarts(fr, per, noisy, seed, gap=None):
    """-> our restarts with Metrica frame numbers; gap = (gap_s, edge_m, side) or None (pipeline default)"""
    import metricalab as ML
    got = {}; o_rs = P.restarts; old = (P.GAP_DEAD_S, P.GAP_EDGE_M, P.GAP_SIDE)
    def rs_(*a, **k): r = o_rs(*a, **k); got["rst"] = r; return r
    P.restarts = rs_
    if gap is not None: P.GAP_DEAD_S, P.GAP_EDGE_M, P.GAP_SIDE = gap
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.restarts = o_rs; P.GAP_DEAD_S, P.GAP_EDGE_M, P.GAP_SIDE = old
    k0 = min(k for k in fr if fr[k][0] == per)
    return [{**r, "frame": k0 + int(round(r["t"] * FPS))} for r in got["rst"]], o["attack_right"]


def score(truth, ours, minutes):
    m = match(truth, ours)
    oop = [i for i, t in enumerate(truth) if t["kind"] in OUT_OF_PLAY]; mi = {i: j for i, j in m}
    kind_ok = sum(1 for i in oop if i in mi and ours[mi[i]]["kind"] == truth[i]["kind"])
    team_ok = sum(1 for i, j in m if ours[j]["team"] == truth[i]["team"])
    return {"truth": len(truth), "truth_out_of_play": len(oop), "ours": len(ours), "found": len(m), "found_out_of_play": sum(1 for i in oop if i in mi),
            "extra": len(ours) - len(m), "kind_right_out_of_play": kind_ok, "team_right": team_ok,
            "ours_per_90": round(len(ours) / minutes * 90, 1), "truth_per_90": round(len(truth) / minutes * 90, 1),
            "found_by_kind": {k: f"{sum(1 for i, t in enumerate(truth) if t['kind'] == k and i in mi)}/{sum(1 for t in truth if t['kind'] == k)}" for k in sorted({t['kind'] for t in truth})}}


VARIANTS = {"old": (0.0, 3.0, "either"), "gap2_edge3": (2.0, 3.0, "either"), "gap3_edge3": (3.0, 3.0, "either"),
            "gap2_edge3_before": (2.0, 3.0, "before"), "gap2_edge3_after": (2.0, 3.0, "after"), "gap2_edge6": (2.0, 6.0, "either"),
            "gap4_edge3": (4.0, 3.0, "either")}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1), variants=VARIANTS):
    from n0_24d_ppda import load_game
    rows = []
    for g in games:
        fr, ev = load_game(d, g)
        for per in halves:
            T = truth_restarts(ev, per); mins = sum(1 for k in fr if fr[k][0] == per) / FPS / 60
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    for vn, gap in variants.items():
                        O, _ = ours_restarts(fr, per, noisy, seed, gap)
                        row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "variant": vn, "minutes": round(mins, 1), **score(T, O, mins)}
                        rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-24e/metrica_restarts.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(main(a[0]), open(out, "w"), indent=1)
