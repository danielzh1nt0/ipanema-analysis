"""Offline event lab (27 Sep): rebuild possession / dead ball / restarts from an exported clip (match_data.json) and score
the restarts against Veo's own event list, per minute. No cloud, no cost; every change to the rules is re-scored here."""
import json, sys, collections, numpy as np
sys.path.insert(0, ".")
from ipanema import possession as P

VEO_KIND = {"Throw-in": "throw-in", "Corner": "corner", "Goal kick": "goal kick", "Free kick": "free kick"}

def load(path):
    d = json.load(open(path)); F = d["frames"]; fps = d["fps"]; L, W = d["pitch"]["length"], d["pitch"]["width"]
    per = {}; ballm = {}; conf = {}; seen = {}
    for k, f in enumerate(F):
        per[k] = [[p["id"], p["team"], np.array(p["m"], float), np.array(p["px"], float), None, p["gk"]] for p in f["players"]]
        b = f.get("ball")
        if b and b.get("m") is not None: ballm[k] = np.array(b["m"], float); conf[k] = b.get("conf", 0.5); seen[k] = b.get("state") == "observed"
    return d, per, ballm, conf, seen, fps, L, W

def veo_truth(path, t0_s, t1_s, first_half_offset_s=0.0):
    """Veo minute m covers video [(m-1)*60, m*60) in the first half"""
    out = []
    for line in open(path):
        p = line.split()
        if not p or not p[0].isdigit(): continue
        m = int(p[0]); kind = " ".join(p[2:]); s0 = (m - 1) * 60 + first_half_offset_s
        if kind in VEO_KIND and s0 >= t0_s - 1e-6 and s0 < t1_s: out.append({"minute": m, "team": p[1], "kind": VEO_KIND[kind], "t0": s0 - t0_s, "t1": s0 - t0_s + 60})
    return out

def score(ours, truth):
    """per minute window: counts by kind, ours vs Veo; plus totals"""
    rows = []; tot = {"veo": len(truth), "ours": len(ours), "matched": 0}
    wins = sorted({(t["t0"], t["t1"]) for t in truth})
    used = set()
    for t in truth:
        cand = [i for i, o in enumerate(ours) if i not in used and t["t0"] - 5 <= o["t"] < t["t1"] + 5]
        same = [i for i in cand if ours[i]["kind"] == t["kind"]]
        pick = (same or cand or [None])[0]
        if pick is not None: used.add(pick); tot["matched"] += 1
        rows.append((t["minute"], t["kind"], t["team"], None if pick is None else f"{ours[pick]['t']:.0f}s {ours[pick]['kind']}"))
    extra = [o for i, o in enumerate(ours) if i not in used]
    return tot, rows, extra

if __name__ == "__main__":
    d, per, ballm, conf, seen, fps, L, W = load("results/volume/runs/matches/SFKBP1109_s1200/match_data.json")
    state, bspeed = P.viterbi(per, dict(ballm), fps, L, W); rst = P.restarts(state, ballm, fps, L, W)
    print("rebuilt restarts:", len(rst), " dead share %.0f%%" % (100 * np.mean(np.array(state) == 3)))
    truth = veo_truth("reference/veo_events_SFKBP1109.txt", 1200, 1500)
    tot, rows, extra = score(rst, truth); print(tot); [print(r) for r in rows]; print("extra:", len(extra))

def outside(b, L, W): return max(-b[0], b[0] - L, -b[1], b[1] - W, 0.0)

def clean_ball(ballm, L, W, max_out=6.0):
    """a pick more than max_out metres outside the pitch is not the match ball (spare balls, fence, trees): no observation"""
    return {k: v for k, v in ballm.items() if outside(v, L, W) <= max_out}

def stoppages(state, fps, min_s=2.0, join_gap_s=1.5):
    """dead runs joined across short gaps, kept when they last min_s: (start, end) frame pairs"""
    st = np.asarray(state); n = len(st); runs = []; i = 0
    while i < n:
        if st[i] == 3:
            j = i
            while j + 1 < n and st[j + 1] == 3: j += 1
            if runs and i - runs[-1][1] <= join_gap_s * fps: runs[-1][1] = j
            else: runs.append([i, j])
            i = j + 1
        else: i += 1
    return [(a, b) for a, b in runs if (b - a + 1) / fps >= min_s]
