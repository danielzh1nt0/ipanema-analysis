"""N0-27b: does the app's pass quality mean anything? (stats.json players[].pass_quality {good, risky_completed, bad_lost,
execution_error}, passes[].lane_open / quality, events pass_risky / pass_bad (hidden), the app's 'risky passes' =
pass_quality.risky_completed, and the open/blocked lanes drawn per frame). Never checked before.

Rule (analytics._blocked / lanes, unchanged): at the passer's last frame on the ball the lane to the receiver is
'blocked' if an opponent stands within 1.5 m of the straight line (10-95% of the way). quality = good (open, completed),
risky_completed (blocked, completed), bad_lost (blocked, lost), execution_error (open, lost).

Part 1 - the rule on PERFECT inputs (Metrica: all 22 players + the ball). Completed PASS events and passes cut out
(BALL LOST ... INTERCEPTION, intended receiver = the passer's team-mate nearest the spot where it was cut out).
A rule that says something calls a lane blocked much more often for passes that were cut out.
Variants: lane half-width, a lane that widens with the distance (an opponent has more time to step in on a long pass).
Part 2 - our inputs (tools/metricalab.ours + analytics.lanes like run.py) vs the rule on perfect inputs, per matched pass.

    PYTHONPATH=. python tools/n0_27b_lanes.py <metrica data dir> [--out results/qa/n0-27b/metrica_lanes.json] [--part 1|2]
    PYTHONPATH=. python tools/n0_27b_lanes.py --dry
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN

L, W, FPS = 105.0, 68.0, 25.0
VARIANTS = {"app (1.5 m)": (1.5, 0.0), "1.0 m": (1.0, 0.0), "2.5 m": (2.5, 0.0), "4.0 m": (4.0, 0.0),
            "widen 0.1/m": (1.5, 0.1), "widen 0.2/m": (1.5, 0.2), "widen 0.3/m": (1.0, 0.3)}


def blocked(c, t, opps, half, widen=0.0):
    """analytics._blocked with a lane that widens by `widen` m per metre from the passer (widen 0 = the app's rule)"""
    if not widen: return AN._blocked(c, t, opps, half)
    d = t - c; n2 = float(d @ d)
    if n2 < 1e-6: return False
    for o in opps:
        u = float((o - c) @ d) / n2
        if 0.1 < u < 0.95 and np.linalg.norm(c + u * d - o) < half + widen * u * np.sqrt(n2): return True
    return False


def truth_passes(E, fr, per, proxy="nearest"):
    """proxy (cut-out passes only): 'nearest' = the team-mate nearest the cut-out spot; 'line' = the team-mate at least as far
    as the cut-out spot whose direction from the passer is closest to it (the interceptor stands at that spot, so the
    nearest mate can be picked BECAUSE he is near the interceptor - 'line' checks that bias).
    (team, f0, passer, receiver, passer xy, receiver xy, opponents xy, failed) for every real pass the rule can see"""
    out = []
    for p in E:
        if p["period"] != per or p["f0"] not in fr or fr[p["f0"]][2] is None or p["x1"] is None: continue
        _, ps, b = fr[p["f0"]]; A = p["team"] == "A"
        mine = [(np.hypot(x - b[0], y - b[1]), pid) for pid, x, y in ps if (pid < 200) == A]
        if not mine or min(mine)[0] > 2.5: continue
        me = min(mine)[1]; e = np.array([p["x1"] * L, p["y1"] * W]); fe = p["f1"] if p["f1"] in fr else p["f0"]
        rec = [(np.hypot(x - e[0], y - e[1]), pid) for pid, x, y in fr[fe][1] if (pid < 200) == A and pid != me]
        if not rec: continue
        to = min(rec)[1]; pos = {pid: np.array([x, y], float) for pid, x, y in ps}
        if p["failed"] and proxy == "line":
            c0 = pos[me]; de = e - c0; ne = float(np.linalg.norm(de)) or 1.0
            cand = [(float(np.arccos(np.clip((pos[q] - c0) @ de / (np.linalg.norm(pos[q] - c0) * ne + 1e-9), -1, 1))), q) for q in pos
                    if (q < 200) == A and q != me and np.linalg.norm(pos[q] - c0) >= ne - 2.0]
            if cand: to = min(cand)[1]
        if to not in pos or np.linalg.norm(pos[to] - pos[me]) > 45.0: continue          # the app values lanes up to 45 m
        opps = [pos[pid] for pid in pos if (pid < 200) != A]
        out.append({"team": p["team"], "f0": p["f0"], "from": me, "to": to, "c": pos[me], "t": pos[to], "opps": opps, "failed": p["failed"]})
    return out


def part1(d, games=("Sample_Game_1", "Sample_Game_2"), proxy="nearest"):
    import metricalab as ML
    from n0_27a_rule import load
    tot = {v: {"completed": [0, 0], "failed": [0, 0]} for v in VARIANTS}; halves = []; lens = {"completed": [], "failed": []}
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; E = load(f"{gd}/{g}_RawEventsData.csv")
        for per in (1, 2):
            hh = {v: {"completed": [0, 0], "failed": [0, 0]} for v in VARIANTS}
            for p in truth_passes(E, fr, per, proxy):
                k = "failed" if p["failed"] else "completed"; lens[k].append(float(np.linalg.norm(p["t"] - p["c"])))
                for v, (h, w) in VARIANTS.items():
                    b = blocked(p["c"], p["t"], p["opps"], h, w)
                    for R in (tot, hh): R[v][k][0] += 1; R[v][k][1] += int(b)
            halves.append({"game": g[-1], "half": per, **{v: summarise(x) for v, x in hh.items()}}); print(json.dumps(halves[-1]), flush=True)
    return {"total": {v: summarise(x) for v, x in tot.items()}, "halves": halves,
            "length_median_m": {k: round(float(np.median(v)), 1) for k, v in lens.items() if v}}


def summarise(x):
    (nc, bc), (nf, bf) = x["completed"], x["failed"]
    pc, pf = bc / max(1, nc), bf / max(1, nf)
    fail_blocked = bf / max(1, bf + bc); fail_open = (nf - bf) / max(1, (nf - bf) + (nc - bc))
    return {"n_completed": nc, "n_failed": nf, "blocked_completed_pct": round(100 * pc, 1), "blocked_failed_pct": round(100 * pf, 1),
            "cut_out_if_blocked_pct": round(100 * fail_blocked, 1), "cut_out_if_open_pct": round(100 * fail_open, 1),
            "lift": round(fail_blocked / fail_open, 2) if fail_open else None}


def quality(open_, completed):
    return "good" if open_ and completed else "risky_completed" if completed else "bad_lost" if not open_ else "execution_error"


def part2(d, games=("Sample_Game_1", "Sample_Game_2"), variants=((False, False), ("flicker", False), (False, True), ("flicker", True)), seeds=(0, 1)):
    """ours vs truth per matched pass (same team, |t| <= 1 s, same outcome): lane open agrees? quality agrees?"""
    import metricalab as ML, n0_23b_press as PR
    from n0_23a_transition import home_attacks_right
    from n0_27a_rule import load
    from n0_27a_better_option import ours_passes, CLIPS
    import n0_22e_shape_view as SV, n0_22c_metrica as C
    runs, gaps, jit = C.our_breaks(list(CLIPS)); V = [SV.view_masks(c) for c in CLIPS]
    out = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; E = load(f"{gd}/{g}_RawEventsData.csv")
        for per in (1, 2):
            T = truth_passes(E, fr, per); f0 = min(k for k in fr if fr[k][0] == per)
            for t in T: t["open"] = not blocked(t["c"], t["t"], t["opps"], 1.5); t["q"] = quality(t["open"], not t["failed"])
            for noisy, view in variants:
                for seed in (seeds if noisy or view else (0,)):
                    fv, share = fr, 1.0
                    if view:
                        v, Lv, Wv = V[seed % len(V)]; fv, share = PR.view_frames(fr, per, v, Lv, Wv, runs, gaps, jit, seed)
                    ps, _ = ours_passes(fv, per, noisy, seed)
                    row = {"game": g[-1], "half": per, "noise": str(noisy) + ("+view" if view else ""), "seed": seed, "seen_share": share, **compare(T, ps, f0)}
                    out.append(row); print(json.dumps(row), flush=True)
    return out


def compare(T, ps, f0, tol_s=1.0):
    valued = [u for u in ps if u.get("lane_open") is not None]
    qs = lambda xs, key: {q: sum(1 for x in xs if x[key] == q) for q in ("good", "risky_completed", "bad_lost", "execution_error")}
    used = set(); pairs = []
    for t in T:
        best = None
        for j, u in enumerate(valued):
            if j in used or u["team"] != t["team"] or u["completed"] == t["failed"]: continue
            dt = abs(f0 + u["t"] * FPS - t["f0"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is not None: used.add(best[1]); pairs.append((t, valued[best[1]]))
    n = len(pairs); same_to = [(a, b) for a, b in pairs if a["to"] == b.get("intended_to", b["to"])]
    bc = sum(u["completed"] and not u["lane_open"] for u in valued); nc = sum(u["completed"] for u in valued)
    bf = sum((not u["completed"]) and not u["lane_open"] for u in valued); nf = sum(not u["completed"] for u in valued)
    lost_pairs = [(a, b) for a, b in pairs if a["failed"]]
    return {"truth_n": len(T), "ours_n": len(ps), "ours_valued": len(valued), "matched": n,
            "truth_quality": qs(T, "q"), "ours_quality": qs(valued, "quality"),
            "risky_share_truth_pct": round(100 * sum(t["q"] == "risky_completed" for t in T) / max(1, sum(not t["failed"] for t in T)), 1),
            "risky_share_ours_pct": round(100 * sum(u["quality"] == "risky_completed" for u in valued) / max(1, sum(u["completed"] for u in valued)), 1),
            "lane_agree_pct": round(100 * sum(a["open"] == b["lane_open"] for a, b in pairs) / max(1, n), 1),
            "lane_agree_same_receiver_pct": round(100 * sum(a["open"] == b["lane_open"] for a, b in same_to) / max(1, len(same_to)), 1),
            "same_receiver": len(same_to),
            "ours_lift": summarise({"completed": [nc, bc], "failed": [nf, bf]}),
            "lost_matched": len(lost_pairs), "lost_quality_agree_pct": round(100 * sum(a["q"] == b["quality"] for a, b in lost_pairs) / max(1, len(lost_pairs)), 1),
            "quality_agree_pct": round(100 * sum(a["q"] == b["quality"] for a, b in pairs) / max(1, n), 1),
            "ours_blocked_real_pct": round(100 * sum((not a["open"]) and (not b["lane_open"]) for a, b in pairs) / max(1, sum(not b["lane_open"] for a, b in pairs)), 1),
            "truth_blocked_found_pct": round(100 * sum((not a["open"]) and (not b["lane_open"]) for a, b in pairs) / max(1, sum(not a["open"] for a, b in pairs)), 1)}


def dry():
    """stand-in: an opponent on the line blocks the lane; one 3 m off does not; a widening lane catches him near the end"""
    c, t = np.array([0.0, 0.0]), np.array([30.0, 0.0])
    assert blocked(c, t, [np.array([15.0, 1.0])], 1.5) and not blocked(c, t, [np.array([15.0, 3.0])], 1.5)
    assert not blocked(c, t, [np.array([25.0, 3.5])], 1.5) and blocked(c, t, [np.array([25.0, 3.5])], 1.5, 0.1)
    assert blocked(c, t, [np.array([15.0, 1.0])], 1.5, 0.0) == AN._blocked(c, t, [np.array([15.0, 1.0])], 1.5)
    s = summarise({"completed": [80, 20], "failed": [20, 10]})
    assert s["blocked_completed_pct"] == 25.0 and s["blocked_failed_pct"] == 50.0 and s["cut_out_if_blocked_pct"] == 33.3 and s["lift"] == 2.33, s
    assert [quality(o, c) for o, c in ((1, 1), (0, 1), (0, 0), (1, 0))] == ["good", "risky_completed", "bad_lost", "execution_error"]
    T = [{"team": "A", "f0": 100, "to": 5, "failed": False, "open": False, "q": "risky_completed"}]
    ps = [{"team": "A", "t": 4.0, "completed": True, "to": 5, "lane_open": False, "quality": "risky_completed"}]
    r = compare(T, ps, 0); assert r["matched"] == 1 and r["lane_agree_pct"] == 100.0 and r["ours_blocked_real_pct"] == 100.0, r
    return {"ok": True}


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--dry" in a: print(json.dumps(dry())); sys.exit(0)
    part = a[a.index("--part") + 1] if "--part" in a else "1"
    out = a[a.index("--out") + 1] if "--out" in a else f"results/qa/n0-27b/metrica_lanes_part{part}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    proxy = a[a.index("--proxy") + 1] if "--proxy" in a else "nearest"
    r = part1(a[0], proxy=proxy) if part == "1" else part2(a[0])
    json.dump(r, open(out, "w"), indent=1)
    if part == "1": print(json.dumps(r["total"], indent=1), r["length_median_m"])
