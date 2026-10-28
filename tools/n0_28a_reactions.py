"""N0-28a: the app's 'reaction after a loss' labels - every turnover_lost event carries payload.reactions {pressed[], jogged[],
stood[]} (players of the team that lost it, over the first 2 s after the win) and stats.players[] carries counterpress_rate
(= pressed / losses_seen). Never checked before.

Rule (possession.reaction_label): pressed if the player ends within 5 m of the ball or the GAP to the ball shrank >= 2 m/s,
jogged >= 0.7 m/s, else stood. The gap also shrinks when the ball comes to him, so part 1 splits the closing speed into
his own run toward the ball ('own') and the ball's movement.

Part 1 (--part 1): Metrica, perfect positions, truth turnovers from the labelled events (BALL LOST -> winner's RECOVERY /
first on-ball event, as N0-23b). Per losing player: app label, 'own' label (own run toward the ball >= 2 m/s or within 5 m),
regain within 5 s (losing team's next RECOVERY / on-ball event within 5 s of the win).
Part 2 (--part 2): the pipeline's own turnovers + reactions (metricalab.ours, like run.py) on perfect / flicker ball, and
through our app clips' camera views (N0-23b view_frames), matched to the truth turnovers -> per-player agreement, coverage.

    PYTHONPATH=. python tools/n0_28a_reactions.py /tmp/metrica/data --part 1|2 [--out f.json]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P

FPS = 25.0; REACT_S = 2.0; NEAR_R = 5.0
ONBALL_REGAIN = ("RECOVERY", "PASS", "CARRY", "SHOT")


def own_label(path, balls, near_r=NEAR_R, press_v=2.0, jog_v=0.7):
    """path, balls: (n, 2) arrays over the reaction window. Own approach speed = mean of his velocity toward the ball's
    position at each frame (the ball's own movement left out)."""
    v = np.diff(path, axis=0) * FPS; to = balls[:-1] - path[:-1]; nrm = np.linalg.norm(to, axis=1)
    u = to / np.maximum(nrm, 1e-6)[:, None]; app = float(np.mean(np.sum(v * u, axis=1)))
    d2 = float(np.linalg.norm(path[-1] - balls[-1]))
    return ("pressed" if (d2 <= near_r or app >= press_v) else "jogged" if app >= jog_v else "stood"), round(app, 2)


def truth_turnovers(ev, per, recov_s=5.0):
    E = sorted([e for e in ev if e["period"] == per], key=lambda e: (e["f0"], e["f1"])); out = []
    for i, e in enumerate(E):
        if e["type"] != "BALL LOST" or e["team"] not in ("Home", "Away"): continue
        wt = "Away" if e["team"] == "Home" else "Home"; later = [q for q in E[i + 1:] if q["f0"] >= e["f0"]]
        win = next((q for q in later if q["team"] == wt and q["type"] == "RECOVERY" and q["f0"] - e["f0"] <= recov_s * FPS), None)
        if win is None: win = next((q for q in later if q["team"] == wt and q["type"] in ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "CHALLENGE")), None)
        if win is None: continue
        reg = any(q["team"] == e["team"] and q["type"] in ONBALL_REGAIN and win["f0"] < q["f0"] <= win["f0"] + 5 * FPS for q in later)
        out_ = any(q["type"] == "BALL OUT" and win["f0"] <= q["f0"] <= win["f0"] + REACT_S * FPS for q in later)
        out.append({"f_lost": e["f0"], "f_won": win["f0"], "lost_by": "A" if e["team"] == "Home" else "B", "won_by": "B" if e["team"] == "Home" else "A",
                    "regained_5s": reg, "ball_out_2s": out_})
    return out


def label_truth(fr, t):
    """app rule and own rule for every losing player seen at the win and 2 s later (perfect positions + ball)"""
    k0 = t["f_won"]; k2 = k0 + int(REACT_S * FPS); home = t["lost_by"] == "A"
    if k0 not in fr or k2 not in fr or fr[k0][2] is None or fr[k2][2] is None: return None
    ks = [k for k in range(k0, k2 + 1) if k in fr and fr[k][2] is not None]
    if len(ks) < int(REACT_S * FPS): return None
    pos = {k: {pid: (x, y) for pid, x, y in fr[k][1]} for k in ks}; balls = np.array([fr[k][2] for k in ks])
    b0, b2 = np.array(fr[k0][2]), np.array(fr[k2][2]); out = {}
    for pid in pos[k0]:
        if (pid < 200) != home or any(pid not in pos[k] for k in ks): continue
        path = np.array([pos[k][pid] for k in ks]); p0, p2 = path[0], path[-1]
        app_lab = P.reaction_label(p0, p2, b0, b2, REACT_S, NEAR_R, mode="gap")
        new_lab = P.reaction_label(p0, p2, b0, b2, REACT_S, NEAR_R, mode="own")
        own, own_v = own_label(path, balls)
        d0, d2 = float(np.linalg.norm(p0 - b0)), float(np.linalg.norm(p2 - b2))
        out[pid] = {"app": app_lab, "new": new_lab, "own": own, "own_v": own_v, "closing": round((d0 - d2) / REACT_S, 2), "d0": round(d0, 1), "d2": round(d2, 1),
                    "speed": round(float(np.linalg.norm(p2 - p0)) / REACT_S, 2)}
    return out


def load_game(d, g):
    import metricalab as ML
    from n0_23a_transition import load_events_xy
    gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
    fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
    return fr, load_events_xy(f"{gd}/{g}_RawEventsData.csv")


def keepers(fr, per):
    """per team the player deepest on average in the period's first minute (Metrica: 11 / 25 = keepers)"""
    ks = [k for k in sorted(fr) if fr[k][0] == per][:int(60 * FPS)]; xs = {}
    for k in ks:
        for pid, x, y in fr[k][1]: xs.setdefault(pid, []).append(x)
    m = {pid: np.mean(v) for pid, v in xs.items()}; out = set()
    for home in (True, False):
        tm = {p: v for p, v in m.items() if (p < 200) == home}; c = np.mean(list(tm.values()))
        out.add(max(tm, key=lambda p: abs(tm[p] - c)))
    return out


def summarise(rows):
    """rows: one per labelled player-turnover {app, own, regained, ...}"""
    n = len(rows); sh = lambda key, lab: round(100 * sum(r[key] == lab for r in rows) / max(1, n), 1)
    app_p = [r for r in rows if r["app"] == "pressed"]; own_p = [r for r in rows if r["own"] == "pressed"]
    out = {"n": n, "app_share": {l: sh("app", l) for l in ("pressed", "jogged", "stood")}, "own_share": {l: sh("own", l) for l in ("pressed", "jogged", "stood")},
           "agree_pct": round(100 * sum(r["app"] == r["own"] for r in rows) / max(1, n), 1),
           "new_share": {l: sh("new", l) for l in ("pressed", "jogged", "stood")}, "new_agree_own_pct": round(100 * sum(r["new"] == r["own"] for r in rows) / max(1, n), 1),
           "app_pressed_not_running": round(100 * sum(r["own_v"] < 0.7 and r["d2"] > NEAR_R for r in app_p) / max(1, len(app_p)), 1),
           "own_pressed_called_stood": round(100 * sum(r["app"] == "stood" for r in own_p) / max(1, len(own_p)), 1),
           "app_stood_moving_away_fast": round(100 * sum(r["own_v"] <= -2.0 for r in rows if r["app"] == "stood") / max(1, sum(r["app"] == "stood" for r in rows)), 1)}
    return out


def outcome(turns):
    """per turnover: number of pressers (app / own, outfield) vs regained within 5 s"""
    res = {}
    for key in ("app", "new", "own"):
        x = np.array([t["n_" + key] for t in turns]); y = np.array([t["regained_5s"] for t in turns], float)
        hi = x >= 2; lo = x <= 1
        res[key] = {"corr": round(float(np.corrcoef(x, y)[0, 1]), 3) if x.std() > 0 else None,
                    "regain_2plus": round(100 * float(y[hi].mean()), 1) if hi.any() else None, "n_2plus": int(hi.sum()),
                    "regain_0_1": round(100 * float(y[lo].mean()), 1) if lo.any() else None, "n_0_1": int(lo.sum()),
                    "median_pressers": float(np.median(x))}
    return res


def part1(d, games=("Sample_Game_1", "Sample_Game_2")):
    allrows, turns, per_player = [], [], {}
    for g in games:
        fr, ev = load_game(d, g)
        for per in (1, 2):
            gk = keepers(fr, per)
            for t in truth_turnovers(ev, per):
                if t["ball_out_2s"]: continue
                lab = label_truth(fr, t)
                if not lab: continue
                lab = {pid: v for pid, v in lab.items() if pid not in gk}
                for pid, v in lab.items():
                    allrows.append(dict(v, regained=t["regained_5s"], game=g[-1], half=per, pid=pid))
                    pp = per_player.setdefault(f"{g[-1]}:{pid}", {"h1": [0, 0], "h2": [0, 0], "own_h1": [0, 0], "own_h2": [0, 0], "new_h1": [0, 0], "new_h2": [0, 0]})
                    pp[f"new_h{per}"][0] += v["new"] == "pressed"; pp[f"new_h{per}"][1] += 1
                    pp[f"h{per}"][0] += v["app"] == "pressed"; pp[f"h{per}"][1] += 1
                    pp[f"own_h{per}"][0] += v["own"] == "pressed"; pp[f"own_h{per}"][1] += 1
                turns.append({"game": g[-1], "half": per, "f_won": t["f_won"], "regained_5s": t["regained_5s"], "n_players": len(lab),
                              "n_app": sum(v["app"] == "pressed" for v in lab.values()), "n_own": sum(v["own"] == "pressed" for v in lab.values()), "n_new": sum(v["new"] == "pressed" for v in lab.values())})
    # does a player's counterpress rate mean something? same player, first half vs second half (>= 8 losses seen in each)
    rel = {}
    for key in ("h", "own_h", "new_h"):
        a = [(v[key + "1"][0] / v[key + "1"][1], v[key + "2"][0] / v[key + "2"][1]) for v in per_player.values() if v[key + "1"][1] >= 8 and v[key + "2"][1] >= 8]
        if len(a) >= 5:
            from scipy.stats import spearmanr
            rel[key] = {"players": len(a), "spearman_h1_h2": round(float(spearmanr([x for x, _ in a], [y for _, y in a])[0]), 3)}
    return {"summary": summarise(allrows), "outcome": outcome(turns), "turnovers": len(turns), "rate_reliability": rel,
            "by_half": {f"{g}:{h}": summarise([r for r in allrows if r["game"] == g and r["half"] == h]) for g in ("1", "2") for h in (1, 2)},
            "examples": sorted([r for r in allrows if r["app"] == "pressed" and r["own_v"] < 0.7 and r["d2"] > NEAR_R], key=lambda r: r["own_v"])[:12]}, turns


def part2(d, games=("Sample_Game_1", "Sample_Game_2"), seeds=(0, 1)):
    from n0_23a_transition import ours_kept
    import n0_23b_press as NB
    import n0_22e_shape_view as SV, n0_22c_metrica as C
    runs, gaps, jit = C.our_breaks(list(NB.CLIPS)); V = [SV.view_masks(c) for c in NB.CLIPS]
    out = []
    for g in games:
        fr, ev = load_game(d, g)
        for per in (1, 2):
            gk = keepers(fr, per); T = [t for t in truth_turnovers(ev, per) if not t["ball_out_2s"]]
            truth = {t["f_won"]: label_truth(fr, t) for t in T}; f0 = min(k for k in fr if fr[k][0] == per)
            for noise, view, mode in [(n_, v_, m_) for n_, v_ in ((False, False), ("flicker", False), (False, True), ("flicker", True)) for m_ in ("gap", "own")]:
                for seed in (seeds if (noise or view) else (0,)):
                    fv, share = fr, 1.0
                    if view:
                        v, Lv, Wv = V[seed % len(V)]; fv, share = NB.view_frames(fr, per, v, Lv, Wv, runs, gaps, jit, seed)
                    P.REACTION = mode
                    try: o = ours_kept(fv, per, noise, seed)
                    finally: P.REACTION = "own"
                    kept = [u for u in o["kept"] if u.get("reactions") is not None]
                    used = set(); agree = tot = cover_ours = cover_truth = agree_run = 0; conf = {}; npress_err = []
                    for t in T:
                        lab = truth[t["f_won"]]
                        if not lab: continue
                        lab = {p: v for p, v in lab.items() if p not in gk}
                        best = None
                        for j, u in enumerate(kept):
                            if j in used or u["won_by"] != t["won_by"]: continue
                            dt = abs(f0 + u["frame"] - t["f_won"])
                            if dt <= 2 * FPS and (best is None or dt < best[0]): best = (dt, j)
                        if best is None: continue
                        used.add(best[1]); rx = kept[best[1]]["reactions"]
                        ours = {p: c for c, lst in rx.items() for p in lst if p not in gk}
                        cover_truth += len(lab); cover_ours += sum(1 for p in lab if p in ours)
                        for p, v in lab.items():
                            if p in ours:
                                tot += 1; agree += ours[p] == v["app" if mode == "gap" else "new"]; agree_run += ours[p] == v["own"]; conf[f"{v['app']}->{ours[p]}"] = conf.get(f"{v['app']}->{ours[p]}", 0) + 1
                        npress_err.append(sum(c == "pressed" for c in ours.values()) - sum(v["app" if mode == "gap" else "new"] == "pressed" for v in lab.values()))
                    row = {"game": g[-1], "half": per, "noise": str(noise) + ("+view" if view else ""), "mode": mode, "seed": seed, "seen_share": share,
                           "truth_turnovers": len(T), "matched": len(npress_err), "players_labelled_pct": round(100 * cover_ours / max(1, cover_truth), 1),
                           "label_agree_pct": round(100 * agree / max(1, tot), 1), "agree_with_real_run_pct": round(100 * agree_run / max(1, tot), 1), "pairs": tot, "confusion": conf,
                           "pressed_count_err_median": float(np.median(npress_err)) if npress_err else None,
                           "pressed_count_abs_err_mean": round(float(np.mean(np.abs(npress_err))), 2) if npress_err else None}
                    out.append(row); print(json.dumps(row), flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; part = a[a.index("--part") + 1] if "--part" in a else "1"
    out = a[a.index("--out") + 1] if "--out" in a else f"results/qa/n0-28a/metrica_reactions_part{part}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if part == "1":
        r, turns = part1(a[0]); print(json.dumps({k: v for k, v in r.items() if k != "examples"}, indent=1)); json.dump(r, open(out, "w"), indent=1)
    else:
        json.dump(part2(a[0]), open(out, "w"), indent=1)
