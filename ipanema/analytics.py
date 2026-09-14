"""Lanes, passes (quality + best option), shapes, player and team stats — all from tracking + possession."""
import numpy as np, pandas as pd, cv2
from collections import defaultdict

def _blocked(c, t, opps, lane_half):
    d = t - c; n2 = d @ d
    if n2 < 1e-6: return False
    for o in opps:
        u = ((o - c) @ d) / n2
        if 0.1 < u < 0.95 and np.linalg.norm(c + u * d - o) < lane_half: return True
    return False

def lanes(per, frames_, attack_right, lane_half=1.5, max_lane=45.0, fwd_m=3.0):
    out = {}
    for k, f in enumerate(frames_):
        c = f["carrier"]
        if c is None: continue
        rows = per[k]; me = next((r for r in rows if r[0] == c), None)
        if me is None: continue
        tm = me[1]; cpos = me[2]; opps = [r[2] for r in rows if r[1] != tm]; sgn = 1 if attack_right[tm] else -1; res = []
        for r in rows:
            if r[1] != tm or r[0] == c or np.linalg.norm(r[2] - cpos) > max_lane: continue
            res.append({"to": int(r[0]), "open": not _blocked(cpos, r[2], opps, lane_half), "forward": bool((r[2][0] - cpos[0]) * sgn > fwd_m)})
        out[k] = res
    return out

def passes(per, frames_, turnovers_, lanes_, attack_right, fps, prog_m=10.0, lane_half=1.5):
    tracks = defaultdict(dict)
    for k in range(len(per)):
        for r in per[k]: tracks[r[0]][k] = r[2]
    episodes = []; cur = None
    for k, f in enumerate(frames_):
        c = f["carrier"]
        if c is None:
            if cur and k - cur[1] > int(0.4 * fps): episodes.append(cur); cur = None
            continue
        if cur and cur[2] == c: cur[1] = k
        else:
            if cur: episodes.append(cur)
            cur = [k, k, c, f["team"]]
    if cur: episodes.append(cur)
    out = []
    for a, b in zip(episodes, episodes[1:]):
        if b[0] - a[1] > int(4 * fps): continue
        pa, pb = tracks[a[2]].get(a[1]), tracks[b[2]].get(b[0])
        if pa is None or pb is None: continue
        sgn = 1 if attack_right[a[3]] else -1; gain = float((pb[0] - pa[0]) * sgn)
        kind = "forward" if gain > 3 else "backward" if gain < -3 else "sideways"
        p = {"t": round(a[1] / fps, 2), "from": int(a[2]), "to": int(b[2]), "team": a[3], "completed": b[3] == a[3], "from_m": [round(float(pa[0]), 1), round(float(pa[1]), 1)],
             "to_m": [round(float(pb[0]), 1), round(float(pb[1]), 1)], "gain_m": round(gain, 1), "length_m": round(float(np.linalg.norm(pb - pa)), 1), "kind": kind, "progressive": gain >= prog_m and b[3] == a[3],
             "lane_open": None, "quality": "unknown", "missed_open_forward": None, "better_option": False}
        k_rel = a[1]; win = [j for j in range(k_rel - 3, k_rel + 1) if j in lanes_]
        if win:
            j = win[-1]; ln = {l["to"]: l for l in lanes_[j]}
            if p["to"] in ln:
                op = ln[p["to"]]["open"]; p["lane_open"] = op
                p["quality"] = "good" if op and p["completed"] else "risky_completed" if (not op) and p["completed"] else "bad_lost" if (not op) else "execution_error"
                p["missed_open_forward"] = bool(any(l["open"] and l["forward"] for l in lanes_[j] if l["to"] != p["to"]) and kind != "forward")
                # best option valuation
                rows = per[j]; me = next((r for r in rows if r[0] == a[2]), None)
                if me is not None:
                    cpos = me[2]; opps = [r[2] for r in rows if r[1] != a[3]]; pos = {r[0]: r[2] for r in rows}
                    def val(rpos, is_open):
                        g = float((rpos[0] - cpos[0]) * sgn); lo, hi = sorted([cpos[0], rpos[0]])
                        byp = int(sum(1 for o in opps if lo < o[0] < hi and abs(o[1] - (cpos[1] + rpos[1]) / 2) < 20)); sp = float(min((np.linalg.norm(o - rpos) for o in opps), default=30.0))
                        return {"gain_m": round(g, 1), "bypassed": byp, "space_m": round(sp, 1), "value": round((g + 4.0 * byp + min(sp, 10.0) * 0.5) - (0 if is_open else 25.0), 1), "open": bool(is_open)}
                    opts = {l["to"]: val(pos[l["to"]], l["open"]) for l in lanes_[j] if l["to"] in pos}
                    if opts:
                        played = opts.get(p["to"]); bt = max(opts, key=lambda t: opts[t]["value"]); best = opts[bt]
                        p.update({"played_value": played["value"] if played else None, "best_to": int(bt), "best_value": best["value"], "best_gain_m": best["gain_m"], "best_bypassed": best["bypassed"], "best_space_m": best["space_m"], "best_open": best["open"],
                                  "better_option": bool(played is not None and bt != p["to"] and best["value"] >= played["value"] + 6.0)})
        out.append(p)
    return out, tracks

def shapes(per, L):
    """per frame, per team: hull (metres), block length/width, deepest/highest x, excluding keepers."""
    out = {}
    for k in range(len(per)):
        sh = {}
        for tm in ("A", "B"):
            pts = np.array([r[2] for r in per[k] if r[1] == tm and not r[5]])
            if len(pts) >= 3:
                hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2); xs = np.sort(pts[:, 0])
                sh[tm] = {"hull_m": [[round(float(x), 1), round(float(y), 1)] for x, y in hull], "n": int(len(pts)), "length": round(float(xs[-1] - xs[0]), 1), "width": round(float(pts[:, 1].max() - pts[:, 1].min()), 1), "x_min": round(float(xs[0]), 1), "x_max": round(float(xs[-1]), 1)}
            else: sh[tm] = None
        out[k] = sh
    return out

def stats(per, frames_, turnovers_, passes_, tracks, state, fps, L, W, attack_right, press_r=2.0, near_r=5.0):
    from .possession import STATES
    n = len(per); dt = 1 / fps; pass_df = pd.DataFrame(passes_)
    ids = [t for t in tracks if len(tracks[t]) * dt >= 2.0]
    team_of = {}
    for k in range(n):
        for r in per[k]: team_of.setdefault(r[0], []).append(r[1])
    team_id = {t: max(set(team_of[t]), key=team_of[t].count) for t in ids}
    react = {}
    for tv in turnovers_:
        for cat, lst in tv.get("reactions", {}).items():
            for tid in lst: d = react.setdefault(tid, {"losses_seen": 0, "pressed": 0, "jogged": 0, "stood": 0}); d["losses_seen"] += 1; d[cat] += 1
    players = []; heat = {}; GRID = (12, 8)
    for tid in ids:
        tm = team_id[tid]; ks = np.array(sorted(tracks[tid])); P = np.array([tracks[tid][k] for k in ks]); sgn = 1 if attack_right[tm] else -1
        Ps = pd.DataFrame(P).rolling(max(1, int(0.8 * fps)), center=True, min_periods=1).mean().values; V = np.full_like(P, np.nan)
        for i in range(1, len(ks)):
            gap = ks[i] - ks[i - 1]
            if gap <= 3:
                v = (Ps[i] - Ps[i - 1]) / (gap * dt); s = np.linalg.norm(v)
                if s <= 10: V[i] = v if s >= 0.5 else 0.0
        sp = np.linalg.norm(V, axis=1); valid = ~np.isnan(sp); step = np.where(valid, sp * np.r_[0, np.diff(ks)] * dt, 0)
        on = [k for k in ks if frames_[k]["carrier"] == tid]; pressed = [k for k in on if frames_[k]["pressure_m"] is not None and frames_[k]["pressure_m"] <= press_r]
        pr = 0; lastp = -99
        for k in ks:
            f = frames_[k]
            if f["carrier"] is not None and f["team"] != tm and f["carrier"] in tracks and k in tracks[f["carrier"]]:
                if np.linalg.norm(tracks[tid][k] - tracks[f["carrier"]][k]) <= press_r and k - lastp > fps: pr += 1; lastp = k
        mp = pass_df[pass_df["from"] == tid] if len(pass_df) else pass_df; rc = pass_df[(pass_df["to"] == tid) & pass_df["completed"]] if len(pass_df) else pass_df
        g = np.zeros(GRID)
        for x, y in P: g[min(GRID[0] - 1, max(0, int(x / L * GRID[0]))), min(GRID[1] - 1, max(0, int(y / W * GRID[1])))] += 1
        heat[str(tid)] = (g / max(1, g.sum())).round(4).tolist(); rx = react.get(tid)
        xs = P[:, 0] if sgn == 1 else L - P[:, 0]
        players.append({"id": int(tid), "team": tm, "time_visible_s": round(len(ks) * dt, 1), "time_on_ball_s": round(len(on) * dt, 1), "touches": int(sum(1 for i in range(1, len(on)) if on[i] - on[i - 1] > int(0.4 * fps)) + (1 if on else 0)),
                        "time_under_pressure_s": round(len(pressed) * dt, 1), "passes": int(len(mp)), "passes_completed": int(mp["completed"].sum()) if len(mp) else 0,
                        "passes_forward": int((mp["kind"] == "forward").sum()) if len(mp) else 0, "passes_sideways": int((mp["kind"] == "sideways").sum()) if len(mp) else 0, "passes_backward": int((mp["kind"] == "backward").sum()) if len(mp) else 0,
                        "progressive_passes": int(mp["progressive"].sum()) if len(mp) else 0, "passes_received": int(len(rc)), "pressures_applied": pr,
                        "pass_quality": {q: int((mp["quality"] == q).sum()) if len(mp) else 0 for q in ("good", "risky_completed", "bad_lost", "execution_error")}, "better_option_count": int(mp["better_option"].sum()) if len(mp) else 0,
                        "distance_m": round(float(step.sum()), 1), "high_intensity_m": round(float(step[valid & (sp > 5.5)].sum()), 1), "top_speed_ms": round(float(np.nanmax(sp)) if valid.any() else 0, 2),
                        "avg_x_m": round(float(P[:, 0].mean()), 1), "avg_y_m": round(float(P[:, 1].mean()), 1),
                        "pct_def_third": round(100 * float((xs < L / 3).mean())), "pct_mid_third": round(100 * float(((xs >= L / 3) & (xs < 2 * L / 3)).mean())), "pct_att_third": round(100 * float((xs >= 2 * L / 3).mean())),
                        "counterpress_rate": round(rx["pressed"] / rx["losses_seen"], 2) if rx and rx["losses_seen"] else None, "losses_seen": rx["losses_seen"] if rx else 0})
    poss = [STATES[s] if s < 2 else None for s in state]; teams = []
    for tm in ("A", "B"):
        opp = "B" if tm == "A" else "A"; tp = pass_df[pass_df.team == tm] if len(pass_df) else pass_df; op = pass_df[pass_df.team == opp] if len(pass_df) else pass_df
        lens, wids, hts = [], [], []
        for k in range(n):
            pts = np.array([r[2] for r in per[k] if r[1] == tm and not r[5]])
            if len(pts) >= 5:
                xs = np.sort(pts[:, 0]); lens.append(xs[-1] - xs[0]); wids.append(pts[:, 1].max() - pts[:, 1].min()); hts.append(xs[0] if attack_right[tm] else L - xs[-1])
        my = [p for p in players if p["team"] == tm]; da = sum(p["pressures_applied"] for p in my); seqs = sum(1 for i in range(1, n) if poss[i] == tm and poss[i - 1] != tm)
        ctrl = sum(1 for p in poss if p is not None); ps = sum(1 for p in poss if p == tm) * dt
        lost = [t for t in turnovers_ if t["lost_by"] == tm]; won = [t for t in turnovers_ if t["won_by"] == tm]
        def med(v): v = [x for x in v if x is not None]; return round(float(np.median(v)), 2) if v else None
        def pct(v): v = [x for x in v if x is not None]; return round(100 * float(np.mean(v))) if v else None
        teams.append({"team": tm, "possession_pct": round(100 * sum(1 for p in poss if p == tm) / max(1, ctrl)), "possession_s": round(ps, 1), "sequences": seqs, "passes": int(len(tp)),
                      "pass_completion_pct": round(100 * tp.completed.mean()) if len(tp) else None, "passes_per_sequence": round(len(tp) / seqs, 1) if seqs else None,
                      "forward_pass_share_pct": round(100 * (tp.kind == "forward").mean()) if len(tp) else None, "progressive_passes": int(tp.progressive.sum()) if len(tp) else 0,
                      "ppda_opp_passes_per_def_action": round(len(op) / da, 1) if da else None, "block_length_median_m": med(lens), "block_width_median_m": med(wids), "def_line_height_median_m": med(hts),
                      "losses": len(lost), "recoveries": len(won), "time_to_press_median_s": med([t["time_to_press"] for t in lost]), "pressed_within_2s_pct": pct([(t["time_to_press"] is not None and t["time_to_press"] <= 2) for t in lost]),
                      "regained_within_5s_pct": pct([t["regained_within_5s"] for t in lost]), "near_at_2s_median": med([t["near_at_2s"] for t in lost]),
                      "time_to_forward_pass_median_s": med([t["time_to_forward_pass"] for t in won]), "forward_within_3s_pct": pct([(t["time_to_forward_pass"] is not None and t["time_to_forward_pass"] <= 3) for t in won]),
                      "lost_back_5s_pct": pct([t["lost_back_5s"] for t in won]), "distance_m_total_visible": round(sum(p["distance_m"] for p in my)), "pressures_applied": da})
    return {"players": players, "teams": teams, "heatmaps": heat, "grid": list(GRID)}
