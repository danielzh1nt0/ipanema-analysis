"""Contract 1.1 additions: shots/goals, pass network, field tilt + final-third entries, high turnovers, shape timeline."""
import numpy as np
from .possession import STATES

def _goal_x(team, attack_right, L): return L if attack_right[team] else 0.0

def shots(state, ballm, bspeed, fps, L, W, attack_right, restarts_):
    """A shot: ball suddenly fast (>12 m/s), heading toward the goal the possessing team attacks, from within 35 m of it."""
    n = len(state); out = []; last_t = -99
    for k in sorted(ballm):
        if k - 1 not in ballm or k + 1 not in ballm: continue
        s = bspeed.get(k, 0.0)
        if s < 16.0 or (k - last_t) < 2 * fps: continue
        team = next((STATES[state[j]] for j in range(k, max(-1, k - int(0.6 * fps)), -1) if state[j] < 2), None)
        if team is None: continue
        gx = _goal_x(team, attack_right, L); goal = np.array([gx, W / 2]); p = ballm[k]; v = ballm[k + 1] - ballm[k - 1]
        if np.linalg.norm(v) < 1e-6 or np.linalg.norm(goal - p) > 30: continue
        cosang = float(v @ (goal - p) / (np.linalg.norm(v) * np.linalg.norm(goal - p)))
        if cosang < np.cos(np.radians(15)): continue
        # outcome over the next 3 s
        outcome = "blocked or saved"; crossed = False
        for j in range(k + 1, min(n, k + int(3 * fps))):
            if j in ballm:
                bx, by = ballm[j]
                if ((bx >= L - 0.5) if gx == L else (bx <= 0.5)) and abs(by - W / 2) < 3.66 + 1.0: crossed = True; break
        ended = crossed or any(STATES[state[j]] == "dead" for j in range(k, min(n, k + int(3 * fps))))
        if not ended: continue
        nxt = next((r for r in restarts_ if r["t"] > k / fps and r["t"] < k / fps + 20), None)
        if nxt and nxt["kind"] == "unknown" and nxt["x_m"] is not None and abs(nxt["x_m"] - L / 2) < 6 and abs(nxt["y_m"] - W / 2) < 6: outcome = "goal"
        elif crossed: outcome = "on target"
        elif nxt and nxt["kind"] in ("goal kick", "corner") and nxt["t"] - k / fps < 8: outcome = "off target" if nxt["kind"] == "goal kick" else "saved or blocked"
        out.append({"t": round(k / fps, 2), "team": team, "x_m": round(float(p[0]), 1), "y_m": round(float(p[1]), 1), "distance_m": round(float(np.linalg.norm(goal - p)), 1),
                    "speed_ms": round(s, 1), "outcome": outcome, "goal": outcome == "goal"}); last_t = k
    return out

def pass_network(passes_, players_):
    pos = {p["id"]: (p["avg_x_m"], p["avg_y_m"]) for p in players_}; net = {"A": {}, "B": {}}
    for p in passes_:
        if not p["completed"]: continue
        key = f"{p['from']}->{p['to']}"; d = net[p["team"]].setdefault(key, {"from": p["from"], "to": p["to"], "count": 0, "forward": 0, "progressive": 0})
        d["count"] += 1; d["forward"] += p["kind"] == "forward"; d["progressive"] += bool(p.get("progressive"))
    return {tm: {"edges": sorted(v.values(), key=lambda e: -e["count"]), "nodes": {str(i): {"x": pos[i][0], "y": pos[i][1]} for i in {e["from"] for e in v.values()} | {e["to"] for e in v.values()} if i in pos}} for tm, v in net.items()}

def field_tilt_and_entries(state, ballm, bspeed, fps, L, attack_right):
    out = {}
    for tm in ("A", "B"):
        sgn = 1 if attack_right[tm] else -1
        def third(x): xx = x if sgn == 1 else L - x; return 0 if xx < L / 3 else 1 if xx < 2 * L / 3 else 2
        ctrl = [k for k in ballm if STATES[state[k]] == tm]; final = sum(1 for k in ctrl if third(ballm[k][0]) == 2)
        entries = []; prev = None
        for k in sorted(ctrl):
            th = third(ballm[k][0])
            if prev is not None and prev[0] < 2 and th == 2 and k - prev[1] <= 3:
                entries.append({"t": round(k / fps, 2), "how": "pass" if bspeed.get(k, 0) > 9 else "carry", "y_m": round(float(ballm[k][1]), 1)})
            prev = (th, k)
        out[tm] = {"field_tilt_pct": round(100 * final / len(ctrl)) if ctrl else None, "final_third_entries": entries, "entries_count": len(entries),
                   "entries_by_pass": sum(e["how"] == "pass" for e in entries), "entries_by_carry": sum(e["how"] == "carry" for e in entries)}
    return out

def high_turnovers(turnovers_, ballm, L, attack_right, fps):
    out = []
    for t in turnovers_:
        b = ballm.get(t["frame"])
        if b is None: continue
        tm = t["won_by"]; x = b[0] if attack_right[tm] else L - b[0]
        zone = "high" if x > 2 * L / 3 else "mid" if x > L / 3 else "low"
        out.append({"t": t["t"], "team": tm, "zone": zone, "x_m": round(float(b[0]), 1), "y_m": round(float(b[1]), 1)})
    return out

def shape_timeline(shapes_, fps, attack_right, L):
    tl = {"A": [], "B": []}
    for k in range(0, len(shapes_), int(round(fps))):
        for tm in ("A", "B"):
            sh = shapes_[k].get(tm)
            if sh: tl[tm].append({"t": round(k / fps, 1), "length": sh["length"], "width": sh["width"], "line_height": round((sh["x_min"] if attack_right[tm] else L - sh["x_max"]), 1), "n": sh["n"]})
            else: tl[tm].append({"t": round(k / fps, 1), "length": None, "width": None, "line_height": None, "n": 0})
    return tl

def runs(per, fps, state, attack_right, speed_min=5.5, min_s=0.8):
    """sprint segments per player: start/end in metres, speed, whether the team had the ball"""
    from collections import defaultdict
    tr = defaultdict(dict); team = {}
    for k in per:
        for r in per[k]: tr[r[0]][k] = r[2]; team[r[0]] = r[1]
    out = []
    for tid, pts in tr.items():
        ks = sorted(pts); seg = None
        for a, b in zip(ks, ks[1:]):
            if b - a > 3: seg = None; continue
            v = np.linalg.norm(pts[b] - pts[a]) * fps / (b - a)
            if 10 >= v >= speed_min:
                if seg is None: seg = [a, b]
                else: seg[1] = b
            else:
                if seg and (seg[1] - seg[0]) / fps >= min_s: out.append(_run(tid, team[tid], seg, pts, fps, state))
                seg = None
        if seg and (seg[1] - seg[0]) / fps >= min_s: out.append(_run(tid, team[tid], seg, pts, fps, state))
    return out

def _run(tid, tm, seg, pts, fps, state):
    a, b = seg; d = float(np.linalg.norm(pts[b] - pts[a])); dur = (b - a) / fps
    poss = STATES[state[a]] if state[a] < 2 else None
    return {"player": int(tid), "team": tm, "t": round(a / fps, 2), "t_end": round(b / fps, 2), "from_m": [round(float(pts[a][0]), 1), round(float(pts[a][1]), 1)],
            "to_m": [round(float(pts[b][0]), 1), round(float(pts[b][1]), 1)], "distance_m": round(d, 1), "speed_ms": round(d / dur, 1) if dur else None,
            "with_ball": poss == tm if poss else None}

def pressure_points(per, frames_, fps, press_r=2.0):
    """where pressures happened: carrier position each time an opponent is within press_r (one point per second max per carrier)"""
    out = []; last = {}
    for k, f in enumerate(frames_):
        if f["carrier"] is None or f["pressure_m"] is None or f["pressure_m"] > press_r: continue
        key = f["carrier"]
        if k - last.get(key, -99) < fps: continue
        last[key] = k; opp = "B" if f["team"] == "A" else "A"
        out.append({"t": round(k / fps, 2), "pressing_team": opp, "carrier_team": f["team"], "x_m": round(f["pos"][0], 1), "y_m": round(f["pos"][1], 1)})
    return out

def tilt_windows(state, ballm, fps, L, attack_right, window_s=15):
    """field tilt per time window: share of controlled frames in each team's final third, for the momentum strip"""
    n = len(state); w = int(window_s * fps); out = []
    for s in range(0, n, w):
        rows = {}
        for tm in ("A", "B"):
            sgn = 1 if attack_right[tm] else -1
            ctrl = [k for k in range(s, min(n, s + w)) if k in ballm and STATES[state[k]] == tm]
            fin = sum(1 for k in ctrl if ((ballm[k][0] if sgn == 1 else L - ballm[k][0]) > 2 * L / 3))
            rows[tm] = {"control_frames": len(ctrl), "final_third_frames": fin}
        tot = rows["A"]["final_third_frames"] + rows["B"]["final_third_frames"]
        out.append({"t": round(s / fps, 1), "t_end": round(min(n, s + w) / fps, 1), "tilt_A": round(rows["A"]["final_third_frames"] / tot, 2) if tot else None,
                    "possession_A": round(rows["A"]["control_frames"] / max(1, rows["A"]["control_frames"] + rows["B"]["control_frames"]), 2)})
    return out

def compute(state, ballm, bspeed, fps, L, W, attack_right, restarts_, passes_, players_, turnovers_, shapes_, per=None, frames_=None):
    sh = shots(state, ballm, bspeed, fps, L, W, attack_right, restarts_); ht = high_turnovers(turnovers_, ballm, L, attack_right, fps)
    return {"contract": "1.1", "shots": sh, "goals": [s for s in sh if s["goal"]], "pass_network": pass_network(passes_, players_),
            "field": field_tilt_and_entries(state, ballm, bspeed, fps, L, attack_right), "high_turnovers": ht,
            "high_turnover_counts": {tm: sum(1 for h in ht if h["team"] == tm and h["zone"] == "high") for tm in ("A", "B")}, "shape_timeline": shape_timeline(shapes_, fps, attack_right, L),
            "runs": runs(per, fps, state, attack_right) if per is not None else [], "pressure_points": pressure_points(per, frames_, fps) if frames_ is not None else [],
            "tilt_windows": tilt_windows(state, ballm, fps, L, attack_right)}

def extra_events(metrics):
    ev = []
    for s in metrics["shots"]:
        ev.append({"id": f"shot_{s['t']}", "t": s["t"], "type": "goal" if s["goal"] else "shot", "team": s["team"], "title": ("GOAL" if s["goal"] else "Shot") + f" · {s['team']}", "subtitle": f"{s['distance_m']} m · {s['outcome']}", "payload": s})
    for h in metrics["high_turnovers"]:
        if h["zone"] == "high": ev.append({"id": f"hto_{h['t']}", "t": h["t"], "type": "high_turnover", "team": h["team"], "title": f"High turnover · {h['team']}", "subtitle": "won the ball in the final third", "payload": h})
    return ev
