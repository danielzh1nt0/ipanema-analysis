"""Contract 1.1 additions: shots/goals, pass network, field tilt + final-third entries, high turnovers, shape timeline."""
import os
import numpy as np
from .possession import STATES

def _goal_x(team, attack_right, L): return L if attack_right[team] else 0.0

def shots(state, ballm, bspeed, fps, L, W, attack_right, restarts_):
    """A shot: ball suddenly fast (>16 m/s), within 15 degrees of the goal the possessing team attacks, from within 30 m of it.
    18 Oct: the 'goal' outcome (next restart at the centre spot) never fires: possession.restarts calls a centre-spot restart
    'stoppage', and 'unknown' only when there is no ball spot. Left as is on purpose: on SFK-BP, AIK and Vallentuna no
    centre restart follows any detected shot, so a fix changes nothing today, and goals come from Veo's list."""
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
        outcome = "saved or blocked"; crossed = False
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

# N0-20c (20 Oct): a node sits where the player played and received this team's completed passes (the usual pass-network
# node), with n = passes played + received. Before, it sat at the track's whole-match average from stats.players: a track
# shorter than the players' list cut-off had no node at all (2-12 edges per repo match pointed at nothing), and a track that
# swapped team (ID slide, M2) put an A edge on a B player's average spot (4-37 edges per team per match). IPANEMA_NET_NODES=avg = old.
NET_NODES = os.environ.get("IPANEMA_NET_NODES", "passes")

def pass_network(passes_, players_, nodes=None):
    nodes = NET_NODES if nodes is None else nodes
    pos = {p["id"]: (p["avg_x_m"], p["avg_y_m"]) for p in players_}; net = {"A": {}, "B": {}}; spots = {"A": {}, "B": {}}
    for p in passes_:
        if not p["completed"] or p.get("team") not in net: continue
        key = f"{p['from']}->{p['to']}"; d = net[p["team"]].setdefault(key, {"from": p["from"], "to": p["to"], "count": 0, "forward": 0, "progressive": 0})
        d["count"] += 1; d["forward"] += p["kind"] == "forward"; d["progressive"] += bool(p.get("progressive"))
        if p.get("from_m") is not None: spots[p["team"]].setdefault(p["from"], []).append(p["from_m"])
        if p.get("to_m") is not None: spots[p["team"]].setdefault(p["to"], []).append(p["to_m"])
    out = {}
    for tm, v in net.items():
        ids = {e["from"] for e in v.values()} | {e["to"] for e in v.values()}
        if nodes == "avg": nd = {str(i): {"x": pos[i][0], "y": pos[i][1]} for i in ids if i in pos}
        else:
            nd = {}
            for i in ids:
                s = spots[tm].get(i)
                if s: m = np.mean(np.asarray(s, float), axis=0); nd[str(i)] = {"x": round(float(m[0]), 1), "y": round(float(m[1]), 1), "n": len(s)}
                elif i in pos: nd[str(i)] = {"x": pos[i][0], "y": pos[i][1], "n": 0}
        out[tm] = {"edges": sorted(v.values(), key=lambda e: -e["count"]), "nodes": nd}
    return out

def field_tilt_and_entries(state, ballm, bspeed, fps, L, attack_right, rearm_m=0.0, rearm_s=0.0, hold_s=0.0):
    """field tilt = share of a team's controlled frames with the ball in its final third; an entry = the ball, under its
    control, going from below the 2/3 line into the final third. N0-20b (20 Oct): re-arm rule - after an entry the next one
    counts only once the ball has been back at least rearm_m metres behind the line, or the team has been out of control of
    it for rearm_s seconds (ball shaking on the line / a flicker of the state counted 3 entries in 0.2 s). 0, 0 = old rule.
    N0-24c (24 Oct): hold_s - the team keeps the ball across loose frames for up to hold_s (a pass in flight is 'loose' in
    our state, so a pass into the final third only counted if the receiver's first controlled frame came <= 3 frames
    after the passer's last); a crossing across the other team's control never counts; an entry across a gap > 3 frames
    is a 'pass'. hold_s 0 = the old rule exactly."""
    out = {}
    for tm in ("A", "B"):
        sgn = 1 if attack_right[tm] else -1
        depth = lambda x: x if sgn == 1 else L - x
        def third(x): xx = depth(x); return 0 if xx < L / 3 else 1 if xx < 2 * L / 3 else 2
        ctrl = [k for k in ballm if STATES[state[k]] == tm]; final = sum(1 for k in ctrl if third(ballm[k][0]) == 2)
        entries = []; prev = None; last = None; armed = True; gap = max(3, int(round(hold_s * fps)))
        for k in sorted(ballm):
            st = STATES[state[k]]
            if st != tm:
                if hold_s > 0 and st in ("A", "B"): prev = None                                      # the other team had it in between
                continue
            th = third(ballm[k][0])
            if last is not None and k - last > max(3, rearm_s * fps, hold_s * fps): armed = True     # left control long enough
            if not armed and depth(ballm[k][0]) <= 2 * L / 3 - rearm_m: armed = True                  # back behind the line
            if prev is not None and prev[0] < 2 and th == 2 and k - prev[1] <= gap and armed:
                entries.append({"t": round(k / fps, 2), "how": "pass" if (k - prev[1] > 3 or bspeed.get(k, 0) > 9) else "carry", "y_m": round(float(ballm[k][1]), 1)})
                if rearm_m > 0 or rearm_s > 0: armed = False
            prev = (th, k); last = k
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

# N0-25c (25 Oct): the 'shape over time' timeline read length / width from the players in the picture only (>= 3), like
# the team cards before N0-22e - on Metrica pro data seen through our app clips' views (tools/n0_25c_timeline.py) each
# second read 0.60-0.67x the truth (median error 10-12 m length, 12-14 m width). With per (and W) given, length / width
# now come from analytics.held_points (last seen spot kept SHAPE_HOLD_S, >= 5 points), the same rule as the Shape tab's
# medians: 1.00x / 0.93-0.97x, error 2.9-3.7 m / 3.7-4.6 m, better on all 8 runs; seconds with a value 87-93% -> 73-82%
# (seconds with 3-4 seen players get null). Line height stays from the seen players (holding makes it worse, N0-22e).
# IPANEMA_TIMELINE_HOLD=0 = old.
TIMELINE_HOLD = os.environ.get("IPANEMA_TIMELINE_HOLD", "1") != "0"

def shape_timeline(shapes_, fps, attack_right, L, per=None, W=None, hold=None, min_pts=5):
    hold = TIMELINE_HOLD if hold is None else hold
    held = None
    if hold and per is not None:
        from . import analytics as AN
        if AN.SHAPE_HOLD_S: held = AN.held_points(per, fps, L=L, W=W)
    tl = {"A": [], "B": []}
    for k in range(0, len(shapes_), int(round(fps))):
        for tm in ("A", "B"):
            sh = shapes_[k].get(tm)
            if sh: e = {"t": round(k / fps, 1), "length": sh["length"], "width": sh["width"], "line_height": round((sh["x_min"] if attack_right[tm] else L - sh["x_max"]), 1), "n": sh["n"]}
            else: e = {"t": round(k / fps, 1), "length": None, "width": None, "line_height": None, "n": 0}
            if held is not None:
                hp = held[k][tm] if k < len(held) else None
                if hp is not None and len(hp) >= min_pts:
                    e["length"] = round(float(np.ptp(hp[:, 0])), 1); e["width"] = round(float(np.ptp(hp[:, 1])), 1)
                else: e["length"] = e["width"] = None
                e["n_held"] = 0 if hp is None else int(len(hp))
            tl[tm].append(e)
    return tl

RUNS_SMOOTH = os.environ.get("IPANEMA_RUNS_SMOOTH", "1") == "1"   # N0-25a: sprint speed from smoothed positions (0 = old raw frame speed)


def runs(per, fps, state, attack_right, speed_min=5.5, min_s=0.8, smooth=None, win_s=0.8, gap_join_s=0.0, jump_m=2.0, merge_s=0.0):
    """sprint segments per player: start/end in metres, speed, whether the team had the ball.
    N0-25a (25 Oct): the old rule read the speed from RAW frame-to-frame feet positions; our feet jitter ~0.3 m a frame,
    so on the app clips 30% of frame speeds are > 5.5 m/s and 14% > 10 m/s (the 'id jump' cap), every sprint was cut to
    pieces and the app's Runs layer was nearly empty (SFK-BP clip 2 runs in 5 min, AIK 0). Now (smooth=True, default)
    the speed comes from positions smoothed over win_s within each seen run (analytics.smooth_runs, as the distance
    figures since N0-22c); a sprint = consecutive seen rows (steps <= 3 frames) at speed_min..10 m/s for >= min_s; its
    from/to/distance are the smoothed spots. Metrica pro tracking cut like our tracks: results/qa/n0-25a.
    A raw step of more than jump_m in <= 3 frames (the track jumped to another person) breaks a run; two runs of one
    player less than merge_s apart are one run when it still averages speed_min (lab only, 0 = off).
    Rows filled by tracking.fill_gaps (straight line, marked 'filled'; older inputs: an exactly straight constant step)
    are not seen and break a run. smooth=False (IPANEMA_RUNS_SMOOTH=0) = old."""
    from collections import defaultdict
    smooth = RUNS_SMOOTH if smooth is None else smooth
    tr = defaultdict(dict); team = {}
    for k in per:
        for r in per[k]:
            if smooth and len(r) > 6 and r[6] == "filled": continue          # N0-25a: a gap filled by a straight line (fill_gaps, up to 8 m/s) is not a seen sprint
            tr[r[0]][k] = r[2]; team[r[0]] = r[1]
    out = []
    if smooth:
        from . import analytics as AN
        for tid, pts in tr.items():
            ks = np.array(sorted(pts))
            if len(ks) < 2: continue
            P = np.array([pts[k] for k in ks], float); win = max(1, int(win_s * fps))
            lin = _straight(ks, P)                                           # unmarked straight-line fills (older saved inputs)
            if lin.any(): ks, P = ks[~lin], P[~lin]
            if len(ks) < 2: continue
            g = np.diff(ks); jump = np.r_[False, (g <= 3) & (np.linalg.norm(np.diff(P, axis=0), axis=1) > jump_m)] if jump_m else np.zeros(len(ks), bool)
            Ps = AN.smooth_runs(ks + 10 * np.cumsum(jump), P, win, runs=True)   # a raw jump (another person) splits the smoothing like a gap
            sp = _speeds(ks, Ps, fps, gap_join_s); sp[jump] = np.nan            # = analytics.track_motion's speeds (0.8 s window)
            on = (sp >= speed_min) & (sp <= 10)                              # NaN (gap / jump / > 10 m/s) is off
            sm = {int(k): Ps[i] for i, k in enumerate(ks)}; i = 1; segs = []
            while i < len(ks):
                if not on[i]: i += 1; continue
                j = i
                while j + 1 < len(ks) and on[j + 1]: j += 1
                seg = [int(ks[i - 1]), int(ks[j])]                           # the step into row i starts at row i-1
                if segs and merge_s and seg[0] - segs[-1][1] <= merge_s * fps and not jump[(ks > segs[-1][1]) & (ks <= seg[0])].any(): segs[-1][1] = seg[1]
                else: segs.append(seg)
                i = j + 1
            rs = [_run(tid, team[tid], s_, sm, fps, state) for s_ in segs if (s_[1] - s_[0]) / fps >= min_s]
            out += [r for r in rs if not merge_s or (r["speed_ms"] or 0) >= speed_min]   # a merged run must still average sprint speed
        return out
    for tid, pts in tr.items():
        ks = sorted(pts); seg = None
        for a, b in zip(ks, ks[1:]):
            if b - a > 3:                                                     # track gap: close the sprint so far (18 Oct: it was thrown away)
                if seg and (seg[1] - seg[0]) / fps >= min_s: out.append(_run(tid, team[tid], seg, pts, fps, state))
                seg = None; continue
            v = np.linalg.norm(pts[b] - pts[a]) * fps / (b - a)
            if 10 >= v >= speed_min:
                if seg is None: seg = [a, b]
                else: seg[1] = b
            else:
                if seg and (seg[1] - seg[0]) / fps >= min_s: out.append(_run(tid, team[tid], seg, pts, fps, state))
                seg = None
        if seg and (seg[1] - seg[0]) / fps >= min_s: out.append(_run(tid, team[tid], seg, pts, fps, state))
    return out

def _straight(ks, P, tol=1e-6):
    """rows inside a stretch of >= 3 consecutive frames moving in an exactly straight line at an exactly constant step
    (second difference 0, step > 1 cm): a straight-line gap fill, never a measurement (feet jitter ~0.3 m)"""
    out = np.zeros(len(ks), bool)
    if len(ks) < 3: return out
    d1 = np.diff(P, axis=0); d2 = np.linalg.norm(np.diff(d1, axis=0), axis=1)
    ok = (np.diff(ks)[1:] == 1) & (np.diff(ks)[:-1] == 1) & (d2 < tol) & (np.linalg.norm(d1[1:], axis=1) > 0.01)
    out[1:-1] |= ok
    return out


def _speeds(ks, Ps, fps, gap_join_s=0.0, gap_max_ms=7.0):
    """per-row speed from already smoothed positions: steps of <= 3 frames, NaN otherwise (as analytics.track_motion).
    gap_join_s > 0: a longer gap up to that many seconds gets the straight-line speed across it when that is at most
    gap_max_ms (faster = another person, N0-22c), so a sprint can run through a short hide."""
    sp = np.full(len(ks), np.nan)
    for i in range(1, len(ks)):
        g = int(ks[i] - ks[i - 1]); v = float(np.linalg.norm(Ps[i] - Ps[i - 1])) * fps / g
        if g <= 3: sp[i] = v
        elif g <= gap_join_s * fps and v <= gap_max_ms: sp[i] = v
    return sp


def _run(tid, tm, seg, pts, fps, state):
    a, b = seg; d = float(np.linalg.norm(pts[b] - pts[a])); dur = (b - a) / fps
    poss = STATES[state[a]] if state[a] < 2 else None
    return {"player": int(tid), "team": tm, "t": round(a / fps, 2), "t_end": round(b / fps, 2), "from_m": [round(float(pts[a][0]), 1), round(float(pts[a][1]), 1)],
            "to_m": [round(float(pts[b][0]), 1), round(float(pts[b][1]), 1)], "distance_m": round(d, 1), "speed_ms": round(d / dur, 1) if dur else None,
            "with_ball": poss == tm if poss else None}

# N0-26a (26 Oct): the pressing-over-time chart and pressure map (pressure_points) checked on Metrica pro data - with the
# raw nearest-player carrier every ball flying or bouncing past a player with an opponent near makes a point: 1.3-1.6x the
# truth with a perfect ball, 1.9x (streak) / 2.3x (flicker) with our noise models, 5-min counts off 48-130%; where (thirds)
# within 3-5 pts. Counting only frames where the spell state gives the carrier's team the ball fixes that on Metrica
# (0.91x / 0.64x / 1.15x, 5-min error 19 / 38 / 25%) BUT on our two app clips, graded blind by eye, the points it drops are
# mostly real (18 of 24 a player on the ball with an opponent on him - duels where the lagging state already names the
# other team), the kept ones 7 of 12. Not better on our footage -> option only: IPANEMA_PRESS_POINTS=spell.
PRESS_POINTS_SPELL = os.environ.get("IPANEMA_PRESS_POINTS", "raw") == "spell"

def pressure_points(per, frames_, fps, press_r=2.0, state=None):
    """where pressures happened: carrier position each time an opponent is within press_r (one point per second max per
    carrier); with state (the spell state), only while that state gives the carrier's team the ball"""
    out = []; last = {}
    for k, f in enumerate(frames_):
        if f["carrier"] is None or f["pressure_m"] is None or f["pressure_m"] > press_r: continue
        if state is not None and (k >= len(state) or state[k] > 1 or STATES[state[k]] != f["team"]): continue
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
        tot = rows["A"]["final_third_frames"] + rows["B"]["final_third_frames"]; ctl = rows["A"]["control_frames"] + rows["B"]["control_frames"]
        pa = round(rows["A"]["control_frames"] / ctl, 2) if ctl else (0.0 if TILT_EMPTY_ZERO else None)
        out.append({"t": round(s / fps, 1), "t_end": round(min(n, s + w) / fps, 1), "tilt_A": round(rows["A"]["final_third_frames"] / tot, 2) if tot else None,
                    "possession_A": pa, "control_s": round(ctl / fps, 1)})
    return out

# N0-22d (22 Oct): a window in which neither team had the ball (before kick-off, half-time, after the whistle - since N0-21d
# those frames are nobody's - or a stretch with no ball) said possession_A 0.0, i.e. the opponent 100% on the app's
# 'possession over time' chart. Now None (the app skips null windows, like tilt_A), and control_s = seconds of control the
# share rests on, so a reader can weight or skip thin windows. IPANEMA_TILT_EMPTY=zero = the old 0.0.
TILT_EMPTY_ZERO = os.environ.get("IPANEMA_TILT_EMPTY", "none") == "zero"

# N0-20b (20 Oct): entries need the ball 3 m back behind the line or 1 s out of control before the next one counts.
# SFK-BP clip, by eye: in the 5 stretches it changes, old 17 entries -> 7, real ~0-2 (ball shaking on the line / the
# pick jumping to a touchline spot on the line). IPANEMA_ENTRY_REARM=0 = the old rule.
ENTRY_REARM = (3.0, 1.0) if os.environ.get("IPANEMA_ENTRY_REARM", "1") != "0" else (0.0, 0.0)

# N0-24c (24 Oct): final-third entries checked on Metrica pro data - the old rule found 42% of the real entries with a
# perfect ball and only 19% of those made by a pass (the ball in flight is 'loose', so the crossing was never seen under
# control). Holding control across loose frames up to 3 s: found 42 -> 72%, real 91 -> 96%, count 0.46 -> 0.75x the
# truth (perfect ball); flicker noise like our clips found 51 -> 65%, real 54 -> 62%, count 0.94 -> 1.04x; streak noise
# 14 -> 31%, 62 -> 78%. Better on every row. IPANEMA_ENTRY_HOLD_S=0 = the old rule.
ENTRY_HOLD_S = float(os.environ.get("IPANEMA_ENTRY_HOLD_S", "3"))

# N0-24b (24 Oct): high turnovers checked on Metrica pro data - with a clean ball our count is ~1/3 of the truth (14 vs
# 43 balls won and kept in the final third, 4 halves; ~57% of ours real), with our clips' ball noise mostly not real
# (streak 1 of 22, flicker 10 of 59). IPANEMA_HIGH_TOV=withhold writes the counts as null (app shows a dash); default
# unchanged until Daniel says. The zone of a ball won we do find is right 93% of the time; the list stays as is.
HIGH_TOV_WITHHOLD = os.environ.get("IPANEMA_HIGH_TOV", "") == "withhold"

def compute(state, ballm, bspeed, fps, L, W, attack_right, restarts_, passes_, players_, turnovers_, shapes_, per=None, frames_=None, cstate=None):
    sh = shots(state, ballm, bspeed, fps, L, W, attack_right, restarts_); ht = high_turnovers(turnovers_, ballm, L, attack_right, fps)
    return {"contract": "1.1", "shots": sh, "goals": [s for s in sh if s["goal"]], "pass_network": pass_network(passes_, players_),
            "field": field_tilt_and_entries(state, ballm, bspeed, fps, L, attack_right, *ENTRY_REARM, hold_s=ENTRY_HOLD_S), "high_turnovers": ht,
            "high_turnover_counts": {tm: (None if HIGH_TOV_WITHHOLD else sum(1 for h in ht if h["team"] == tm and h["zone"] == "high")) for tm in ("A", "B")}, "shape_timeline": shape_timeline(shapes_, fps, attack_right, L, per=per, W=W),
            "runs": runs(per, fps, state, attack_right) if per is not None else [], "pressure_points": pressure_points(per, frames_, fps, state=(cstate if PRESS_POINTS_SPELL else None)) if frames_ is not None else [],
            "tilt_windows": tilt_windows(state, ballm, fps, L, attack_right)}

def extra_events(metrics):
    ev = []
    for s in metrics["shots"]:
        dist = f"{s['distance_m']} m · " if s.get("distance_m") is not None else ""
        ev.append({"id": f"shot_{s['t']}", "t": s["t"], "type": "goal" if s["goal"] else "shot", "team": s["team"], "title": ("GOAL" if s["goal"] else "Shot") + (f" · {s['team']}" if s.get("team") else ""), "subtitle": f"{dist}{s['outcome']}" + (" · from Veo" if s.get("source") == "veo" else ""), "payload": s})
    for h in metrics["high_turnovers"]:
        if h["zone"] == "high": ev.append({"id": f"hto_{h['t']}", "t": h["t"], "type": "high_turnover", "team": h["team"], "title": f"High turnover · {h['team']}", "subtitle": "won the ball in the final third", "payload": h})
    return ev
