"""Shots and goals from Veo's own detections (highlight clip times, to the second), made richer by our tracking.

Why: Veo already detects shots and goals for every match it films; measured against Veo, our own shot detector found 1 of
25 shots (21 Sep). Rebuilding shot detection badly is pointless; our value is the tactical layer on top. So when Veo's
events exist they become the shot/goal events, and our detector keeps being scored against them.

Veo is not perfect either (SFK-BP ended 2-1 but Veo tagged 4 goals: one at 0:00:28), so goals are checked: a goal in the
first minute of a period with no kick-off restart after it is rejected.
"""
import os, re, numpy as np

def load(path):
    """reference/veo_highlights_<match>.txt -> sorted [(t_s, 'shot'|'goal')]"""
    out = []
    for line in open(path):
        m = re.match(r"\s*(\d+)\s+(shot|goal)\b", line)
        if m: out.append((int(m.group(1)), m.group(2)))
    return sorted(out)

def _where(t, fps, ballm, per, L, window_s=(-2.0, 1.0), players_s=(-20.0, 0.0)):
    """position of the action at time t: the ball if we have it around t, else the players' centre just before"""
    k0 = int(t * fps)
    bs = [ballm[k] for k in range(k0 + int(window_s[0] * fps), k0 + int(window_s[1] * fps)) if k in ballm]
    if bs: return np.median(np.array(bs), 0), "ball"
    for k in range(k0, k0 + int(players_s[0] * fps), -1):          # most recent frame with players (goal areas may be 'unknown')
        rows = per.get(k) or []
        if len(rows) >= 4: return np.mean(np.array([r[2] for r in rows]), 0), "players"
    return None, None

def _attacking_end_team(x, L, attack_right):
    """the team attacking the end the action is at"""
    right_end = x > L / 2
    for team, right in attack_right.items():
        if bool(right) == right_end: return team
    return None

def build(highlights, fps, ballm, per, L, W, attack_right, restarts_, periods=None, log=print):
    """-> (shots, rejected): shot dicts shaped like metrics.shots (goal=True for goals), source 'veo'"""
    starts = [p["t_start"] for p in (periods or [])] or [0.0]
    shots_t = sorted({t for t, k in highlights if k == "shot"}); goals_t = sorted({t for t, k in highlights if k == "goal"})
    rejected = []
    kickoffs = [r["t"] for r in restarts_ if r.get("x_m") is not None and abs(r["x_m"] - L / 2) < 6 and abs(r["y_m"] - W / 2) < 6]
    good_goals = []
    for g in goals_t:
        early = any(0 <= g - s < 60 for s in starts)
        restart_after = any(g < k < g + 150 for k in kickoffs)
        if early and not restart_after: rejected.append({"t": g, "why": "goal tagged within a minute of kick-off with no kick-off restart after it"}); continue
        good_goals.append(g)
    events = sorted({(t, False) for t in shots_t if not any(abs(t - g) <= 2 for g in good_goals)} | {(g, True) for g in good_goals})
    out = []
    for t, is_goal in events:
        p, src = _where(t, fps, ballm, per, L)
        team = _attacking_end_team(p[0], L, attack_right) if p is not None else None
        gx = L if (team and attack_right.get(team)) else 0.0
        out.append({"t": float(t), "team": team, "x_m": round(float(p[0]), 1) if p is not None else None, "y_m": round(float(p[1]), 1) if p is not None else None,
                    "distance_m": round(float(np.hypot(p[0] - gx, p[1] - W / 2)), 1) if (p is not None and src == "ball") else None,
                    "speed_ms": None, "outcome": "goal" if is_goal else "on target", "goal": is_goal, "source": "veo", "located_by": src})
    log(f"veo: {len(out)} shots incl. {sum(s['goal'] for s in out)} goals imported; {len(rejected)} goal tag(s) rejected; "
        f"team from the ball for {sum(1 for s in out if s['located_by'] == 'ball')}, from the players for {sum(1 for s in out if s['located_by'] == 'players')}, unknown for {sum(1 for s in out if s['team'] is None)}")
    return out, rejected

def score_detector(detected, veo_shots, tol_s=10.0):
    """how our own shot detector does against Veo's events"""
    vt = [s["t"] for s in veo_shots]; dt = [s["t"] for s in detected]
    found = sum(1 for v in vt if any(abs(d - v) <= tol_s for d in dt)); real = sum(1 for d in dt if any(abs(d - v) <= tol_s for v in vt))
    return {"veo_shots": len(vt), "found": found, "ours": len(dt), "real": real}
