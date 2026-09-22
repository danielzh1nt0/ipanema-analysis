"""Match periods from the footage itself: when each half kicks off and ends.

1. Players on the pitch per second (smoothed over 30 s): play = many players; half-time = the longest stretch with few
   players in the middle part of the recording; after the final whistle the pitch empties again.
2. Kick-off by formation: near the start of each half, the first moment where one team is entirely in one half and the
   other team in the other half for a couple of seconds (warm-ups don't look like that, so they aren't counted as play).
3. Optional cross-check from Veo: a goal Veo labels at minute m in the second half, whose clip is at video time t, implies
   the second half kicked off at about t - (m - 46) * 60 s (Veo minutes are whole minutes, so within about a minute).
"""
import numpy as np

def per_second(per, fps, L, W, trusted=None):
    """-> per-second arrays: players on the pitch, A and B on each side of halfway, frames used"""
    n = max(per) + 1 if per else 0; secs = int(n / fps) + 1
    on = np.zeros(secs); aL = np.zeros(secs); aR = np.zeros(secs); bL = np.zeros(secs); bR = np.zeros(secs); used = np.zeros(secs)
    for k, rows in per.items():
        if trusted is not None and not trusted[k]: continue
        s = int(k / fps); used[s] += 1
        for r in rows:
            x, y = float(r[2][0]), float(r[2][1])
            if not (0 <= x <= L and 0 <= y <= W): continue
            on[s] += 1; left = x < L / 2
            if r[1] == "A": aL[s] += left; aR[s] += not left
            else: bL[s] += left; bR[s] += not left
    d = np.maximum(used, 1)
    return {"on": on / d, "aL": aL / d, "aR": aR / d, "bL": bL / d, "bR": bR / d, "used": used}

def _runs(mask):
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None: start = i
        if not v and start is not None: out.append((start, i)); start = None
    return out

def _roll_median(x, w):
    h = w // 2; return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])

def kickoff_formation(ps, s, min_each=5, hold_s=2):
    """True if at second s (and the next hold_s seconds) one team is all on one side of halfway and the other all on the other"""
    for t in range(s, min(s + hold_s, len(ps["on"]))):
        a_left = ps["aL"][t] >= min_each and ps["aR"][t] < 0.5 and ps["bR"][t] >= min_each and ps["bL"][t] < 0.5
        a_right = ps["aR"][t] >= min_each and ps["aL"][t] < 0.5 and ps["bL"][t] >= min_each and ps["bR"][t] < 0.5
        if not (a_left or a_right): return False
    return True

def detect(ps, window_s=180, min_halftime_s=240):
    """-> {"periods": [(start_s, end_s), (start_s, end_s)], "evidence": {...}} or None"""
    on = _roll_median(ps["on"], 31); secs = len(on)
    if secs < 1200: return None
    level = np.percentile(on, 90); play = on >= 0.6 * level
    lo, hi = int(0.3 * secs), int(0.75 * secs)
    gaps = [(a, b) for a, b in _runs(~play) if b - a >= min_halftime_s and a < hi and b > lo]
    if not gaps: return None
    ht = max(gaps, key=lambda g: g[1] - g[0])
    first_play = next((a for a, b in _runs(play) if b - a >= 120), 0)
    tail = [(a, b) for a, b in _runs(~play) if a > ht[1] and b >= secs - 1 and b - a >= 120]
    end2 = tail[0][0] if tail else secs
    def kick(around):
        for s in range(max(0, around - window_s), min(secs, around + window_s)):
            if kickoff_formation(ps, s): return s, True
        return around, False
    k1, f1 = kick(first_play); k2, f2 = kick(ht[1])
    return {"periods": [(float(k1), float(ht[0])), (float(k2), float(end2))],
            "evidence": {"players_on_pitch_typical": round(float(level), 1), "halftime": (int(ht[0]), int(ht[1])),
                         "kickoff_1_by_formation": f1, "kickoff_2_by_formation": f2}}

def veo_second_half_kickoff(goal_times_s, goal_minutes, first_half_end_s):
    """from Veo: goals at video times + their match minutes (second half only) -> implied second-half kick-off range (s)"""
    est = []
    for t, m in zip(sorted(g for g in goal_times_s if g > first_half_end_s), sorted(m for m in goal_minutes if m >= 46)):
        est.append((t - (m - 45) * 60, t - (m - 46) * 60))     # minute m = match time (m-1):00-(m-1):59, i.e. (m-46) to (m-45) min after the 2nd-half kick-off
    if not est: return None
    return max(a for a, b in est), min(b for a, b in est)
