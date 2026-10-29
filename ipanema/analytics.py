"""Lanes, passes (quality + best option), shapes, player and team stats — all from tracking + possession."""
import numpy as np, pandas as pd, cv2
from collections import defaultdict
from .vote import majority
import os
RUN_SMOOTH = os.environ.get("IPANEMA_RUN_SMOOTH", "1") == "1"   # N0-22c: smooth a track's positions within each seen run, never across a gap (0 = old)


def smooth_runs(ks, P, win, max_gap=3, runs=None):
    """N0-22c (22 Oct): centred rolling mean of a track's positions P (rows at frames ks), computed separately for each
    run of rows whose frames are at most max_gap apart. The old single rolling mean over all rows mixed positions from
    before and after a gap (player out of view, track stitched across up to 2.5 s): a player standing still on both sides
    of a 2 s gap 'ran' 4.8 m, all of it at sprint speed (6.25 m/s). runs=False = the old mean over all rows."""
    P = np.asarray(P, float); runs = RUN_SMOOTH if runs is None else runs
    if not runs or len(ks) < 2: return pd.DataFrame(P).rolling(max(1, win), center=True, min_periods=1).mean().values
    cut = np.flatnonzero(np.diff(np.asarray(ks)) > max_gap) + 1; out = np.empty_like(P)
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(ks)]):
        out[a:b] = pd.DataFrame(P[a:b]).rolling(max(1, win), center=True, min_periods=1).mean().values
    return out

GAP_LINE_S = float(os.environ.get("IPANEMA_GAP_LINE_S", "2.5"))   # N0-22c: a track gap up to this long counts as a straight line (0 = not counted)
GAP_MAX_MS = float(os.environ.get("IPANEMA_GAP_MAX_MS", "7.0"))    # N0-22c: ... when that straight line is at most this fast (about a U15 sprint; faster = another person)


def track_motion(ks, P, fps, runs=None, gap_line_s=None, gap_max_ms=None, hi_ms=5.5, max_ms=10.0):
    """per-row speed (m/s, NaN = not measured), distance and high-intensity distance of one track (rows at frames ks).
    Speeds come from positions smoothed over 0.8 s (per seen run, N0-22c), steps of <= 3 frames, 0.5-10 m/s (below 0.5 = 0).
    N0-22c (22 Oct): a gap of more than 3 frames (player hidden / track stitched, median 0.33 s on our clips) also adds
    the straight line across it when the gap is at most gap_line_s long and that line at most gap_max_ms fast (on our
    clips a fifth of gap jumps are > 10 m/s: another person, not a run); it is high-intensity when that speed is > hi_ms.
    Gap speeds never count as the top speed. Metrica pro tracking cut like our tracks, with id swaps at our rate
    (tools/n0_22c_metrica.py, results/qa/n0-22c): distance 81-85% -> 89-93% of the truth, high-intensity 139-237% ->
    111-163%, median player error 21-24% -> 14-18%. runs=False, gap_line_s=0 (IPANEMA_RUN_SMOOTH=0 IPANEMA_GAP_LINE_S=0) = old."""
    ks = np.asarray(ks); dt = 1 / fps; gap_line_s = GAP_LINE_S if gap_line_s is None else gap_line_s
    gap_max_ms = GAP_MAX_MS if gap_max_ms is None else gap_max_ms
    if runs is None: runs = RUN_SMOOTH
    Ps = smooth_runs(ks, P, int(0.8 * fps), runs=runs)
    sp = np.full(len(ks), np.nan); step = np.zeros(len(ks)); hi = np.zeros(len(ks))
    for i in range(1, len(ks)):
        gap = int(ks[i] - ks[i - 1]); d = float(np.linalg.norm(Ps[i] - Ps[i - 1])); s = d / (gap * dt)
        if gap <= 3:
            if s <= max_ms:
                sp[i] = s if s >= 0.5 else 0.0; step[i] = sp[i] * gap * dt; hi[i] = step[i] if s > hi_ms else 0.0
        elif gap * dt <= gap_line_s and 0.5 <= s <= gap_max_ms:                    # standing still (< 0.5 m/s) counts 0, like a seen step
            step[i] = d; hi[i] = d if s > hi_ms else 0.0
    return sp, step, hi


TOP_SPEED = os.environ.get("IPANEMA_TOP_SPEED", "sus1")   # N0-29c: best 1-s average inside a seen run (old = fastest single row)
HI_MODE = os.environ.get("IPANEMA_HI", "old")             # N0-29c: tried > 5.5 m/s held >= 1 s (sus1): 15-35% of the truth on Metrica (our seen runs are ~0.6 s), old 103-150% - old kept


def seen_speeds(ks, P, fps, jump_m=2.0, max_ms=10.0):
    """N0-29c: per-row speed (m/s, NaN = not measured) from positions smoothed 0.8 s within each seen run, as
    track_motion; a raw step of more than jump_m in <= 3 frames (the track jumped to another person, as metrics.runs)
    splits the run and is not measured. Returns (speeds, seen-step mask: gap <= 3 frames and measured)."""
    ks = np.asarray(ks); P = np.asarray(P, float); n = len(ks); sp = np.full(n, np.nan)
    if n < 2: return sp, np.zeros(n, bool)
    g = np.diff(ks)
    jump = np.r_[False, (g <= 3) & (np.linalg.norm(np.diff(P, axis=0), axis=1) > jump_m)] if jump_m else np.zeros(n, bool)
    Ps = smooth_runs(ks + 10 * np.cumsum(jump), P, int(0.8 * fps), runs=True)
    for i in range(1, n):
        gi = int(ks[i] - ks[i - 1])
        if gi <= 3 and not jump[i]:
            s = float(np.linalg.norm(Ps[i] - Ps[i - 1])) * fps / gi
            if s <= max_ms: sp[i] = s if s >= 0.5 else 0.0
    return sp, ~np.isnan(sp)


def _stretches(ok):
    """(start, end) index pairs of consecutive True rows"""
    out = []; i = 0; n = len(ok)
    while i < n:
        if not ok[i]: i += 1; continue
        j = i
        while j + 1 < n and ok[j + 1]: j += 1
        out.append((i, j)); i = j + 1
    return out


def top_speed(ks, P, fps, old=None, pct=None, win_s=1.0, jump_m=3.0):
    """N0-29c (29 Oct): a player's top speed = his best win_s-second average speed, only over time he was seen the whole
    time (one seen run, no gap, no jump to another person). The old figure (old=True, IPANEMA_TOP_SPEED=old) was the
    fastest single row of track_motion - with our ~0.3 m feet jitter that is 8+ m/s on most tracks. Metrica pro data
    cut like our tracks: results/qa/n0-29c. pct = a percentile of row speeds instead (lab only)."""
    old = (TOP_SPEED == "old") if old is None else old
    if old:
        sp = track_motion(ks, P, fps)[0]
        return round(float(np.nanmax(sp)), 2) if (~np.isnan(sp)).any() else 0.0
    sp, ok = seen_speeds(ks, P, fps, jump_m=jump_m)
    if not ok.any(): return 0.0
    if pct is not None: return round(float(np.nanpercentile(sp, pct)), 2)
    ks = np.asarray(ks); best = 0.0; need = win_s * fps
    for a, b in _stretches(ok):                                         # rows a..b: steps (ks[i-1] -> ks[i]) all seen
        fr = (ks[a:b + 1] - ks[a - 1:b]).astype(float); d = sp[a:b + 1] * fr / fps
        cf = np.r_[0, np.cumsum(fr)]; cd = np.r_[0, np.cumsum(d)]; lo = 0
        for hi_ in range(1, len(cf)):
            while cf[hi_] - cf[lo + 1] >= need: lo += 1
            if cf[hi_] - cf[lo] >= need: best = max(best, (cd[hi_] - cd[lo]) / ((cf[hi_] - cf[lo]) / fps))
    return round(best, 2)


def high_intensity(ks, P, fps, old=None, min_s=1.0, gap_hi=True, hi_ms=5.5, jump_m=2.0):
    """N0-29c (29 Oct): high-intensity distance = seen stretches that stay above hi_ms for at least min_s, plus a
    track gap's straight line (track_motion's rule: <= 2.5 s, <= 7 m/s) above hi_ms when the seen rows either side are
    above hi_ms too. The old figure (old=True, IPANEMA_HI=old) counted every step above hi_ms, so feet jitter on a
    jogging player and every fast gap line counted (103-163% of the truth on Metrica, N0-22c)."""
    old = (HI_MODE == "old") if old is None else old
    if old: return round(float(track_motion(ks, P, fps, hi_ms=hi_ms)[2].sum()), 1)
    ks = np.asarray(ks); sp, ok = seen_speeds(ks, P, fps, jump_m=jump_m); tot = 0.0
    fast = ok & (sp > hi_ms)
    for a, b in _stretches(fast):
        fr = (ks[a:b + 1] - ks[a - 1:b]).astype(float)
        if fr.sum() / fps >= min_s: tot += float((sp[a:b + 1] * fr / fps).sum())
    if gap_hi and GAP_LINE_S > 0:
        Ps = smooth_runs(ks, np.asarray(P, float), int(0.8 * fps), runs=True)
        for i in range(1, len(ks) - 1):
            gi = int(ks[i] - ks[i - 1])
            if gi <= 3 or gi / fps > GAP_LINE_S: continue
            d = float(np.linalg.norm(Ps[i] - Ps[i - 1])); s = d * fps / gi
            if hi_ms < s <= GAP_MAX_MS and fast[i - 1] and fast[i + 1]: tot += d
    return round(tot, 1)


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

def carriers_from_state(per, ballm, state, max_m=2.5):
    """S1 (29 Sep): carrier records built from a possession state (e.g. possession.possession_simple): while the state
    says team A (0) or B (1) has the ball, the carrier is that team's player nearest the ball (if within max_m);
    loose/dead frames have no carrier. Same record shape as possession.carriers, so passes() can use it."""
    out = []
    for k in range(len(per)):
        rec = {"frame": k, "carrier": None, "team": None, "pressure_m": None, "near_opps": None, "pos": None}
        s = int(state[k]) if k < len(state) else 2; b = ballm.get(k)
        if s in (0, 1) and b is not None:
            tm = "A" if s == 0 else "B"; rows = [r for r in per[k] if r[1] == tm]
            if rows:
                d = [float(np.linalg.norm(r[2] - b)) for r in rows]; i = int(np.argmin(d))
                if d[i] <= max_m: rec.update({"carrier": rows[i][0], "team": tm, "pos": rows[i][2].tolist()})
        out.append(rec)
    return out

def _ball_vel(ballm, ks, fps, out_m=2.0):
    """robust straight-line fit of ball positions over frames ks -> velocity (m/s) or None (too few picks)"""
    pts = [(k, ballm[k]) for k in ks if k in ballm]
    if len(pts) < 4: return None
    t = np.array([q[0] for q in pts], float); X = np.array([q[1] for q in pts], float)
    for _ in range(2):                                                         # drop wrong picks, refit
        A = np.c_[t, np.ones_like(t)]; co = np.linalg.lstsq(A, X, rcond=None)[0]; res = np.linalg.norm(A @ co - X, axis=1)
        keep = res < max(out_m, 3 * float(np.median(res)))
        if keep.sum() < 4: return None
        t, X = t[keep], X[keep]
    A = np.c_[t, np.ones_like(t)]; return np.linalg.lstsq(A, X, rcond=None)[0][0] * fps

def touch_episodes(frames_, fps, min_touch_s=0.6, floor_s=0.08, sandwich=True, team_sandwich_s=0.3, mate_sandwich_s=0.0, state=None,
                   ballm=None, pass_by_mps=0.0, pass_by_win_s=0.4):
    """N0-26c (26 Oct): the cleaned ball contacts that passes() links into passes, as [first frame, last frame, player,
    team] - split out of passes() unchanged so the per-player touch count can use the same contacts."""
    episodes = []; cur = None
    for k, f in enumerate(frames_):
        c = f["carrier"]
        if c is not None and state is not None and k < len(state) and int(state[k]) in (0, 1) and f["team"] != ("A" if int(state[k]) == 0 else "B"): c = None
        if c is None:
            if cur and k - cur[1] > int(0.4 * fps): episodes.append(cur); cur = None
            continue
        if cur and cur[2] == c: cur[1] = k
        else:
            if cur: episodes.append(cur)
            cur = [k, k, c, f["team"]]
    if cur: episodes.append(cur)
    # 28 Sep: a wrong ball pick for a few frames next to another player looks like a touch; on Metrica pro data with
    # our tracking noise this counted 2-3x too many passes. A touch must last min_touch_s; touches by the same player
    # separated only by flicker are one touch.
    mt = max(1, int(round(min_touch_s * fps))); floor = max(1, int(round(floor_s * fps)))
    def merge(eps):
        out_ = []
        for e in eps:
            if out_ and out_[-1][2] == e[2] and e[0] - out_[-1][1] <= int(1.0 * fps): out_[-1][1] = e[1]
            else: out_.append(list(e))
        return out_
    episodes = merge([e for e in episodes if e[1] - e[0] + 1 >= floor])
    if sandwich:                                                                # A, short X, A = flicker, not two passes
        changed = True
        while changed:
            changed = False; keep = []
            for i, e in enumerate(episodes):
                short = e[1] - e[0] + 1 < mt
                if short and 0 < i < len(episodes) - 1 and episodes[i - 1][2] == episodes[i + 1][2]: changed = True; continue
                keep.append(e)
            episodes = merge(keep)
    else:
        episodes = merge([e for e in episodes if e[1] - e[0] + 1 >= mt])
    if team_sandwich_s or mate_sandwich_s:
        ts = int(round(team_sandwich_s * fps)); ms = int(round(mate_sandwich_s * fps)); i = 1; keep = episodes[:1]
        wn = max(4, int(round(pass_by_win_s * fps)))
        def brief(e, tm):
            if e[1] - e[0] + 1 >= (ts if e[3] != tm else ms): return False
            if not (pass_by_mps and ballm): return True
            va, vb = _ball_vel(ballm, range(e[0] - wn, e[0]), fps), _ball_vel(ballm, range(e[1] + 1, e[1] + 1 + wn), fps)
            return va is None or vb is None or float(np.linalg.norm(vb - va)) < pass_by_mps
        while i < len(episodes):
            j = i
            while j < len(episodes) and brief(episodes[j], keep[-1][3]): j += 1
            if j > i and j < len(episodes) and episodes[j][3] == keep[-1][3] and episodes[j][0] - keep[-1][1] <= int(4 * fps): i = j; continue
            keep.append(episodes[i]); i += 1
        episodes = merge(keep)
    return episodes


TOUCH_MIN_S = float(os.environ.get("IPANEMA_TOUCH_MIN_S", "0"))


def touch_counts(episodes, fps, min_s=None):
    """N0-26c (26 Oct): touches per player = his cleaned ball contacts (touch_episodes, the ones passes are built from)
    that last >= min_s. Metrica pro data, 20 runs (perfect ball + our ball noise): the old count (every run of carrier
    frames, new after a 0.4 s gap) read 1.7-4.0x the real spells on the ball, a player median 172% off (59-290), rank
    order 0.60; contacts read 1.2-1.7x, 29% off (18-62), rank 0.88. A 0.2-0.5 s minimum is closer still on Metrica
    (0.7-1.5x), but on our clips most contacts are shorter (median 0.25 s: 0.4 s would keep 42 of 108), so min_s 0 =
    every contact (IPANEMA_TOUCH_MIN_S)."""
    m = TOUCH_MIN_S if min_s is None else min_s; out = {}
    for e in episodes:
        if (e[1] - e[0] + 1) >= m * fps: out[int(e[2])] = out.get(int(e[2]), 0) + 1
    return out


def touch_frames(frames_, episodes):
    """N0-26d (26 Oct): per player, the frames he is the carrier inside one of his cleaned contacts (touch_episodes)."""
    out = {}
    for e in episodes:
        p = int(e[2]); S = out.setdefault(p, set())
        S.update(k for k in range(int(e[0]), min(len(frames_), int(e[1]) + 1)) if frames_[k]["carrier"] is not None and int(frames_[k]["carrier"]) == p)
    return out


def touch_time(frames_, episodes, fps):
    """N0-26d (26 Oct): seconds on the ball per player = his carrier frames that fall inside one of his cleaned contacts
    - not every frame he is nearest to the ball (a ball flying past, a 1-frame pick, the frames the other team has it
    in the spell state). Metrica pro data, 20 runs (tools/n0_26d_onball.py): per-player error 63 -> 54% (perfect ball),
    71 -> 30% (flicker, like our clips), 23 -> 28% (streak); share of the team's time 0.9-1.4 -> 0.7-1.1 pts off, rank
    order same or better. App clips: no player piece shows time on the ball with 0 touches (was ~70 per 5 min)."""
    return {p: len(S) / fps for p, S in touch_frames(frames_, episodes).items()}


BETTER_MARGIN = 6.0
# N0-27a (27 Oct): 'a better pass was open' (teams[] / players[] better_option_count, the Insights card). With PERFECT inputs
# (Metrica pro data, all 22 players) the rule calls 83% of completed pro passes second best and only 66% of the passes
# that were cut out - it rewards the furthest forward team-mate (best option median 18 m forward, 4 points per opponent
# 'bypassed') and so says the opposite of a decision quality; no tried variant fixes that (tools/n0_27a_rule.py).
# IPANEMA_BETTER=withhold writes None for both counts (the app then shows Withheld); default 'on' = unchanged.
BETTER = os.environ.get("IPANEMA_BETTER", "on")


def option_values(rows, me_id, team, sgn, lanes_j):
    """value of every team-mate the carrier me_id could pass to (lanes_j = lanes()[k]): gain towards goal + 4 per opponent
    bypassed + 0.5 per metre of space (max 10), -25 if the lane is blocked. {} when the carrier is not in rows."""
    me = next((r for r in rows if r[0] == me_id), None)
    if me is None: return {}
    cpos = me[2]; opps = [r[2] for r in rows if r[1] != team]; pos = {r[0]: r[2] for r in rows}
    def val(rpos, is_open):
        g = float((rpos[0] - cpos[0]) * sgn); lo, hi = sorted([cpos[0], rpos[0]])
        byp = int(sum(1 for o in opps if lo < o[0] < hi and abs(o[1] - (cpos[1] + rpos[1]) / 2) < 20)); sp = float(min((np.linalg.norm(o - rpos) for o in opps), default=30.0))
        return {"gain_m": round(g, 1), "bypassed": byp, "space_m": round(sp, 1), "value": round((g + 4.0 * byp + min(sp, 10.0) * 0.5) - (0 if is_open else 25.0), 1), "open": bool(is_open)}
    return {l["to"]: val(pos[l["to"]], l["open"]) for l in lanes_j if l["to"] in pos}


def best_option(rows, me_id, team, sgn, lanes_j, to):
    """the 'a better pass was open' fields of one pass (N0-27a: pulled out of passes() unchanged, so the Metrica check
    uses the very same rule): better_option = another team-mate valued >= BETTER_MARGIN above the one played to."""
    opts = option_values(rows, me_id, team, sgn, lanes_j)
    if not opts: return {}
    played = opts.get(to); bt = max(opts, key=lambda t: opts[t]["value"]); best = opts[bt]
    return {"played_value": played["value"] if played else None, "best_to": int(bt), "best_value": best["value"], "best_gain_m": best["gain_m"], "best_bypassed": best["bypassed"], "best_space_m": best["space_m"], "best_open": best["open"],
            "better_option": bool(played is not None and bt != to and best["value"] >= played["value"] + BETTER_MARGIN)}


# N0-27b (27 Oct): a lost pass's 'to' is the opponent who won it, never a lane of the passer's team, so a lost pass never got a
# lane or a quality: bad_lost / execution_error (players[].pass_quality, events pass_bad) were ~always 0 (1-3 of 25-77 lost
# passes per app copy). Now the intended receiver of a lost pass = the passer's team-mate in reach (the lanes) nearest the
# spot where the ball was won (the same proxy as the Metrica check, tools/n0_27b_lanes.py); passes[].intended_to.
# IPANEMA_LOST_LANE=old = old behaviour.
LOST_LANE = os.environ.get("IPANEMA_LOST_LANE", "mate")


def intended_mate(rows, lanes_j, won_at):
    """the team-mate (one of lanes_j's 'to') nearest the spot won_at where the opponent got the ball; None if no lane"""
    pos = {r[0]: r[2] for r in rows}; c = [(float(np.linalg.norm(pos[l["to"]] - won_at)), l["to"]) for l in lanes_j if l["to"] in pos]
    return int(min(c)[1]) if c else None


def passes(per, frames_, turnovers_, lanes_, attack_right, fps, prog_m=10.0, lane_half=1.5, min_touch_s=0.6, floor_s=0.08, sandwich=True,
           team_sandwich_s=0.3, mate_sandwich_s=0.0, min_pass_m=5.0, state=None, ballm=None, pass_by_mps=0.0, pass_by_win_s=0.4, debug=False):
    """S1 (29 Sep, Metrica pro games, results/metrica/passes_2026-09-29.md): defaults team_sandwich_s=0.3 and min_pass_m=5
    found 73% of the real passes (was 61%) with 73% of ours real (was 65%) and put counts within 9% on all 4 halves,
    clean and noisy (was 13%). 0 turns each rule off (old behaviour).
    team_sandwich_s: a short contact (< team_sandwich_s) by the OTHER team between two contacts of one team is
    the ball running past an opponent (or a wrong ball pick), not two turnovers; it is dropped so the pass between
    the two teammates is counted. mate_sandwich_s: same for a short contact by a TEAMMATE (the ball running past him).
    min_pass_m: ball moves between two contacts shorter than this are not passes.
    pass_by_mps (with ballm {frame: metres}): a short contact only counts as 'ball ran past' (and is dropped) if the ball's
    velocity (robust line fit over pass_by_win_s before and after) changes by less than this; a real touch deflects it.
    state: optional possession state per frame (0 A, 1 B, 2 loose, 3 dead); a carrier
    frame whose team disagrees with a team state is ignored."""
    tracks = defaultdict(dict)
    for k in range(len(per)):
        for r in per[k]: tracks[r[0]][k] = r[2]
    episodes = touch_episodes(frames_, fps, min_touch_s=min_touch_s, floor_s=floor_s, sandwich=sandwich, team_sandwich_s=team_sandwich_s,
                              mate_sandwich_s=mate_sandwich_s, state=state, ballm=ballm, pass_by_mps=pass_by_mps, pass_by_win_s=pass_by_win_s)
    out = []
    for a, b in zip(episodes, episodes[1:]):
        if b[0] - a[1] > int(4 * fps): continue
        pa, pb = tracks[a[2]].get(a[1]), tracks[b[2]].get(b[0])
        if pa is None or pb is None: continue
        if min_pass_m and float(np.linalg.norm(pb - pa)) < min_pass_m: continue
        sgn = 1 if attack_right[a[3]] else -1; gain = float((pb[0] - pa[0]) * sgn)
        kind = "forward" if gain > 3 else "backward" if gain < -3 else "sideways"
        p = {"t": round(a[1] / fps, 2), "from": int(a[2]), "to": int(b[2]), "team": a[3], "completed": b[3] == a[3], "from_m": [round(float(pa[0]), 1), round(float(pa[1]), 1)],
             "to_m": [round(float(pb[0]), 1), round(float(pb[1]), 1)], "gain_m": round(gain, 1), "length_m": round(float(np.linalg.norm(pb - pa)), 1), "kind": kind, "progressive": gain >= prog_m and b[3] == a[3],
             "lane_open": None, "quality": "unknown", "missed_open_forward": None, "better_option": False}
        if debug: p["_eps"] = (a[0], a[1], b[0], b[1])                              # frames of the passer's and receiver's contact
        k_rel = a[1]; win = [j for j in range(k_rel - 3, k_rel + 1) if j in lanes_]
        if win:
            j = win[-1]; ln = {l["to"]: l for l in lanes_[j]}
            if LOST_LANE == "mate" and not p["completed"]:
                mt = intended_mate(per[j], lanes_[j], pb)
                if mt is not None: p["intended_to"] = mt
            tgt = p.get("intended_to", p["to"])
            if tgt in ln:
                op = ln[tgt]["open"]; p["lane_open"] = op
                p["quality"] = "good" if op and p["completed"] else "risky_completed" if (not op) and p["completed"] else "bad_lost" if (not op) else "execution_error"
                p["missed_open_forward"] = bool(any(l["open"] and l["forward"] for l in lanes_[j] if l["to"] != tgt) and kind != "forward")
                # best option valuation
                if tgt == p["to"]: p.update(best_option(per[j], a[2], a[3], sgn, lanes_[j], tgt))   # N0-27b: a lost pass gets no 'better option' (unchanged)
        out.append(p)
    return out, tracks

# N0-22e (22 Oct): the camera shows only part of the pitch, so a team's block length / width from the players in the
# picture came out ~25% / ~20% short (Metrica pro tracking seen through our clips' real camera views + our track breaks:
# length 23 vs 32 m, width 32 vs 40 m). A player who drops out of the picture now keeps his last seen spot for
# SHAPE_HOLD_S seconds (the usual 'last known position'): length / width -5% / -7%. The line height (deepest player) was
# already right from the seen players (+-2 m) and holding makes it worse, so it keeps the seen players. 0 = old numbers.
SHAPE_HOLD_S = float(os.environ.get("IPANEMA_SHAPE_HOLD_S", "5"))
# N0-22f (22 Oct): 'regained within 5 s' (counter-press success) and its mirror 'lost back in 5 s'. On Metrica pro data
# (tools/n0_22f_regain.py) the truth is 17-28% of all losses per half; the pipeline gives 2-19% with a perfect ball and
# 0-11% with our streak noise, because a quick regain is mostly not a counted loss at all (spell state + S9 rule).
# With the ball flicker of our clips the raw state gives ~80%. No rule tried tracks the truth under our noise.
# IPANEMA_REGAIN=withhold writes None for both team figures (the app then shows a dash); default 'on' = unchanged.
REGAIN = os.environ.get("IPANEMA_REGAIN", "on")

SHAPE_EDGE_M = float(os.environ.get("IPANEMA_SHAPE_EDGE_M", "1.5"))

def held_points(per, fps, hold_s=None, min_seen=3, L=None, W=None, edge_m=None):
    """per frame k -> {team: (n x 2) array}: the outfield players in the picture + the last seen spot of each track of that
    team that left the picture within hold_s seconds (keepers left out, like the shapes). A frame with fewer than
    min_seen seen players of a team gives None for it (a held team alone is no reading); an empty frame (half-time,
    untrusted camera) clears the held spots. With L, W: a spot within edge_m of the pitch's edge is not held - on our clips
    16-18% of held spots sat on the touchline / goal line (seen spots 3-8%, Metrica pro players 0.7%): people at the line
    and feet cut by the picture's edge, not players."""
    hold_s = SHAPE_HOLD_S if hold_s is None else hold_s; hk = hold_s * fps; last = {}; out = []
    e = SHAPE_EDGE_M if edge_m is None else edge_m
    def off(m): return L is not None and W is not None and e > 0 and (m[0] < e or m[0] > L - e or m[1] < e or m[1] > W - e)
    for k in range(len(per)):
        rows = per.get(k, []) if isinstance(per, dict) else per[k]
        if not rows: last.clear(); out.append({"A": None, "B": None}); continue
        seen = {}
        for r in rows:
            if r[1] in ("A", "B") and not r[5]: seen[r[0]] = (r[1], np.asarray(r[2], float)); last[r[0]] = (r[1], np.asarray(r[2], float), k)
        for t in [t for t, v in last.items() if k - v[2] > hk + 1e-9 or (t not in seen and off(v[1]))]: del last[t]
        fr = {}
        for tm in ("A", "B"):
            if sum(1 for v in seen.values() if v[0] == tm) < min_seen: fr[tm] = None; continue
            fr[tm] = np.array([v[1] for v in last.values() if v[0] == tm])
        out.append(fr)
    return out

# N0-25b (25 Oct): a TEAM heat map, ready for the app (stats.teams[].heatmap). The app had to build it from the frames'
# seen players or by summing the per-track grids (stats.heatmaps, each track normalised to 1); our camera shows ~30% of the
# pitch around the ball, so both lean to where the ball is. On Metrica pro data seen through our app clips' camera views
# (tools/n0_25b_heat.py) the seen-players map is 12.5% off the truth (TV distance; thirds shares 4.4 pts off); keeping each
# player's last seen spot for 20 s (held_points, edge rule on) gives 11.0% / 2.2 pts, better on all 16 team-halves' TV.
# 0 = only the players in the picture (frames with >= 3 seen).
TEAM_HEAT_HOLD_S = float(os.environ.get("IPANEMA_TEAM_HEAT_HOLD_S", "20"))

# N0-28f (28 Oct): per-player average position / time in each third / heat map (players[].avg_x_m, avg_y_m, pct_*_third,
# stats.heatmaps) come from the frames where the player is in the picture, so they lean to where the ball was. On Metrica
# pro data seen through our clips' camera views (tools/n0_28f_playerpos.py, one id per player = what M2 gives) the
# average spot is 3.2 m off (p90 6.1), thirds 4.5 pts, heat map 15.9%; keeping each player's last seen spot 10 s after he
# leaves the picture: 1.7 m (p90 4.2), 3.5 pts, 13.9% - better on 8/8 runs for all three. On today's broken track pieces
# it is slightly worse (each piece is ~19 m from its player's average either way) and the app hides these fields, so it
# is OFF until M2: IPANEMA_PLAYER_HOLD_S=10 switches it on.
PLAYER_HOLD_S = float(os.environ.get("IPANEMA_PLAYER_HOLD_S", "0"))


def held_track_points(per, fps, hold_s, L=None, W=None, edge_m=None):
    """{track id: (n x 2) array}: every spot where the track is in the picture + its last seen spot for each frame up to
    hold_s seconds after it left (same rules as held_points: an empty frame clears, no held spot within edge_m of a line)."""
    hk = hold_s * fps; e = SHAPE_EDGE_M if edge_m is None else edge_m; last = {}; out = {}
    def off(m): return L is not None and W is not None and e > 0 and (m[0] < e or m[0] > L - e or m[1] < e or m[1] > W - e)
    for k in range(len(per)):
        rows = per.get(k, []) if isinstance(per, dict) else per[k]
        if not rows: last.clear(); continue
        seen = set()
        for r in rows:
            m = np.asarray(r[2], float); last[r[0]] = (m, k); seen.add(r[0])
        for t in [t for t, v in last.items() if t not in seen and (k - v[1] > hk + 1e-9 or off(v[0]))]: del last[t]
        for t, (m, _) in last.items(): out.setdefault(t, []).append(m)
    return {t: np.array(v) for t, v in out.items()}


def team_heatmaps(per, fps, L, W, attack_right=None, hold_s=None, grid=(12, 8)):
    """-> {team: {grid (cells x along the pitch, normalised to 1, same layout as stats.heatmaps), hold_s, frames,
    thirds_pct [own, middle, attacking] when attack_right is given} or None when the team was never seen}. Outfield only."""
    h = TEAM_HEAT_HOLD_S if hold_s is None else hold_s
    held = held_points(per, fps, h, L=L, W=W); out = {}
    for tm in ("A", "B"):
        g = np.zeros(grid); nf = 0
        for fr in held:
            P = fr[tm]
            if P is None or len(P) == 0: continue
            i = np.clip((P[:, 0] / L * grid[0]).astype(int), 0, grid[0] - 1); j = np.clip((P[:, 1] / W * grid[1]).astype(int), 0, grid[1] - 1)
            np.add.at(g, (i, j), 1); nf += 1
        if g.sum() == 0: out[tm] = None; continue
        g = g / g.sum(); d = {"grid": g.round(4).tolist(), "hold_s": h, "frames": nf}
        if attack_right is not None:
            col = g.sum(1); xc = (np.arange(grid[0]) + 0.5) / grid[0] * L
            th = [float(col[xc < L / 3].sum()), float(col[(xc >= L / 3) & (xc < 2 * L / 3)].sum()), float(col[xc >= 2 * L / 3].sum())]
            d["thirds_pct"] = [round(100 * v) for v in (th if attack_right[tm] else th[::-1])]
        out[tm] = d
    return out

def shapes(per, L, fps=25.0):
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
    # a team's shape that drops out for under hold_s (a player or two briefly untracked) keeps its last value: no flicker
    hold = int(1.5 * fps) if fps else 0; last = {"A": (None, -10 ** 9), "B": (None, -10 ** 9)}
    for k in range(len(per)):
        for tm in ("A", "B"):
            if out[k][tm] is not None: last[tm] = (out[k][tm], k)
            elif last[tm][0] is not None and k - last[tm][1] <= hold: out[k][tm] = dict(last[tm][0], held=True)
    return out

def stats(per, frames_, turnovers_, passes_, tracks, state, fps, L, W, attack_right, press_r=2.0, near_r=5.0, sequences_=None, episodes_=None):
    """sequences_ (2 Oct, tools/presslab.py): when given, pressures_applied = one per presser per possession sequence (the
    usual 'pressure event'); without it the old count (one per second within press_r of the opposing carrier, which gave
    700-1,500 per team per 90 min on AIK/SFK-BP because the carrier flickers; per sequence: 250-340)."""
    from .possession import STATES
    n = len(per); dt = 1 / fps; pass_df = pd.DataFrame(passes_)
    ids = [t for t in tracks if len(tracks[t]) * dt >= 2.0]
    team_of = {}
    for k in range(n):
        for r in per[k]: team_of.setdefault(r[0], []).append(r[1])
    team_id = {t: majority(team_of[t]) for t in ids}                       # N0-20d: tie -> the team it read last (was: chance)
    react = {}
    for tv in turnovers_:
        for cat, lst in tv.get("reactions", {}).items():
            for tid in lst: d = react.setdefault(tid, {"losses_seen": 0, "pressed": 0, "jogged": 0, "stood": 0}); d["losses_seen"] += 1; d[cat] += 1
    press_by_seq = {}
    if sequences_ is not None:
        for sq in sequences_:
            seen = set()
            for k in range(sq["start"], min(n, sq["end"] + 1)):
                f = frames_[k]; c = f["carrier"]
                if c is None or f["team"] != sq["team"] or c not in tracks or k not in tracks[c]: continue
                for r in per[k]:
                    if r[1] != sq["team"] and r[0] not in seen and np.linalg.norm(r[2] - tracks[c][k]) <= press_r: seen.add(r[0]); press_by_seq[r[0]] = press_by_seq.get(r[0], 0) + 1
    # N0-26c (26 Oct): episodes_ (touch_episodes, as passes() saw them) -> touches = contacts >= TOUCH_MIN_S (touch_counts);
    # None or IPANEMA_TOUCHES=old = the old count (every run of carrier frames)
    tcount = touch_counts(episodes_, fps) if episodes_ is not None and os.environ.get("IPANEMA_TOUCHES", "contacts") != "old" else None
    # N0-26d (26 Oct): time on the ball / under pressure = carrier frames inside his cleaned contacts (touch_frames);
    # None or IPANEMA_ONBALL=old = every carrier frame
    tframes = touch_frames(frames_, episodes_) if episodes_ is not None and os.environ.get("IPANEMA_ONBALL", "contacts") != "old" else None
    players = []; heat = {}; GRID = (12, 8)
    held_pos = held_track_points(per, fps, PLAYER_HOLD_S, L=L, W=W) if PLAYER_HOLD_S > 0 else {}   # N0-28f, off until M2
    for tid in ids:
        tm = team_id[tid]; ks = np.array(sorted(tracks[tid])); P = np.array([tracks[tid][k] for k in ks]); sgn = 1 if attack_right[tm] else -1
        PP = held_pos.get(tid, P) if len(held_pos.get(tid, ())) else P       # spots for position / thirds / heat (seen + held)
        sp, step, hi_step = track_motion(ks, P, fps); valid = ~np.isnan(sp)   # N0-22c: smoothed per seen run + short gaps as a straight line (was: smoothed across gaps, gaps not counted)
        on_all = [k for k in ks if frames_[k]["carrier"] == tid]; on = on_all if tframes is None else [k for k in on_all if k in tframes.get(int(tid), ())]; pressed = [k for k in on if frames_[k]["pressure_m"] is not None and frames_[k]["pressure_m"] <= press_r]
        pr = 0; lastp = -99
        if sequences_ is not None: pr = press_by_seq.get(tid, 0)
        else:
            for k in ks:
                f = frames_[k]
                if f["carrier"] is not None and f["team"] != tm and f["carrier"] in tracks and k in tracks[f["carrier"]]:
                    if np.linalg.norm(tracks[tid][k] - tracks[f["carrier"]][k]) <= press_r and k - lastp > fps: pr += 1; lastp = k
        mp = pass_df[pass_df["from"] == tid] if len(pass_df) else pass_df; rc = pass_df[(pass_df["to"] == tid) & pass_df["completed"]] if len(pass_df) else pass_df
        g = np.zeros(GRID)
        for x, y in PP: g[min(GRID[0] - 1, max(0, int(x / L * GRID[0]))), min(GRID[1] - 1, max(0, int(y / W * GRID[1])))] += 1
        heat[str(tid)] = (g / max(1, g.sum())).round(4).tolist(); rx = react.get(tid)
        xs = PP[:, 0] if sgn == 1 else L - PP[:, 0]
        players.append({"id": int(tid), "team": tm, "time_visible_s": round(len(ks) * dt, 1), "time_on_ball_s": round(len(on) * dt, 1), "touches": int(tcount.get(int(tid), 0)) if tcount is not None else int(sum(1 for i in range(1, len(on_all)) if on_all[i] - on_all[i - 1] > int(0.4 * fps)) + (1 if on_all else 0)),
                        "time_under_pressure_s": round(len(pressed) * dt, 1), "passes": int(len(mp)), "passes_completed": int(mp["completed"].sum()) if len(mp) else 0,
                        "passes_forward": int((mp["kind"] == "forward").sum()) if len(mp) else 0, "passes_sideways": int((mp["kind"] == "sideways").sum()) if len(mp) else 0, "passes_backward": int((mp["kind"] == "backward").sum()) if len(mp) else 0,
                        "progressive_passes": int(mp["progressive"].sum()) if len(mp) else 0, "passes_received": int(len(rc)), "pressures_applied": pr,
                        "pass_quality": {q: int((mp["quality"] == q).sum()) if len(mp) else 0 for q in ("good", "risky_completed", "bad_lost", "execution_error")}, "better_option_count": (int(mp["better_option"].sum()) if len(mp) else 0) if BETTER != "withhold" else None, "passes_risky": int((mp["quality"] == "risky_completed").sum()) if len(mp) else 0,
                        "distance_m": round(float(step.sum()), 1), "high_intensity_m": round(float(hi_step.sum()), 1) if HI_MODE == "old" else high_intensity(ks, P, fps), "top_speed_ms": top_speed(ks, P, fps),   # N0-29c: best 1-s average while seen (was: fastest single row = jitter, 9.9 m/s on everyone)
                        "avg_x_m": round(float(PP[:, 0].mean()), 1), "avg_y_m": round(float(PP[:, 1].mean()), 1),
                        "pct_def_third": round(100 * float((xs < L / 3).mean())), "pct_mid_third": round(100 * float(((xs >= L / 3) & (xs < 2 * L / 3)).mean())), "pct_att_third": round(100 * float((xs >= 2 * L / 3).mean())),
                        "counterpress_rate": round(rx["pressed"] / rx["losses_seen"], 2) if rx and rx["losses_seen"] else None, "losses_seen": rx["losses_seen"] if rx else 0})
    poss = [STATES[s] if s < 2 else None for s in state]; teams = []
    held = held_points(per, fps, L=L, W=W) if SHAPE_HOLD_S else None
    theat = team_heatmaps(per, fps, L, W, attack_right)
    for tm in ("A", "B"):
        opp = "B" if tm == "A" else "A"; tp = pass_df[pass_df.team == tm] if len(pass_df) else pass_df; op = pass_df[pass_df.team == opp] if len(pass_df) else pass_df
        lens, wids, hts = [], [], []
        for k in range(n):
            pts = np.array([r[2] for r in per[k] if r[1] == tm and not r[5]])
            if len(pts) >= 5:
                xs = np.sort(pts[:, 0]); hts.append(xs[0] if attack_right[tm] else L - xs[-1])
                if not SHAPE_HOLD_S: lens.append(xs[-1] - xs[0]); wids.append(pts[:, 1].max() - pts[:, 1].min())
            hp = held[k][tm] if held is not None else None                     # N0-22e: length / width with players just out of the picture
            if hp is not None and len(hp) >= 5: lens.append(float(hp[:, 0].max() - hp[:, 0].min())); wids.append(float(hp[:, 1].max() - hp[:, 1].min()))
        my = [p for p in players if p["team"] == tm]; da = sum(p["pressures_applied"] for p in my); seqs = (sum(1 for q in sequences_ if q["team"] == tm) if sequences_ is not None else sum(1 for i in range(1, n) if poss[i] == tm and poss[i - 1] != tm))   # S8: the exported sequences, not raw-state flips
        ctrl = sum(1 for p in poss if p is not None); ps = sum(1 for p in poss if p == tm) * dt
        lost = [t for t in turnovers_ if t["lost_by"] == tm]; won = [t for t in turnovers_ if t["won_by"] == tm]
        def med(v): v = [x for x in v if x is not None]; return round(float(np.median(v)), 2) if v else None
        def pct(v): v = [x for x in v if x is not None]; return round(100 * float(np.mean(v))) if v else None
        teams.append({"team": tm, "possession_pct": round(100 * sum(1 for p in poss if p == tm) / max(1, ctrl)), "possession_s": round(ps, 1), "sequences": seqs, "passes": int(len(tp)),
                      "pass_completion_pct": round(100 * tp.completed.mean()) if len(tp) else None, "passes_per_sequence": round(len(tp) / seqs, 1) if seqs else None,
                      "forward_pass_share_pct": round(100 * (tp.kind == "forward").mean()) if len(tp) else None, "progressive_passes": int(tp.progressive.sum()) if len(tp) else 0,
                      "ppda_opp_passes_per_def_action": round(len(op) / da, 1) if da else None, "block_length_median_m": med(lens), "block_width_median_m": med(wids), "def_line_height_median_m": med(hts), "block_hold_s": SHAPE_HOLD_S,
                      "losses": len(lost), "recoveries": len(won), "time_to_press_median_s": med([t["time_to_press"] for t in lost]), "pressed_within_2s_pct": pct([(t["time_to_press"] is not None and t["time_to_press"] <= 2) for t in lost]),
                      "regained_within_5s_pct": pct([t["regained_within_5s"] for t in lost]) if REGAIN != "withhold" else None, "near_at_2s_median": med([t["near_at_2s"] for t in lost]),
                      "time_to_forward_pass_median_s": med([t["time_to_forward_pass"] for t in won]), "forward_within_3s_pct": pct([(t["time_to_forward_pass"] is not None and t["time_to_forward_pass"] <= 3) for t in won]),
                      "lost_back_5s_pct": pct([t["lost_back_5s"] for t in won]) if REGAIN != "withhold" else None, "distance_m_total_visible": round(sum(p["distance_m"] for p in my)), "pressures_applied": da,
                      "better_option_count": (int(tp["better_option"].sum()) if len(tp) else 0) if BETTER != "withhold" else None, "heatmap": theat[tm]}
                     | ({"regain_withheld": "N0-22f: not measurable from our ball (Metrica check)"} if REGAIN == "withhold" else {}))   # 3 Oct: the app reads it per team
    return {"players": players, "teams": teams, "heatmaps": heat, "grid": list(GRID)}

def confirm_passes(ps, kicks, tol=0.7):
    """P-PASS (6 Oct): keep only the passes that a video ball-action model also sees. kicks = video seconds of the model's
    pass moments (SoccerNet ball-action spotting, T-DEED, PASS/HIGH PASS/CROSS/FREE KICK, score >= 0.2, merged < 0.8 s).
    Each kick takes the nearest of our passes within tol s (one-to-one); our pass keeps its own team, players and positions.
    Vallentuna 66:40 clip vs Daniel's 21 tapped passes: ours 52 (16 real) -> 22 (15 real, team right 13/15)."""
    used, keep = set(), []
    order = sorted(range(len(ps)), key=lambda i: ps[i]["t"])
    ts = [ps[i]["t"] for i in order]
    import bisect
    for q in sorted(kicks):
        j = bisect.bisect_left(ts, q - tol); best = None
        while j < len(ts) and ts[j] <= q + tol:
            i = order[j]
            if i not in used and (best is None or abs(ts[j] - q) < abs(ps[best]["t"] - q)): best = i
            j += 1
        if best is not None: used.add(best); keep.append(best)
    return [ps[i] for i in sorted(keep, key=lambda i: ps[i]["t"])]
