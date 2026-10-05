"""V2 (5 Oct): automatic checks on every upload. Each check looks at the exported match (match_data + frames + stats) and,
when something is off, names the part of the app that should say "needs review" instead of showing the numbers.

Checks (all on what the app shows, in match time, inside the periods):
  players    - observed players per team per frame (median), and the two teams not wildly unequal
  keepers    - frames where a team has more than one keeper
  goals      - goal team vs the end it was scored at (attack direction) and vs who kicks off after it
  shot_map   - share of shots that have a pitch spot (x/y)
  ball_map   - share of ball positions (metres, the mini-map) off the pitch
  possession - who-has-the-ball by eye (reference/<match>/ball_key_graded.json 'owner' or owner_key.json) vs our state

Result: {"needs_review": [...parts...], "checks": [{"name", "ok", "value", "limit", "detail"}]}. ok=None means "could not check".
   PYTHONPATH=. python tools/upload_check.py <match_id>      (reads results/volume/runs/matches/<match_id>)
"""
import os, json, glob, bisect, statistics

LIMITS = {
    "players_min_median": 4,      # fewer observed players per team per frame (median) than this -> players need review
    "players_ratio_min": 0.5,     # smaller team's median / larger team's median (Vallentuna 3 vs 11 on 4 Oct)
    "players_over11_share": 0.05, # frames with more than 11 of one team
    "keepers_multi_share": 0.02,  # frames with 2+ keepers in one team (262 'keepers' on Vallentuna before the 4 Oct fix)
    "shots_located_min": 0.8,     # shots with a pitch spot
    "ball_offpitch_max": 0.10,    # ball positions more than 3 m outside the lines
    "ball_offpitch_margin_m": 3.0,
    "possession_key_min": 0.7,    # share of clear by-eye moments where our team with the ball is right
    "possession_key_n": 10,       # fewer clear moments than this -> not graded
    "kickoff_window_s": (10.0, 180.0), "centre_r_m": 9.0,
}
PART = {"players": "players", "keepers": "keepers", "goals": "goals", "shots": "shot_map", "ball": "ball_map", "possession": "possession"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _in_periods(t, periods):
    return not periods or any(p["t_start"] - 1e-6 <= t <= p["t_end"] + 1e-6 for p in periods)


def _row(name, ok, value, limit, detail=""):
    return {"name": name, "ok": ok, "value": value, "limit": limit, "detail": detail}


def check_players(frames, periods, lim=LIMITS):
    per = {"A": [], "B": []}; over = multi = n = 0
    for f in frames:
        if not f.get("cal_ok", True) or not _in_periods(f["t"], periods): continue
        obs = [p for p in f.get("players") or [] if p.get("state", "observed") == "observed"]
        if not f.get("players"): continue                       # no people at all: camera on the sky / cut, not a team fault
        n += 1
        for ab in "AB":
            c = sum(1 for p in obs if p.get("team") == ab); per[ab].append(c)
            if c > 11: over += 1
            if sum(1 for p in f["players"] if p.get("team") == ab and p.get("gk")) > 1: multi += 1
    if not n:
        return [_row("players", None, None, None, "no frames with players"), _row("keepers", None, None, None, "no frames with players")]
    med = {ab: statistics.median(v) for ab, v in per.items()}
    ratio = min(med.values()) / max(med.values()) if max(med.values()) else 0.0
    ok = min(med.values()) >= lim["players_min_median"] and ratio >= lim["players_ratio_min"] and over / n <= lim["players_over11_share"]
    rows = [_row("players", ok, {"median_A": med["A"], "median_B": med["B"], "ratio": round(ratio, 2), "over11_share": round(over / n, 3), "frames": n},
                 {"min_median": lim["players_min_median"], "ratio_min": lim["players_ratio_min"], "over11_share": lim["players_over11_share"]},
                 f"median players per frame A {med['A']:g}, B {med['B']:g}")]
    rows.append(_row("keepers", multi / n <= lim["keepers_multi_share"], round(multi / n, 3), lim["keepers_multi_share"],
                     f"{multi} of {n} frames with 2+ keepers in one team"))
    return rows


def _period_at(t, periods):
    for p in periods or []:
        if p["t_start"] - 30 <= t <= p["t_end"] + 30: return p
    return None


def find_kickoff(frames, t0, t1, L, W, r=9.0, hold_s=1.0):
    """first spell after t0 (before t1) where the ball stays within r m of the centre spot for hold_s -> (t, team) or None.
    Team = who has the ball (frame 'possession') in that spell and the 2 s after it (>= 60% of the readings). The restart list misses most
    kick-offs (SFK-BP, Vallentuna), the frames don't."""
    run = []
    for f in frames:
        if f["t"] < t0: continue
        if f["t"] > t1: break
        b = f.get("ball"); near = bool(b and b.get("m") is not None and ((b["m"][0] - L / 2) ** 2 + (b["m"][1] - W / 2) ** 2) ** 0.5 <= r)
        if near: run.append(f)
        elif run and f["t"] - run[-1]["t"] > 0.5:
            if run[-1]["t"] - run[0]["t"] >= hold_s: break
            run = []
    if not run or run[-1]["t"] - run[0]["t"] < hold_s: return None
    t_end = run[-1]["t"] + 2.0                                                  # the spell + 2 s: the kicker's team keeps it briefly
    votes = [f.get("possession") for f in frames if run[0]["t"] <= f["t"] <= t_end and f.get("possession") in ("A", "B")]
    if not votes: return None
    team = max(("A", "B"), key=votes.count)
    return (run[0]["t"], team) if votes.count(team) >= 0.6 * len(votes) else None


def check_goals(stats, events, frames, periods, L, W, lim=LIMITS):
    """goals as the app counts them (stats.metrics.shots with goal) - checked against the end they were scored at, against
    who kicks off after them, and against the goal events in match_data (both files must say the same)."""
    goals = sorted([s for s in ((stats.get("metrics") or {}).get("shots") or []) if s.get("goal")], key=lambda s: s["t"])
    ev = sorted([e for e in events if e["type"] == "goal"], key=lambda e: e["t"])
    if not goals and not ev: return _row("goals", None, {"goals": 0}, None, "no goals")
    wrong, checked, notes = 0, 0, []
    a = [(round(g["t"]), g.get("team")) for g in goals]; b = [(round(e["t"]), e.get("team")) for e in ev]
    if a != b: wrong += 1; notes.append(f"stats.json goals {a} but match_data goal events {b}")
    lo, hi = lim["kickoff_window_s"]
    for i, g in enumerate(goals):
        team = g.get("team"); x = g.get("x_m"); by_eye = "by eye" in str(g.get("located_by", ""))
        if team not in ("A", "B"): wrong += 1; notes.append(f"{g['t']:.0f}: no team"); continue
        p = _period_at(g["t"], periods); ar = (p or {}).get("attack_right")
        if x is not None and isinstance(ar, dict) and team in ar:                   # end check: scored at the end that team attacks
            checked += 1; right_end = x > L / 2
            if right_end != bool(ar[team]): wrong += 1; notes.append(f"{g['t']:.0f}: {team} scored at the {'right' if right_end else 'left'} end, but attacks the other way")
        nxt = goals[i + 1]["t"] if i + 1 < len(goals) else 1e12                     # kick-off: the other team restarts at the centre spot
        ko = find_kickoff(frames, g["t"] + lo, min(g["t"] + hi, nxt), L, W, lim["centre_r_m"])
        if not ko: notes.append(f"{g['t']:.0f}: kick-off not seen"); continue
        checked += 1
        if ko[1] != team: notes.append(f"{g['t']:.0f}: {team} scored, {ko[1]} kicks off at {ko[0]:.0f}")
        elif by_eye: notes.append(f"{g['t']:.0f}: {team} scored (team checked by eye, kept) but our possession says {team} kicks off at {ko[0]:.0f}")
        else: wrong += 1; notes.append(f"{g['t']:.0f}: {team} scored but {team} kicks off at {ko[0]:.0f}")
    ok = (wrong == 0) if checked or wrong else None
    return _row("goals", ok, {"goals": len(goals), "checks": checked, "wrong": wrong}, {"wrong": 0}, "; ".join(notes))


def check_shots(stats, lim=LIMITS):
    shots = ((stats.get("metrics") or {}).get("shots")) or []
    if not shots: return _row("shots", None, {"shots": 0}, lim["shots_located_min"], "no shots")
    loc = sum(1 for s in shots if s.get("x_m", s.get("x")) is not None)
    share = loc / len(shots)
    return _row("shots", share >= lim["shots_located_min"], round(share, 2), lim["shots_located_min"], f"{loc} of {len(shots)} shots have a spot on the pitch")


def check_ball(frames, periods, L, W, lim=LIMITS):
    m = lim["ball_offpitch_margin_m"]; tot = off = 0
    for f in frames:
        b = f.get("ball")
        if not b or b.get("m") is None or not _in_periods(f["t"], periods): continue
        tot += 1; x, y = b["m"]
        if x < -m or x > L + m or y < -m or y > W + m: off += 1
    if not tot: return _row("ball", None, None, lim["ball_offpitch_max"], "no ball positions")
    return _row("ball", off / tot <= lim["ball_offpitch_max"], round(off / tot, 3), lim["ball_offpitch_max"], f"{off} of {tot} ball positions more than {m:g} m off the pitch")


def load_owner_key(root, match_id):
    """by-eye 'who has the ball' moments -> [(t, 'A'|'B')]. Sources: reference/<m>/owner_key.json ([{t, owner}]) or the 'owner'
    field in reference/<m>/ball_key_graded.json (A/B only; loose / ? / - skipped)."""
    ref = os.path.join(root, "reference", match_id); items = []
    for name in ("owner_key.json", "ball_key_graded.json"):
        p = os.path.join(ref, name)
        if not os.path.exists(p): continue
        with open(p) as fh: d = json.load(fh)
        rows = d.get("items", d.get("moments", [])) if isinstance(d, dict) else d
        items += [(float(r["t"]), r["owner"]) for r in rows if r.get("owner") in ("A", "B")]
    return sorted(set(items))


def check_possession(frames, key, lim=LIMITS, max_dt=0.6):
    if len(key) < lim["possession_key_n"]:
        return _row("possession", None, {"moments": len(key)}, lim["possession_key_min"], f"only {len(key)} by-eye moments (need {lim['possession_key_n']})")
    ts = [f["t"] for f in frames]; right = n = 0; miss = []
    for t, owner in key:
        i = bisect.bisect_left(ts, t); cand = [j for j in (i - 1, i) if 0 <= j < len(ts)]
        if not cand: continue
        j = min(cand, key=lambda j: abs(ts[j] - t))
        if abs(ts[j] - t) > max_dt: continue
        n += 1; ours = frames[j].get("possession")
        if ours == owner: right += 1
        else: miss.append(f"{t:.1f} ours {ours} eye {owner}")
    if n < lim["possession_key_n"]:
        return _row("possession", None, {"moments": n}, lim["possession_key_min"], f"only {n} by-eye moments inside the exported frames")
    return _row("possession", right / n >= lim["possession_key_min"], round(right / n, 2), lim["possession_key_min"],
                f"{right} of {n} clear moments right" + (f" (wrong: {', '.join(miss[:6])})" if miss else ""))


def run_checks(md, stats, frames, owner_key=(), lim=LIMITS):
    L, W = md["pitch"]["length"], md["pitch"]["width"]; periods = md.get("periods") or []
    frames = sorted(frames, key=lambda f: f["t"])
    rows = check_players(frames, periods, lim)
    rows.append(check_goals(stats, md.get("events") or [], frames, periods, L, W, lim))
    rows.append(check_shots(stats, lim))
    rows.append(check_ball(frames, periods, L, W, lim))
    rows.append(check_possession(frames, list(owner_key), lim))
    needs = sorted({PART[r["name"]] for r in rows if r["ok"] is False})
    return {"needs_review": needs, "checks": rows}


def load_export(match_dir):
    with open(os.path.join(match_dir, "match_data.json")) as fh: md = json.load(fh)
    with open(os.path.join(match_dir, "stats.json")) as fh: st = json.load(fh)
    frames = list(md.get("frames") or [])
    for p in sorted(glob.glob(os.path.join(match_dir, "frames_*.json"))):
        with open(p) as fh: frames += json.load(fh)["frames"]
    return md, st, frames


def safe_run(md, stats, frames, match_id, root=REPO, log=print):
    """for export.write: never breaks an upload; returns None when the checks themselves fail"""
    try:
        rep = run_checks(md, stats, frames, load_owner_key(root, match_id)); rep["match_id"] = match_id
        log("  upload checks: " + text(rep).replace("\n", "\n  ")); return rep
    except Exception as e:                                                       # pragma: no cover - logged, upload goes on
        log(f"  upload checks failed: {e!r}"); return None


def check_export(match_dir, root=".", match_id=None):
    md, st, frames = load_export(match_dir)
    mid = match_id or md.get("match_id") or os.path.basename(match_dir.rstrip("/"))
    rep = run_checks(md, st, frames, load_owner_key(root, mid))
    rep["match_id"] = mid; rep["frames_checked"] = len(frames)
    return rep


def text(rep):
    out = [f"{rep.get('match_id', '')}: needs review -> {', '.join(rep['needs_review']) or 'nothing'}"]
    for r in rep["checks"]:
        out.append(f"  {'OK  ' if r['ok'] else ('??  ' if r['ok'] is None else 'FLAG')} {r['name']}: {r['detail']}")
    return "\n".join(out)
