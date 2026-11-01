"""N0-01b (1 Nov): do the app's numbers agree with each other? On each of the repo's app match copies
(results/volume/runs/matches/<id>/stats.json + match_data.json) this checks sums and limits that must hold whatever the
video shows: team totals vs the lists they come from, per-player sub-counts vs their totals, team A vs team B mirrors,
shares that must add to 100, times that can't exceed other times. No model, no video - only the exported numbers.

    python tools/n0_01b_consistency.py [--out results/qa/n0-01b/consistency.json]
"""
import sys, os, json, glob, collections


def _pct_ok(a, b, tol=1):
    return abs((a or 0) + (b or 0) - 100) <= tol


def check(stats, md=None):
    """Returns a list of {check, where, detail} for every rule broken. An empty list = everything agrees."""
    bad = []
    def fail(name, where, detail): bad.append({"check": name, "where": where, "detail": detail})
    teams = {t["team"]: t for t in stats.get("teams", [])}
    passes = stats.get("passes") or []; seqs = stats.get("sequences") or []; players = stats.get("players") or []
    # --- teams vs their lists ---
    for k, t in teams.items():
        lp = [p for p in passes if p.get("team") == k]
        if t.get("passes") is not None and t["passes"] != len(lp):
            fail("team passes = passes list", k, f"team says {t['passes']}, list has {len(lp)}")
        if lp and t.get("pass_completion_pct") is not None:
            pc = round(100 * sum(bool(p.get("completed")) for p in lp) / len(lp))
            if abs(pc - t["pass_completion_pct"]) > 1: fail("team completion % = passes list", k, f"team {t['pass_completion_pct']}%, list {pc}%")
        if lp and t.get("forward_pass_share_pct") is not None:
            fw = round(100 * sum(p.get("kind") == "forward" for p in lp) / len(lp))
            if abs(fw - t["forward_pass_share_pct"]) > 1: fail("team forward share = passes list", k, f"team {t['forward_pass_share_pct']}%, list {fw}%")
        if lp and t.get("progressive_passes") is not None:
            pr = sum(bool(p.get("progressive")) for p in lp)
            if pr != t["progressive_passes"]: fail("team progressive = passes list", k, f"team {t['progressive_passes']}, list {pr}")
        ls = [s for s in seqs if s.get("team") == k]
        if t.get("sequences") is not None and t["sequences"] != len(ls):
            fail("team sequences = sequences list", k, f"team says {t['sequences']}, list has {len(ls)}")
        if t.get("passes_per_sequence") is not None and t.get("sequences"):
            pps = round(t["passes"] / t["sequences"], 1)
            if abs(pps - t["passes_per_sequence"]) > 0.051: fail("passes per sequence = passes / sequences", k, f"team {t['passes_per_sequence']}, {t['passes']}/{t['sequences']} = {pps}")
        pl = [p for p in players if p.get("team") == k]
        if pl and t.get("distance_m_total_visible") is not None:
            sd = sum(p.get("distance_m") or 0 for p in pl)
            if abs(sd - t["distance_m_total_visible"]) > max(5, 0.01 * sd): fail("team distance = sum of players", k, f"team {t['distance_m_total_visible']} m, players {sd:.0f} m")
        if pl and t.get("pressures_applied") is not None:
            sp = sum(p.get("pressures_applied") or 0 for p in pl)
            if sp != t["pressures_applied"]: fail("team pressures = sum of players", k, f"team {t['pressures_applied']}, players {sp}")
        if pl and t.get("better_option_count") is not None:
            sb = sum(p.get("better_option_count") or 0 for p in pl)
            if sb > t["better_option_count"]: fail("players' better-option <= team", k, f"players {sb} > team {t['better_option_count']}")
        if pl and t.get("passes") is not None:
            sp = sum(p.get("passes") or 0 for p in pl)
            if sp > t["passes"]: fail("players' passes <= team passes", k, f"players {sp} > team {t['passes']}")
        for f in ("pressed_within_2s_pct", "regained_within_5s_pct", "forward_within_3s_pct", "lost_back_5s_pct", "pass_completion_pct", "forward_pass_share_pct", "possession_pct"):
            v = t.get(f)
            if v is not None and not (0 <= v <= 100): fail("share between 0 and 100", f"{k}.{f}", str(v))
        if t.get("duration_s") and t.get("possession_s") and t["possession_s"] > t["duration_s"] + 1:
            fail("possession time <= match time", k, f"{t['possession_s']} s > {t['duration_s']} s")
    # --- team A vs team B mirrors ---
    if "A" in teams and "B" in teams:
        A, B = teams["A"], teams["B"]
        if A.get("possession_pct") is not None and not _pct_ok(A["possession_pct"], B.get("possession_pct")):
            fail("possession A + B = 100%", "A+B", f"{A['possession_pct']} + {B.get('possession_pct')}")
        if A.get("possession_s") and B.get("possession_s") and A.get("possession_pct") is not None:
            ps = round(100 * A["possession_s"] / (A["possession_s"] + B["possession_s"]))
            if abs(ps - A["possession_pct"]) > 1: fail("possession % = possession seconds", "A", f"{A['possession_pct']}% vs seconds {ps}%")
        for x, y in (("A", "B"), ("B", "A")):
            if teams[x].get("losses") is not None and teams[y].get("recoveries") is not None and teams[x]["losses"] != teams[y]["recoveries"]:
                fail("balls lost by one = balls won by the other", f"{x} lost / {y} won", f"{teams[x]['losses']} vs {teams[y]['recoveries']}")
    # --- per player ---
    for p in players:
        w = f"player {p.get('id')} ({p.get('team')})"; n = p.get("passes") or 0
        if (p.get("passes_completed") or 0) > n: fail("completed <= passes", w, f"{p['passes_completed']} > {n}")
        kinds = sum(p.get(f) or 0 for f in ("passes_forward", "passes_sideways", "passes_backward"))
        if kinds != n: fail("forward + sideways + backward = passes", w, f"{kinds} vs {n}")
        if (p.get("progressive_passes") or 0) > (p.get("passes_forward") or 0): fail("progressive <= forward", w, f"{p['progressive_passes']} > {p.get('passes_forward')}")
        if (p.get("passes_risky") or 0) > n: fail("risky <= passes", w, f"{p['passes_risky']} > {n}")
        q = p.get("pass_quality") or {}
        if sum(q.values()) > n: fail("pass quality counts <= passes", w, f"{sum(q.values())} > {n}")
        th = [p.get(f) for f in ("pct_def_third", "pct_mid_third", "pct_att_third")]
        if all(v is not None for v in th) and (p.get("time_visible_s") or 0) > 0 and abs(sum(th) - 100) > 2: fail("thirds add to 100%", w, f"{th}")
        if (p.get("high_intensity_m") or 0) > (p.get("distance_m") or 0) + 0.5: fail("high-intensity <= distance", w, f"{p['high_intensity_m']} > {p.get('distance_m')}")
        if (p.get("time_on_ball_s") or 0) > (p.get("time_visible_s") or 0) + 0.5: fail("time on ball <= time seen", w, f"{p['time_on_ball_s']} > {p.get('time_visible_s')}")
        if (p.get("time_under_pressure_s") or 0) > (p.get("time_on_ball_s") or 0) + 0.5: fail("time under pressure <= time on ball", w, f"{p['time_under_pressure_s']} > {p.get('time_on_ball_s')}")
        if n and not (p.get("touches") or 0) and not (p.get("time_on_ball_s") or 0): fail("a passer has touches", w, f"{n} passes, 0 touches")
        cr = p.get("counterpress_rate")
        if cr is not None and not (0 <= cr <= 1): fail("counterpress rate 0..1", w, str(cr))
        if (p.get("top_speed_ms") or 0) > 12: fail("top speed <= 12 m/s", w, str(p["top_speed_ms"]))
    # --- periods / duration ---
    for per in stats.get("periods") or []:
        if isinstance(per, dict) and per.get("start_s") is not None and per.get("end_s") is not None and per["end_s"] <= per["start_s"]:
            fail("period ends after it starts", str(per), "")
    # --- export copy (match_data) vs stats ---
    if md is not None:
        ms, ss = md.get("sequences"), seqs
        if isinstance(ms, list) and len(ms) != len(ss): fail("match_data sequences = stats sequences", "", f"{len(ms)} vs {len(ss)}")
        mr, sr = md.get("restarts"), stats.get("restarts")
        if isinstance(mr, list) and isinstance(sr, list) and len(mr) != len(sr): fail("match_data restarts = stats restarts", "", f"{len(mr)} vs {len(sr)}")
        dur = stats.get("duration_s")
        if dur:
            late = [e for e in md.get("events") or [] if (e.get("t") or 0) > dur + 1 or (e.get("t") or 0) < 0]
            if late: fail("events inside the match time", "", f"{len(late)} events outside 0-{dur} s")
    return bad


def main(out):
    res = {}
    for pth in sorted(glob.glob("results/volume/runs/matches/*/stats.json")):
        mid = pth.split("/")[-2]; st = json.load(open(pth))
        mp = os.path.join(os.path.dirname(pth), "match_data.json"); md = json.load(open(mp)) if os.path.exists(mp) else None
        b = check(st, md); by = collections.Counter(x["check"] for x in b)
        res[mid] = {"broken": len(b), "by_check": dict(by), "examples": [x for i, x in enumerate(b) if [y["check"] for y in b[:i]].count(x["check"]) < 3]}
        print(mid, "broken", len(b)); [print("  ", k, v) for k, v in by.most_common()]
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/qa/n0-01b/consistency.json")
