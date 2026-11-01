"""N0-01d (1 Nov): do the app's PLACES and the labels built from them agree? N0-01b checked sums; this checks every
spot, length and place-label in the export against the pitch and against each other, on the repo's app match copies
(results/volume/runs/matches/<id>/stats.json). No model, no video - only the exported numbers.

Rules (each must hold whatever the video shows):
  - a located event lies on the pitch or within MARGIN_M of it (restarts, balls won, pressure points, shots, passes,
    runs, final-third entries, players' average spots)
  - a ball won's zone (low / mid / high) matches its x and the team's attack direction; high_turnover_counts = highs
  - a restart's kind matches its spot (corner at a corner, goal kick in the 6-m area, throw-in at a touchline)
  - a shot's distance = distance from its spot to the goal it attacks
  - a pass's length = |to - from|, gain = x gain towards the goal, kind from the gain (+-3 m), progressive = gain >= 10
    and completed
  - a run's distance = |to - from|, speed = distance / its time
  - a sequence's duration = t_end - t_start
  - entries_count / by pass / by carry = the entries list; field tilt 0..100
  - each heat map sums to 1 over the grid's cells
  - pass network: an edge count never exceeds the team's passes between those two players

    python tools/n0_01d_places.py [--out results/qa/n0-01d/places.json]
"""
import sys, os, json, glob, math, collections

MARGIN_M = 4.0        # same margin as N0-31e's final-third entries


def _off(x, y, L, W):
    if x is None or y is None: return None
    return max(0.0, -x, x - L, -y, y - W)


def check(stats, margin=MARGIN_M):
    """Returns a list of {check, where, detail} for every rule broken."""
    bad = []
    def fail(name, where, detail): bad.append({"check": name, "where": where, "detail": detail})
    L = (stats.get("pitch") or {}).get("length", 105.0); W = (stats.get("pitch") or {}).get("width", 68.0)
    ar = stats.get("attack_right") or {"A": True, "B": False}
    mx = stats.get("metrics") or {}
    def onpitch(name, where, x, y):
        o = _off(x, y, L, W)
        if o is not None and o > margin: fail(f"{name} on the pitch", where, f"({x}, {y}) is {o:.1f} m off a {L:g} x {W:g} pitch")
    # --- restarts ---
    for r in stats.get("restarts") or []:
        w = f"t={r.get('t')} {r.get('kind')}"; x, y = r.get("x_m"), r.get("y_m")
        onpitch("restart", w, x, y)
        if x is None or y is None: continue
        k = r.get("kind"); near_end = x < 3 or x > L - 3; near_side = y < 3 or y > W - 3
        if k == "corner" and not (near_end and near_side): fail("restart kind matches its spot", w, f"corner at ({x}, {y})")
        if k == "throw-in" and not near_side: fail("restart kind matches its spot", w, f"throw-in at ({x}, {y})")
        if k == "goal kick" and not ((x < 6.5 or x > L - 6.5) and abs(y - W / 2) < 12.5): fail("restart kind matches its spot", w, f"goal kick at ({x}, {y})")
    # --- balls won (high turnovers) ---
    hts = mx.get("high_turnovers") or []; highs = collections.Counter()
    for r in hts:
        w = f"t={r.get('t')} {r.get('team')}"; x, y = r.get("x_m"), r.get("y_m"); onpitch("ball won", w, x, y)
        if x is None or r.get("team") not in ar: continue
        xa = x if ar[r["team"]] else L - x
        z = "high" if xa > 2 * L / 3 else "mid" if xa > L / 3 else "low"
        if r.get("zone") != z and min(abs(xa - L / 3), abs(xa - 2 * L / 3)) > 0.2: fail("ball won zone matches its x", w, f"zone {r.get('zone')} at x {x} (attack right {ar[r['team']]})")
        if r.get("zone") == "high": highs[r["team"]] += 1
    htc = mx.get("high_turnover_counts") or {}
    for k, v in htc.items():
        if hts and v != highs.get(k, 0): fail("high_turnover_counts = high balls won", k, f"count {v}, list {highs.get(k, 0)}")
    # --- pressure points ---
    for r in mx.get("pressure_points") or []: onpitch("pressure point", f"t={r.get('t')}", r.get("x_m"), r.get("y_m"))
    # --- shots ---
    for nm in ("shots_detected", "shots"):
        for s in mx.get(nm) or []:
            x, y = s.get("x_m"), s.get("y_m"); w = f"{nm} t={s.get('t')} {s.get('team')}"
            if x is None or y is None: continue
            onpitch("shot", w, x, y)
            if s.get("distance_m") is not None and s.get("team") in ar:
                gx = L if ar[s["team"]] else 0.0; d = math.hypot(gx - x, W / 2 - y)
                if abs(d - s["distance_m"]) > 0.6: fail("shot distance = spot to goal", w, f"says {s['distance_m']} m, spot is {d:.1f} m from the goal")
    # --- passes ---
    for p in stats.get("passes") or []:
        w = f"t={p.get('t')} {p.get('team')}"; a, b = p.get("from_m"), p.get("to_m")
        if not a or not b: continue
        onpitch("pass start", w, *a); onpitch("pass end", w, *b)
        ln = math.dist(a, b)
        if p.get("length_m") is not None and abs(ln - p["length_m"]) > 0.25: fail("pass length = |to - from|", w, f"says {p['length_m']}, spots give {ln:.1f}")
        if p.get("team") in ar and p.get("gain_m") is not None:
            g = (b[0] - a[0]) * (1 if ar[p["team"]] else -1)
            if abs(g - p["gain_m"]) > 0.25: fail("pass gain = x towards goal", w, f"says {p['gain_m']}, spots give {g:.1f}")
            gm = p["gain_m"]; kind = "forward" if gm > 3 else "backward" if gm < -3 else "sideways"
            if p.get("kind") and p["kind"] != kind and abs(abs(gm) - 3) > 0.06:   # gain is rounded to 0.1: 3.0 can be 3.04
                fail("pass kind matches its gain", w, f"{p['kind']} with gain {gm}")
            if p.get("progressive") is not None and bool(p["progressive"]) != (gm >= 10 and bool(p.get("completed"))) and abs(gm - 10) > 0.06:
                fail("progressive = gain >= 10 m and completed", w, f"progressive {p['progressive']}, gain {gm}, completed {p.get('completed')}")
    # --- runs ---
    for r in mx.get("runs") or []:
        w = f"t={r.get('t')} {r.get('player')}"; a, b = r.get("from_m"), r.get("to_m")
        if not a or not b: continue
        onpitch("run start", w, *a); onpitch("run end", w, *b)
        d = math.dist(a, b)
        if r.get("distance_m") is not None and abs(d - r["distance_m"]) > 0.25: fail("run distance = |to - from|", w, f"says {r['distance_m']}, spots give {d:.1f}")
        dt = (r.get("t_end") or 0) - (r.get("t") or 0)
        if dt > 0 and r.get("speed_ms") is not None and r.get("distance_m") is not None and abs(r["distance_m"] / dt - r["speed_ms"]) > max(0.6, 0.15 * r["speed_ms"]):
            fail("run speed = distance / time", w, f"says {r['speed_ms']} m/s, {r['distance_m']} m in {dt:.2f} s = {r['distance_m'] / dt:.1f}")
    # --- sequences ---
    for q in stats.get("sequences") or []:
        if None not in (q.get("t_start"), q.get("t_end"), q.get("duration_s")) and abs((q["t_end"] - q["t_start"]) - q["duration_s"]) > 0.15:
            fail("sequence duration = end - start", f"t={q['t_start']}", f"says {q['duration_s']}, times give {q['t_end'] - q['t_start']:.2f}")
    # --- field: entries, tilt ---
    for k, f in (mx.get("field") or {}).items():
        es = f.get("final_third_entries") or []
        if f.get("entries_count") is not None and f["entries_count"] != len(es): fail("entries_count = entries list", k, f"{f['entries_count']} vs {len(es)}")
        for how in ("pass", "carry"):
            c = f.get(f"entries_by_{how}")
            if c is not None and c != sum(e.get("how") == how for e in es): fail("entries by pass / carry = entries list", k, f"{how} {c} vs {sum(e.get('how') == how for e in es)}")
        for e in es:
            if e.get("y_m") is not None and (e["y_m"] < -margin or e["y_m"] > W + margin): fail("entry on the pitch", f"{k} t={e.get('t')}", f"y {e['y_m']} on a {W:g} m wide pitch")
        if f.get("field_tilt_pct") is not None and not 0 <= f["field_tilt_pct"] <= 100: fail("field tilt 0..100", k, str(f["field_tilt_pct"]))
    # --- players' average spots, heat maps ---
    for p in stats.get("players") or []:
        onpitch("player average spot", str(p.get("id")), p.get("avg_x_m"), p.get("avg_y_m"))
    for pid, h in (stats.get("heatmaps") or {}).items():
        tot = sum(sum(r) for r in h)
        if tot and abs(tot - 1) > 0.01: fail("heat map sums to 1", str(pid), f"{tot:.3f}")
    # --- pass network vs passes ---
    pc = collections.Counter((p.get("team"), p.get("from"), p.get("to")) for p in stats.get("passes") or [] if p.get("completed"))
    pa = collections.Counter((p.get("team"), p.get("from"), p.get("to")) for p in stats.get("passes") or [])
    for k, net in (mx.get("pass_network") or {}).items():
        for e in (net or {}).get("edges") or []:
            if e.get("count", 0) > pa.get((k, e.get("from"), e.get("to")), 0):
                fail("pass network edge <= passes between the two", k, f"{e.get('from')}->{e.get('to')} edge {e.get('count')}, passes {pa.get((k, e.get('from'), e.get('to')), 0)} (completed {pc.get((k, e.get('from'), e.get('to')), 0)})")
    return bad


def main(out):
    res = {}
    for pth in sorted(glob.glob("results/volume/runs/matches/*/stats.json")):
        mid = pth.split("/")[-2]; st = json.load(open(pth))
        b = check(st); by = collections.Counter(x["check"] for x in b)
        res[mid] = {"broken": len(b), "by_check": dict(by), "examples": [x for i, x in enumerate(b) if [y["check"] for y in b[:i]].count(x["check"]) < 4]}
        print(mid, "broken", len(b)); [print("  ", k, v) for k, v in by.most_common()]
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    return res


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/qa/n0-01d/places.json")
