"""N0-31b: shot spots vs the attack direction.

For every Veo shot / goal in a match's stats.json that has a clicked spot (x_m, y_m), check:
  - the spot is on the pitch (-1 .. L+1, -1 .. W+1),
  - it lies in the half the shooting team attacks in that period (periods[].attack_right; the export
    already mirrors the second half, so attack_right holds for the whole match),
  - distance_m equals the distance to the centre of that team's target goal (+-0.2 m),
  - distance <= 40 m (a youth shot from further out is almost always a wrong click / wrong team).
A wrong attack direction (like AIK's, N0-26b) shows up as shots in the team's own half.

   PYTHONPATH=. python tools/n0_31b_shotspots.py [match_id ...]   -> results/qa/n0-31b/shotspots.json
"""
import os, sys, json, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES = ["p15u-vs-vallentuna-2026-10-03-6cce", "SFKBP1109", "p15u-vs-aik-2026-09-21-bd09"]


def check(stats, L=None, W=None):
    L = L or (stats.get("pitch") or {}).get("length") or 106.0
    W = W or (stats.get("pitch") or {}).get("width") or 64.0
    periods = stats.get("periods") or []
    ar = (periods[0].get("attack_right") if periods else None) or {}
    rows = []
    for s in (stats.get("metrics") or {}).get("shots") or []:
        x, y, team = s.get("x_m"), s.get("y_m"), s.get("team")
        if x is None or y is None or team not in ar:
            continue
        gx = L if ar[team] else 0.0
        d = math.hypot(x - gx, y - W / 2)
        faults = []
        if not (-1 <= x <= L + 1 and -1 <= y <= W + 1): faults.append("off the pitch")
        if (x > L / 2) != bool(ar[team]): faults.append("in own half")
        if s.get("distance_m") is not None and abs(s["distance_m"] - d) > 0.2: faults.append(f"distance {s['distance_m']} vs {d:.1f}")
        if d > 40: faults.append("over 40 m")
        rows.append({"t": s["t"], "team": team, "goal": bool(s.get("goal")), "x_m": x, "y_m": y,
                     "distance_m": round(d, 1), "faults": faults})
    return rows


def main(ids):
    out = {}
    for m in ids:
        p = os.path.join(REPO, "results/volume/runs/matches", m, "stats.json")
        if not os.path.exists(p):
            continue
        st = json.load(open(p))
        rows = check(st)
        n_shots = len((st.get("metrics") or {}).get("shots") or [])
        out[m] = {"shots": n_shots, "with_spot": len(rows), "faults": [r for r in rows if r["faults"]], "rows": rows}
        print(f"{m}: {n_shots} shots, {len(rows)} with a spot, {len(out[m]['faults'])} with a fault;"
              f" distances {sorted(r['distance_m'] for r in rows)}")
    os.makedirs(os.path.join(REPO, "results/qa/n0-31b"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "results/qa/n0-31b/shotspots.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or MATCHES)
