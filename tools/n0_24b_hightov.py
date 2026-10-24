"""N0-24b: are the app's high turnovers and field tilt right? Never checked on pro data before.

  metrics.high_turnovers / high_turnover_counts (metrics.high_turnovers): every ball won (the pipeline's turnovers,
      after S9's confirm_losses) gets a zone from the ball's spot at the winning frame in the WINNER's attacking
      direction: high = his final third, mid, low. The app's 'High turnovers' count = zone high.
  metrics.field[tm].field_tilt_pct (metrics.field_tilt_and_entries): share of the team's controlled frames (raw state,
      as run.py passes it) with the ball in its final third.

Truth (Metrica's labelled events + tracking):
  ball won = a RECOVERY by a team after the other team's on-ball event (PASS / CARRY / RECOVERY / SHOT / SET PIECE /
      BALL LOST), with no BALL OUT / SET PIECE in between (open play); zone from the tracked ball at the event frame
      (event Start X if the ball is missing); attack direction from each team's mean x at the half's first frame.
  field tilt = per frame the team of the last on-ball event (as metricalab.truth's possession), dead time (ball out ->
      restart) left out, ball from the tracking -> share of the team's frames with the ball in its final third.
Ours = the pipeline (tools/metricalab.ours: possession_simple, spell state, turnovers, passes, confirm_losses) on
Metrica's positions, perfect ball and our 'streak' / 'flicker' ball noise (2 seeds each). Per truth ball won: nearest
our ball won by the same team within 2 s -> found?, same zone?

    PYTHONPATH=. python tools/n0_24b_hightov.py <metrica sample-data/data dir> [--out results/qa/n0-24b/metrica_hightov.json]
        [--before pipe,0.5,0.2]   lab: the turnover rule's 'loser had it >= X s' (pipe = the pipeline's 1 s)
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, metrics as M

L, W, FPS = 105.0, 68.0, 25.0
ON_BALL = ("PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST")


def zone(x, attacks_right):
    xx = x if attacks_right else L - x
    return "high" if xx > 2 * L / 3 else "mid" if xx > L / 3 else "low"


def truth_direction(fr, per):
    """team attacks right if its players' mean x at the half's first frame is in the left half"""
    k = min(k for k in fr if fr[k][0] == per); ps = fr[k][1]
    ma = np.mean([x for pid, x, y in ps if pid < 200]); mb = np.mean([x for pid, x, y in ps if pid >= 200])
    return {"A": bool(ma < mb), "B": bool(mb < ma)}


def truth_wins(ev_rows, fr, per, ar, keep_s=3.0, before_s=1.0):
    """-> [{team (winner), f (Metrica frame), x_m, zone, kept}] open-play balls won. A win = the first on-ball event
    of a team's chain that directly follows the other team's chain (no ball out / set piece between). kept = what the
    pipeline's rules can count: the loser's chain lasted >= before_s and the winner's lasts >= keep_s or has a pass
    (turnovers' 1 s holds + S9's confirm_losses)."""
    E = sorted([r for r in ev_rows if int(r["Period"]) == per], key=lambda r: int(r["Start Frame"]))
    chains = []; cur = None
    for r in E:
        ty = r["Type"]
        if ty == "BALL OUT" or ty == "SET PIECE":
            if cur: chains.append(cur); cur = None
            if ty == "BALL OUT": chains.append(None); continue          # None = break: next chain is not a ball won
            chains.append(None)
        if ty not in ON_BALL or r["Team"] not in ("Home", "Away"): continue
        tm = "A" if r["Team"] == "Home" else "B"; f0 = int(r["Start Frame"]); f1 = max(f0, int(r["End Frame"] or 0))
        if cur and cur["team"] == tm: cur["f1"] = max(cur["f1"], f1); cur["pass"] |= ty == "PASS"; continue
        if cur: chains.append(cur)
        cur = {"team": tm, "f0": f0, "f1": f1, "pass": ty == "PASS", "first": r}
    if cur: chains.append(cur)
    for i, c in enumerate(chains):                                       # a chain lasts until the next chain starts
        if c is None: continue
        nxt = chains[i + 1] if i + 1 < len(chains) else None
        c["until"] = max(c["f1"], nxt["f0"]) if nxt is not None else c["f1"]
    out = []
    for a, b in zip(chains, chains[1:]):
        if a is None or b is None or a["team"] == b["team"]: continue
        f = b["f0"]; r = b["first"]; bb = fr.get(f, (None, None, None))[2]
        x = bb[0] if bb is not None else (float(r["Start X"]) * L if r["Start X"] not in ("NaN", "") else None)
        if x is None: continue
        kept = (a["until"] - a["f0"]) >= before_s * FPS and ((b["until"] - b["f0"]) >= keep_s * FPS or b["pass"])
        out.append({"team": b["team"], "f": f, "x_m": round(x, 1), "zone": zone(x, ar[b["team"]]), "type": r["Type"], "kept": bool(kept)})
    return out


def truth_tilt(ev_rows, fr, per, ar):
    E = sorted([r for r in ev_rows if int(r["Period"]) == per], key=lambda r: int(r["Start Frame"]))
    ks = sorted(k for k in fr if fr[k][0] == per); f1 = ks[-1]
    own = {}; cur = None; dead = False
    for a, b in zip(E, E[1:] + [None]):
        ty = a["Type"]
        if ty == "BALL OUT": dead = True; cur = None
        elif ty in ON_BALL and a["Team"] in ("Home", "Away"): dead = False; cur = "A" if a["Team"] == "Home" else "B"
        end = int(b["Start Frame"]) if b is not None else f1 + 1
        if cur and not dead:
            for k in range(int(a["Start Frame"]), min(end, f1 + 1)): own[k] = cur
    res = {}
    for tm in ("A", "B"):
        ctrl = [k for k, t in own.items() if t == tm and fr.get(k, (0, 0, None))[2] is not None]
        fin = sum(1 for k in ctrl if zone(fr[k][2][0], ar[tm]) == "high")
        res[tm] = round(100 * fin / len(ctrl)) if ctrl else None
    return res


def ours_run(fr, per, noisy, seed, before_s=None):
    """the pipeline via metricalab.ours, capturing the raw state, ball metres and the final (S9) turnovers.
    before_s (lab only): override turnovers()' 'loser had it >= min_before_s' rule (None = the pipeline's 1 s)"""
    import metricalab as ML
    got = {}; o_ps, o_cl, o_car, o_tv = P.pipeline_state, P.confirm_losses, P.carriers, P.turnovers
    if before_s is not None: P.turnovers = lambda *a, **k: o_tv(*a, **{**k, "min_before_s": before_s})
    def ps_(*a, **k): r = o_ps(*a, **k); got["state"] = r[0]; got["bspeed"] = r[1]; return r
    def cl_(*a, **k): r = o_cl(*a, **k); got["tvs"] = r[0]; return r
    def car_(*a, **k): r = o_car(*a, **k); got["ballm"] = r[1]; return r
    P.pipeline_state, P.confirm_losses, P.carriers = ps_, cl_, car_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.pipeline_state, P.confirm_losses, P.carriers, P.turnovers = o_ps, o_cl, o_car, o_tv
    ar = o["attack_right"]
    ht = M.high_turnovers(got["tvs"], got["ballm"], L, ar, FPS)
    for h, t in zip([h for h in ht], [t for t in got["tvs"] if got["ballm"].get(t["frame"]) is not None]): h["frame"] = t["frame"]
    field = M.field_tilt_and_entries(got["state"], got["ballm"], got["bspeed"], FPS, L, ar, *M.ENTRY_REARM)
    return ht, field, ar


def match(tw, ht, k0, tol_s=2.0):
    used = set(); rows = []
    for t in sorted(tw, key=lambda t: t["f"]):
        best = None
        for j, h in enumerate(ht):
            if j in used or h["team"] != t["team"]: continue
            dt = abs(k0 + h["frame"] - t["f"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is not None: used.add(best[1]); rows.append((t["zone"], ht[best[1]]["zone"]))
        else: rows.append((t["zone"], None))
    th = [r for r in rows if r[0] == "high"]
    oh = [h for h in ht if h["zone"] == "high"]
    oh_real = sum(1 for j, h in enumerate(ht) if h["zone"] == "high" and j in used and
                  any(abs(k0 + h["frame"] - t["f"]) <= tol_s * FPS and t["team"] == h["team"] and t["zone"] == "high" for t in tw))
    return {"truth_wins": len(tw), "found": sum(1 for r in rows if r[1] is not None), "zone_same": sum(1 for r in rows if r[1] is not None and r[0] == r[1]),
            "truth_high": len(th), "truth_high_found_high": sum(1 for r in th if r[1] == "high"), "ours_high": len(oh), "ours_high_real": oh_real}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1), befores=(None,)):
    import csv, metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        ev = list(csv.DictReader(open(f"{gd}/{g}_RawEventsData.csv")))
        for per in halves:
            k0 = min(k for k in fr if fr[k][0] == per); ar_t = truth_direction(fr, per)
            tw = truth_wins(ev, fr, per, ar_t); tt = truth_tilt(ev, fr, per, ar_t)
            for noisy in noises:
                for seed, bs in [(s_, b_) for s_ in (seeds if noisy else (0,)) for b_ in befores]:
                    ht, field, ar = ours_run(fr, per, noisy, seed, bs)
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "before_s": bs, "direction_same": ar == ar_t, "teams": {}}
                    for tm in ("A", "B"):
                        row["teams"][tm] = {"truth_high": sum(1 for t in tw if t["team"] == tm and t["zone"] == "high"), "ours_high": sum(1 for h in ht if h["team"] == tm and h["zone"] == "high"),
                                            "truth_wins": sum(1 for t in tw if t["team"] == tm), "ours_wins": sum(1 for h in ht if h["team"] == tm),
                                            "truth_tilt_pct": tt[tm], "ours_tilt_pct": field[tm]["field_tilt_pct"]}
                    row["matched"] = match(tw, ht, k0); row["matched_kept"] = match([t for t in tw if t["kept"]], ht, k0)
                    for tm in ("A", "B"): row["teams"][tm]["truth_high_kept"] = sum(1 for t in tw if t["team"] == tm and t["zone"] == "high" and t["kept"]); row["teams"][tm]["truth_wins_kept"] = sum(1 for t in tw if t["team"] == tm and t["kept"])
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-24b/metrica_hightov.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bef = tuple(None if b == "pipe" else float(b) for b in a[a.index("--before") + 1].split(",")) if "--before" in a else (None,)
    rows = main(a[0], befores=bef); json.dump(rows, open(out, "w"), indent=1)
