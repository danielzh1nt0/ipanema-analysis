"""N0-29e: the team rows' balls lost / balls won and passes per sequence (stats.json teams[].losses, recoveries,
passes_per_sequence) - never checked on pro data as the app computes them today.

What the app writes (analytics.stats): losses = the kept turnovers (P.confirm_losses, S9: the winner keeps it 3 s or
plays a pass) lost by the team, recoveries = the same turnovers won by it (so losses A = recoveries B by build);
passes_per_sequence = all passes tried by the team / its exported sequences (P.sequences on the spell state).

Truth (Metrica's 2 pro games, 4 halves, hand-labelled events):
  loss_all - every BALL LOST of the team (open play; END HALF / OFFSIDE / REFEREE HIT left out)
  loss_s9  - those where the winner's next chain lasts >= 3 s or has a pass (the S9 rule the app applies)
  recovery_metrica - Metrica's own RECOVERY rows (context only: they include loose balls after saves / blocks)
  passes per sequence - pass tries (PASS + failed passes, n0_24d_ppda.is_pass_try) / truth chains of the team
                        (metricalab.truth_sequences)
Ours = analytics.stats on Metrica's positions as the pipeline sees them (n0_24d_ppda.ours_run), perfect ball and our
'flicker' (ball dot hops to a nearby opponent, like our clips) and 'streak' noise, 2 seeds.
Matching: a truth loss and one of ours of the same team within 2 s = found.

    PYTHONPATH=. python tools/n0_29e_lossrec.py /tmp/metrica/data [--out results/qa/n0-29e/metrica_lossrec.json]
    PYTHONPATH=. python tools/n0_29e_lossrec.py --dry
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FPS = 25.0
SKIP_SUB = ("END HALF", "OFFSIDE", "REFEREE HIT")
TOL_S = 2.0


def truth_half(ev_rows, per):
    """-> dict of truth counts and loss frames for one half"""
    import metricalab as ML, n0_24d_ppda as PP
    ev = PP.ML_events(ev_rows); seqs = ML.truth_sequences(ev, per)
    R = [r for r in ev_rows if int(r["Period"]) == per and r["Team"] in ("Home", "Away")]
    tm = lambda r: "A" if r["Team"] == "Home" else "B"
    losses = []
    for r in R:
        if r["Type"] != "BALL LOST" or r["Subtype"] in SKIP_SUB: continue
        f = int(r["Start Frame"]); lo = tm(r); wi = "B" if lo == "A" else "A"
        nxt = next((s for s in seqs if s["start"] >= f - FPS and s["team"] == wi), None)
        wl = (nxt["end"] - nxt["start"]) / FPS if nxt else 0.0
        wp = bool(nxt and any(x["Type"] == "PASS" and tm(x) == wi and nxt["start"] <= int(x["Start Frame"]) <= nxt["end"] for x in R))
        losses.append({"f": f, "lost_by": lo, "s9": wl >= 3.0 or wp})
    out = {"loss_frames": [(l["f"], l["lost_by"]) for l in losses], "loss_s9_frames": [(l["f"], l["lost_by"]) for l in losses if l["s9"]]}
    for t in ("A", "B"):
        tries = sum(1 for r in R if tm(r) == t and PP.is_pass_try(r) and r["Subtype"] not in SKIP_SUB); ch = sum(1 for s in seqs if s["team"] == t)
        out[t] = {"loss_all": sum(1 for l in losses if l["lost_by"] == t), "loss_s9": sum(1 for l in losses if l["lost_by"] == t and l["s9"]),
                  "recovery_metrica": sum(1 for r in R if tm(r) == t and r["Type"] == "RECOVERY"),
                  "pass_tries": tries, "chains": ch, "passes_per_sequence": round(tries / ch, 2) if ch else None}
    for t in ("A", "B"):
        o = "B" if t == "A" else "A"; out[t]["won_all"] = out[o]["loss_all"]; out[t]["won_s9"] = out[o]["loss_s9"]
    return out


def match(truth_frames, ours_frames, tol=TOL_S * FPS):
    """greedy nearest match, same loser within tol frames -> (found truth, ours matched)"""
    used = set(); hit = 0
    for f, lo in sorted(truth_frames):
        best = None
        for j, (g, lo2) in enumerate(ours_frames):
            if j in used or lo2 != lo: continue
            d = abs(g - f)
            if d <= tol and (best is None or d < best[0]): best = (d, j)
        if best: used.add(best[1]); hit += 1
    return hit, len(used)


def ours_half(fr, per, noisy, seed):
    import n0_24d_ppda as PP
    st, got = PP.ours_run(fr, per, noisy, seed)
    k0 = min(k for k in fr if fr[k][0] == per)
    T = {t["team"]: t for t in st["teams"]}
    lf = [(k0 + int(t["frame_lost"]), t["lost_by"]) for t in got["tvs"]]
    return {t: {"losses": T[t]["losses"], "recoveries": T[t]["recoveries"], "passes": T[t]["passes"], "sequences": T[t]["sequences"],
                "passes_per_sequence": T[t]["passes_per_sequence"]} for t in ("A", "B")} | {"loss_frames": lf}


def score(tr, ou):
    """one half: per team ratios + matching"""
    row = {}
    for t in ("A", "B"):
        a, b = tr[t], ou[t]
        row[t] = {"truth": {k: a[k] for k in ("loss_all", "loss_s9", "won_all", "won_s9", "recovery_metrica", "pass_tries", "chains", "passes_per_sequence")},
                  "ours": {k: b[k] for k in ("losses", "recoveries", "passes", "sequences", "passes_per_sequence")}}
    h_all, m_all = match(tr["loss_frames"], ou["loss_frames"]); h_s9, m_s9 = match(tr["loss_s9_frames"], ou["loss_frames"])
    n = len(ou["loss_frames"])
    row["match"] = {"ours": n, "truth_all": len(tr["loss_frames"]), "truth_s9": len(tr["loss_s9_frames"]),
                    "found_all_pct": round(100 * h_all / max(1, len(tr["loss_frames"]))), "found_s9_pct": round(100 * h_s9 / max(1, len(tr["loss_s9_frames"]))),
                    "ours_real_pct": round(100 * m_all / max(1, n))}
    return row


def summarise(rows):
    """rows of one noise model -> mean ratios / errors"""
    rl_all, rl_s9, ppsq_err, ppsq_ratio, more_right = [], [], [], [], []
    for r in rows:
        for t in ("A", "B"):
            a, b = r[t]["truth"], r[t]["ours"]
            rl_all.append(b["losses"] / max(1, a["loss_all"])); rl_s9.append(b["losses"] / max(1, a["loss_s9"]))
            if a["passes_per_sequence"] and b["passes_per_sequence"] is not None:
                ppsq_ratio.append(b["passes_per_sequence"] / a["passes_per_sequence"]); ppsq_err.append(abs(b["passes_per_sequence"] - a["passes_per_sequence"]))
        ta, tb = r["A"]["truth"]["loss_s9"], r["B"]["truth"]["loss_s9"]; oa, ob = r["A"]["ours"]["losses"], r["B"]["ours"]["losses"]
        if ta != tb: more_right.append((oa > ob) == (ta > tb))
    m = lambda v: round(float(np.mean(v)), 2) if v else None
    return {"halves": len(rows), "losses_vs_all_x": m(rl_all), "losses_vs_all_range": [round(min(rl_all), 2), round(max(rl_all), 2)],
            "losses_vs_s9_x": m(rl_s9), "losses_vs_s9_range": [round(min(rl_s9), 2), round(max(rl_s9), 2)],
            "found_all_pct": m([r["match"]["found_all_pct"] for r in rows]), "found_s9_pct": m([r["match"]["found_s9_pct"] for r in rows]),
            "ours_real_pct": m([r["match"]["ours_real_pct"] for r in rows]),
            "team_losing_more_right": f"{sum(more_right)}/{len(more_right)}",
            "passes_per_sequence_x": m(ppsq_ratio), "passes_per_sequence_abs_err": m(ppsq_err),
            "truth_pps_mean": m([r[t]["truth"]["passes_per_sequence"] for r in rows for t in "AB"]),
            "ours_pps_mean": m([r[t]["ours"]["passes_per_sequence"] for r in rows for t in "AB" if r[t]["ours"]["passes_per_sequence"] is not None])}


def main(d, out):
    import n0_24d_ppda as PP
    rows = {}; tcache = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr, ev = PP.load_game(d, g)
        for per in (1, 2):
            tr = truth_half(ev, per); tcache[f"{g[-1]}-{per}"] = {t: tr[t] for t in "AB"}
            for noisy in (False, "flicker", "streak"):
                for seed in ((0, 1) if noisy else (0,)):
                    r = score(tr, ours_half(fr, per, noisy, seed)); r["half"] = f"{g[-1]}-{per}"; r["seed"] = seed
                    rows.setdefault(str(noisy), []).append(r); print(g, per, noisy, seed, json.dumps(r["match"]), flush=True)
    S = {nz: summarise(R) for nz, R in rows.items()}
    print(json.dumps(S, indent=1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": S, "truth": tcache, "rows": rows}, open(out, "w"), indent=1)
    return S


def dry():
    """stand-in: hand-made truth / ours rows through score + summarise"""
    tr = {"loss_frames": [(100, "A"), (500, "B"), (900, "A")], "loss_s9_frames": [(100, "A"), (500, "B")],
          "A": {"loss_all": 2, "loss_s9": 1, "won_all": 1, "won_s9": 1, "recovery_metrica": 1, "pass_tries": 6, "chains": 3, "passes_per_sequence": 2.0},
          "B": {"loss_all": 1, "loss_s9": 1, "won_all": 2, "won_s9": 1, "recovery_metrica": 2, "pass_tries": 4, "chains": 2, "passes_per_sequence": 2.0}}
    ou = {"loss_frames": [(110, "A"), (530, "B"), (2000, "B")],
          "A": {"losses": 1, "recoveries": 2, "passes": 5, "sequences": 4, "passes_per_sequence": 1.2},
          "B": {"losses": 2, "recoveries": 1, "passes": 4, "sequences": 2, "passes_per_sequence": 2.0}}
    r = score(tr, ou); return r, summarise([r])


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--dry" in a:
        r, S = dry(); print(json.dumps(r["match"]), json.dumps(S)); sys.exit(0)
    main(a[0], a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-29e/metrica_lossrec.json")
