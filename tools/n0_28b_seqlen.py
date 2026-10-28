"""N0-28b: are the app's sequence length and metres gained right? Every stats.sequences[] row carries duration_s and
gain_m; the app shows them on each 'sequence ended' event ("12.4 s · gain 18.0 m") and the Ball tab's sequence-length
card draws the duration spread. Never checked on pro data (S6 / N0-25e checked the count, reach and end labels).

Truth (Metrica's labelled events + clean tracked ball): the chains of tools/n0_25e_sequences.truth_seqs; duration =
first event start .. last event end; gain = clean ball x at the chain's end minus at its start, in the team's attacking
direction (nearest frame with a ball within 1 s).
Ours = the pipeline (tools/metricalab.ours) on Metrica's positions, perfect ball and our 'streak' / 'flicker' ball noise
(2 seeds each). P.sequences is re-run on the same inputs with each variant (VARIANTS): old; split = cut at the restart
step's stoppages (P.split_at_stops); split_edge = split + gain from the median ball of the first / last 0.5 s.
Matching: each true sequence -> nearest unused one of ours, same team, start within 2 s (n0_25e_sequences.match).

    PYTHONPATH=. python tools/n0_28b_seqlen.py <metrica sample-data/data dir> [--out results/qa/n0-28b/metrica_seqlen.json]"""
import sys, os, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P
from n0_24b_hightov import truth_direction
from n0_25e_sequences import truth_seqs, match, stop_frames

L, W, FPS = 105.0, 68.0, 25.0
VARIANTS = {"old": dict(split_stop=False, gain="exact"), "split": dict(split_stop=True, gain="exact"),
            "split_edge": dict(split_stop=True, gain="edge0.5")}


def ball_near(fr, k, win=int(FPS)):
    for d in range(win + 1):
        for kk in (k - d, k + d):
            b = fr.get(kk, (0, 0, None))[2]
            if b is not None: return b
    return None


def truth_gain(fr, s, right):
    b0, b1 = ball_near(fr, s["start"]), ball_near(fr, s["end"])
    if b0 is None or b1 is None: return None
    return (b1[0] - b0[0]) * (1 if right else -1)


def merge_short(T, max_s=1.0):
    """truth as a 'possession' (StatsBomb-like): a chain of the other team shorter than max_s that did not end out of
    play (a deflection / a touch it never controlled) does not end the possession around it"""
    out = []
    for t in T:
        t = dict(t)
        if len(out) >= 2 and out[-1]["team"] != t["team"] and out[-2]["team"] == t["team"] and out[-1]["dur"] < max_s \
                and not out[-1]["out"] and not out[-2]["out"]:
            short = out.pop(); prev = out[-1]; prev["end"] = t["end"]; prev["out"] = t["out"]; prev["dur"] = (prev["end"] - prev["start"] + 1) / FPS
            continue
        out.append(t)
    return out


def across(ours, SF, k0):
    """share of our sequences with a real stoppage (ball out, foul, set piece) more than 1 s inside both ends"""
    return round(100 * sum(1 for o in ours if ((SF >= o["start"] + k0 + FPS) & (SF <= o["end"] + k0 - FPS)).any()) / max(1, len(ours)), 1)


def swallowed(T, ours, k0, min_s=2.0):
    """share of our sequences that contain a whole true chain of the OTHER team lasting >= min_s"""
    n = 0
    for o in ours:
        a, b = o["start"] + k0, o["end"] + k0
        if any(t["team"] != o["team"] and t["start"] >= a and t["end"] <= b and t["dur"] >= min_s for t in T): n += 1
    return round(100 * n / max(1, len(ours)), 1)


def ours_run(fr, per, noisy, seed):
    import metricalab as ML
    got = {}; o_seq = P.sequences
    def seq_(*a, **k): got["a"] = a; got["k"] = k; return o_seq(*a, **k)
    P.sequences = seq_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.sequences = o_seq
    a, k = got["a"], got["k"]
    return {m: P.sequences(*a, **k, **v) for m, v in VARIANTS.items()}, o["attack_right"]


def q(v, p): return round(float(np.percentile(v, p)), 2) if len(v) else None


def score(T, ours, pairs):
    td = [t["dur"] for t in T]; od = [o["duration_s"] for o in ours]
    r = {"truth": len(T), "ours": len(ours), "matched": len(pairs),
         "dur_truth_q": [q(td, 25), q(td, 50), q(td, 75), q(td, 90)], "dur_ours_q": [q(od, 25), q(od, 50), q(od, 75), q(od, 90)],
         "dur_truth_mean": round(float(np.mean(td)), 2), "dur_ours_mean": round(float(np.mean(od)), 2) if od else None}
    de = [ours[j]["duration_s"] - T[i]["dur"] for i, j in pairs]
    r["dur_err_matched_median"] = q(de, 50); r["dur_abs_err_matched_median"] = q(np.abs(de), 50) if de else None
    r["dur_within_2s_pct"] = round(100 * float(np.mean(np.abs(de) <= 2)), 1) if de else None
    gp = [(T[i]["gain"], ours[j]["gain_m"]) for i, j in pairs if T[i]["gain"] is not None]
    r["gain_missing_pct"] = round(100 * sum(1 for o in ours if o["gain_m"] is None) / max(1, len(ours)), 1)
    g2 = [(a, b) for a, b in gp if b is not None]
    if g2:
        e = np.array([b - a for a, b in g2]); r["gain_abs_err_median"] = q(np.abs(e), 50); r["gain_abs_err_p90"] = q(np.abs(e), 90)
        r["gain_within_5m_pct"] = round(100 * float(np.mean(np.abs(e) <= 5)), 1)
        r["gain_sign_right_pct"] = round(100 * float(np.mean([(a >= 0) == (b >= 0) for a, b in g2 if abs(a) >= 5])), 1)
    tg = [t["gain"] for t in T if t["gain"] is not None]; og = [o["gain_m"] for o in ours if o["gain_m"] is not None]
    r["gain_truth_q"] = [q(tg, 25), q(tg, 50), q(tg, 75)]; r["gain_ours_q"] = [q(og, 25), q(og, 50), q(og, 75)]
    return r


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        ev = list(csv.DictReader(open(f"{gd}/{g}_RawEventsData.csv")))
        for per in halves:
            k0 = min(k for k in fr if fr[k][0] == per); ar_t = truth_direction(fr, per); T = truth_seqs(ev, fr, per, ar_t)
            for t in T: t["dur"] = (t["end"] - t["start"] + 1) / FPS; t["gain"] = truth_gain(fr, t, ar_t[t["team"]])
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    runs, ar = ours_run(fr, per, noisy, seed)
                    T2 = merge_short(T); SF = np.array(stop_frames(ev, per))
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "direction_same": ar == ar_t,
                           **{m: dict(score(T, v, match(T, v, k0)), merged_truth=score(T2, v, match(T2, v, k0)), swallow_pct=swallowed(T, v, k0),
                                      across_stop_pct=across(v, SF, k0)) for m, v in runs.items()}}
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-28b/metrica_seqlen.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); json.dump(rows, open(out, "w"), indent=1); print("->", out)
