"""N0-25e: are the app's sequence labels right? stats.sequences[] carries per sequence 'reached_final_third' and
'end_reason' (pass intercepted / dispossessed / loose ball / out of play / stoppage / clip ends). Never checked on pro data.

Truth (Metrica's labelled events + clean tracked ball): chains of one team's on-ball events (metricalab.truth_sequences'
rule); a chain 'ends out of play' when the next event is BALL OUT; it 'reached the final third' when the clean ball is
in the team's final third on any frame of the chain.
Ours = the pipeline (tools/metricalab.ours) on Metrica's positions, perfect ball and our 'streak' / 'flicker' ball noise
(2 seeds each); P.sequences is run twice on the same inputs: old (reach = last frame only, no stoppage end) and new
(N0-25e: reach on any frame, loose-ball end running into a stoppage = 'out of play / stoppage').
Matching: each true sequence -> nearest unused one of ours, same team, start within 2 s.

    PYTHONPATH=. python tools/n0_25e_sequences.py <metrica sample-data/data dir> [--out results/qa/n0-25e/metrica_sequences.json]"""
import sys, os, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P
from n0_24b_hightov import truth_direction, ON_BALL
from n0_25d_tilt_windows import is_foul

L, W, FPS = 105.0, 68.0, 25.0
MINS = (0.0, 0.5, 1.0)   # lab: seconds the ball must spend in the final third for 'reached' (0 = one frame)


def infinal(b, right): return b is not None and ((b[0] > 2 * L / 3) if right else (b[0] < L / 3))


def truth_seqs(ev, fr, per, ar):
    E = sorted([e for e in ev if int(e["Period"]) == per], key=lambda e: (int(e["Start Frame"]), int(e["End Frame"] or 0)))
    out = []; cur = None
    def close(how):
        nonlocal cur
        if cur: cur["out"] = how == "out"; out.append(cur)
        cur = None
    for e in E:
        if e["Type"] == "BALL OUT" or is_foul(e): close("out"); continue                # play stopped (ball out, foul)
        if e["Type"] == "SET PIECE": close("out")                                     # a restart: whatever came before was stopped
        if e["Type"] not in ON_BALL or e["Team"] not in ("Home", "Away"): continue
        tm = "A" if e["Team"] == "Home" else "B"; f0 = int(e["Start Frame"]); f1 = max(f0, int(e["End Frame"] or f0))
        if cur and cur["team"] == tm: cur["end"] = max(cur["end"], f1); continue
        close("other"); cur = {"team": tm, "start": f0, "end": f1}
    close("half")
    for s in out:
        s["reach"] = any(infinal(fr.get(k, (0, 0, None))[2], ar[s["team"]]) for k in range(s["start"], s["end"] + 1))
        s["reach_end"] = infinal(fr.get(s["end"], (0, 0, None))[2], ar[s["team"]])
    return out


def ours_run(fr, per, noisy, seed):
    import metricalab as ML
    got = {}; o_seq = P.sequences
    def seq_(*a, **k): got["a"] = a; got["k"] = k; return o_seq(*a, **k)
    P.sequences = seq_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.sequences = o_seq
    a, k = got["a"], dict(got["k"]); stops = k.pop("stops", None)
    old = P.sequences(*a, **k, reach="end", stop_s=0)
    new = {f"new_min{m:g}": P.sequences(*a, **k, stops=stops, reach="any", stop_s=2.0, reach_min_s=m) for m in MINS}
    return old, new, o["attack_right"]


def match(truth, ours, k0, tol=2 * FPS):
    pairs = sorted((abs(o["start"] + k0 - t["start"]), i, j) for i, t in enumerate(truth) for j, o in enumerate(ours)
                   if o["team"] == t["team"] and abs(o["start"] + k0 - t["start"]) <= tol)
    ut, uo, out = set(), set(), []
    for d, i, j in pairs:
        if i in ut or j in uo: continue
        ut.add(i); uo.add(j); out.append((i, j))
    return out


def stop_frames(ev, per):
    """Metrica frames where play stopped: BALL OUT, fouls, and the start of every SET PIECE"""
    return sorted(int(e["Start Frame"]) for e in ev if int(e["Period"]) == per and (e["Type"] in ("BALL OUT", "SET PIECE") or is_foul(e)))


def near_stop(ours, stops, k0, before=1 * FPS, after=4 * FPS):
    """per end reason: how many of ours end within -1..+4 s of a real stoppage (any sequence, matched or not)"""
    import bisect; out = {}
    for o in ours:
        e = o["end"] + k0; i = bisect.bisect_left(stops, e - before); hit = i < len(stops) and stops[i] <= e + after
        r = out.setdefault(o["end_reason"], [0, 0]); r[0] += 1; r[1] += hit
    return out


def score(truth, ours, pairs):
    """labels on matched pairs + shares over all sequences"""
    r = {"truth": len(truth), "ours": len(ours), "matched": len(pairs)}
    tp = sum(1 for i, j in pairs if truth[i]["reach"] and ours[j]["reached_final_third"])
    r["reach_right"] = sum(1 for i, j in pairs if truth[i]["reach"] == ours[j]["reached_final_third"])
    r["reach_truth_yes"] = sum(1 for i, j in pairs if truth[i]["reach"]); r["reach_found"] = tp
    r["reach_ours_yes"] = sum(1 for i, j in pairs if ours[j]["reached_final_third"])
    ot = sum(1 for i, j in pairs if truth[i]["out"] and ours[j]["end_reason"] == "out of play / stoppage")
    r["out_truth_yes"] = sum(1 for i, j in pairs if truth[i]["out"]); r["out_found"] = ot
    r["out_ours_yes"] = sum(1 for i, j in pairs if ours[j]["end_reason"] == "out of play / stoppage")
    r["share_reach_truth"] = round(100 * sum(t["reach"] for t in truth) / max(1, len(truth)), 1)
    r["share_reach_ours"] = round(100 * sum(o["reached_final_third"] for o in ours) / max(1, len(ours)), 1)
    r["share_out_truth"] = round(100 * sum(t["out"] for t in truth) / max(1, len(truth)), 1)
    r["share_out_ours"] = round(100 * sum(o["end_reason"] == "out of play / stoppage" for o in ours) / max(1, len(ours)), 1)
    return r


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1)):
    import metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        ev = list(csv.DictReader(open(f"{gd}/{g}_RawEventsData.csv")))
        for per in halves:
            k0 = min(k for k in fr if fr[k][0] == per); ar_t = truth_direction(fr, per); T = truth_seqs(ev, fr, per, ar_t); SF = stop_frames(ev, per)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    old, new, ar = ours_run(fr, per, noisy, seed)
                    pairs = match(T, old, k0)       # old and new have the same spans; only labels differ
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "direction_same": ar == ar_t,
                           "truth_reach_at_end_only": round(100 * sum(t["reach_end"] for t in T) / max(1, len(T)), 1),
                           "old": score(T, old, pairs), **{k: score(T, v, pairs) for k, v in new.items()},
                           "ends_near_real_stop": {"old": near_stop(old, SF, k0), "new": near_stop(new["new_min0.5"], SF, k0)}}
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-25e/metrica_sequences.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = main(a[0]); json.dump(rows, open(out, "w"), indent=1); print("->", out)
