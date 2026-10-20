"""N0-20d: how often a per-track team vote is a tie (old rule = chance, set order), replayed on the saved tracking rows of the
free-runner checks (results/qa/**/rows_all.json.gz: each row's own colour reading per frame). For each piece: player-frames
whose team was a coin flip, and team changes along tracks under the old rule (both possible set orders) vs the new one (tie -> latest reading)
and a tried alternative (tie -> keep the track's team so far; worse by eye, not used).
Approximate: frames a track was skipped (off pitch / referee) are not in the rows, so their readings are missing.
PYTHONPATH=. python tools/n0_20d_ties.py [out_dir]"""
import gzip, json, glob, os, sys
from ipanema.vote import team_vote


def old_vote(v, first):
    """old rule with a fixed set order: on an A/B tie the label listed first wins (Python's set order, i.e. chance)"""
    ab = [x for x in v if x in ("A", "B")]
    if (v.count("K") >= 0.6 * len(v) or not ab) and "K" in v: return "K"
    if not ab: return max(sorted(set(v)), key=v.count)
    a, b = ab.count("A"), ab.count("B")
    return "A" if a > b else "B" if b > a else first


def replay(R, T):
    votes, prev, last = {}, {}, {"A": {}, "B": {}, "new": {}, "keep": {}}
    c = {"rows": 0, "tie_rows": 0, "differ_rows": 0, "flips_old_A": 0, "flips_old_B": 0, "flips_new": 0, "flips_keep_prev": 0, "tracks": set(), "tie_tracks": set()}
    for k in sorted(R, key=int):
        for r, raw in zip(R[k], T.get(k, [None] * len(R[k]))):
            tid = r[0]; v = votes.setdefault(tid, []); c["tracks"].add(tid)
            if raw is not None: v.append(raw); del v[:-25]
            if not v: continue
            res = {"A": old_vote(v, "A"), "B": old_vote(v, "B"), "new": team_vote(v, 0.6), "keep": team_vote(v, 0.6, prev=prev.get(tid))}
            prev[tid] = res["keep"]; c["rows"] += 1
            if res["A"] != res["B"]: c["tie_rows"] += 1; c["tie_tracks"].add(tid)
            if res["new"] not in (res["A"], res["B"]): c["differ_rows"] += 1          # must stay 0: new = one of the old outcomes
            for key, name in (("A", "flips_old_A"), ("B", "flips_old_B"), ("new", "flips_new"), ("keep", "flips_keep_prev")):
                if tid in last[key] and last[key][tid] != res[key]: c[name] += 1
                last[key][tid] = res[key]
    c["tracks"], c["tie_tracks"] = len(c["tracks"]), len(c["tie_tracks"]); return c


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-20d"; os.makedirs(out, exist_ok=True); rep = {}
    for f in sorted(glob.glob("results/qa/**/rows_all.json.gz", recursive=True)):
        d = json.load(gzip.open(f, "rt"))
        if "raw_team" not in d: continue
        for lab in d["rows"]:
            rep[f"{os.path.relpath(os.path.dirname(f), 'results/qa')} | {lab}"] = replay(d["rows"][lab], d["raw_team"].get(lab, {}))
    tot = {k: sum(r[k] for r in rep.values()) for k in next(iter(rep.values()))}
    rep["TOTAL"] = tot; json.dump(rep, open(f"{out}/ties.json", "w"), indent=1)
    for k, r in rep.items(): print(f"{k[:70]:70s} rows {r['rows']:6d} tie {r['tie_rows']:4d} ({100 * r['tie_rows'] / max(1, r['rows']):.1f}%) tracks {r['tie_tracks']}/{r['tracks']} flips old {r['flips_old_A']}/{r['flips_old_B']} new {r['flips_new']} keep-prev {r['flips_keep_prev']} differ {r['differ_rows']}")
