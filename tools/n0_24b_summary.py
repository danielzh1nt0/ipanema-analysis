"""N0-24b: summary tables from results/qa/n0-24b/metrica_hightov*.json -> stdout (markdown)"""
import json, sys
from collections import defaultdict


def summary(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["noise"], r.get("before_s"))
        for tm, t in r["teams"].items():
            agg[key]["tilt_err"].append(t["ours_tilt_pct"] - t["truth_tilt_pct"])
            for f in ("ours_high", "truth_high", "truth_high_kept", "ours_wins", "truth_wins", "truth_wins_kept"): agg[key][f].append(t[f])
        for m in ("matched", "matched_kept"):
            for k, v in r[m].items(): agg[key][f"{m}.{k}"].append(v)
        agg[key]["dir_ok"].append(r["direction_same"])
    out = {}
    for key, a in agg.items():
        e = a["tilt_err"]; s = {k: sum(v) for k, v in a.items() if k not in ("tilt_err", "dir_ok")}
        s.update({"tilt_err_min": min(e), "tilt_err_mean": round(sum(e) / len(e), 1), "tilt_err_max": max(e), "n_team_halves": len(e), "dir_ok": all(a["dir_ok"])})
        out[f"{key[0]}|{key[1]}"] = s
    return out


if __name__ == "__main__":
    rows = sum((json.load(open(f)) for f in sys.argv[1:]), [])
    for k, s in summary(rows).items(): print(k, json.dumps(s))
