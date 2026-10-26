"""N0-26a: summary table of results/qa/n0-26a/metrica_pressmap.json -> summary.json + printed table.
    python tools/n0_26a_summary.py [results/qa/n0-26a/metrica_pressmap.json]"""
import sys, os, json, numpy as np


def summarise(rows):
    out = {}
    for rule in sorted({r["rule"] for r in rows}):
        for noise in ("False", "streak", "flicker"):
            R = [r for r in rows if r["rule"] == rule and r["noise"] == noise]
            if not R: continue
            ratios = [t["ratio"] for r in R for t in r["teams"].values() if t["ratio"] is not None]
            out[f"{rule}/{noise}"] = {
                "runs": len(R),
                "count_ratio_range": [min(ratios), max(ratios)],
                "count_ratio_median": round(float(np.median(ratios)), 2),
                "ours_real_pct": round(100 * sum(r["matched"] for r in R) / max(1, sum(r["points_ours"] for r in R)), 1),
                "truth_found_pct": round(100 * sum(r["matched"] for r in R) / max(1, sum(r["points_truth"] for r in R)), 1),
                "block_top_right": f'{sum(r["block_top_right"] for r in R)}/{sum(r["block_top_n"] for r in R)}',
                "block_err_median_pct": round(float(np.median([r["block_err_median_pct"] for r in R])), 1),
                "thirds_off_pts_median": round(float(np.median([t["thirds_off_pts"] for r in R for t in r["teams"].values() if t["thirds_off_pts"] is not None])), 1),
                "grid_misplaced_median_pct": round(float(np.median([t["grid_misplaced_pct"] for r in R for t in r["teams"].values()])), 1),
                "more_pressing_right": f'{sum(r["more_pressing_right"] for r in R)}/{len(R)}',
            }
    return out


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-26a/metrica_pressmap.json"
    s = summarise(json.load(open(p)))
    json.dump(s, open(os.path.join(os.path.dirname(p), "summary.json"), "w"), indent=1)
    keys = list(next(iter(s.values())).keys())
    print("rule/noise".ljust(16), *[k[:18].ljust(18) for k in keys])
    for k, v in s.items(): print(k.ljust(16), *[str(v[x]).ljust(18) for x in keys])
