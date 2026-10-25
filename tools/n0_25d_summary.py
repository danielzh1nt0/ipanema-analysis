"""N0-25d summary table from results/qa/n0-25d/metrica_tilt_windows.json -> stdout (README rows) + summary.json"""
import json, sys
KEYS = ["poss_window", "poss_roll_app", "poss_roll_weighted", "poss_roll_min2", "poss_bars_app", "poss_bars_weighted", "tilt_window", "tilt_roll_app"]


def main(p="results/qa/n0-25d/metrica_tilt_windows.json"):
    rows = json.load(open(p)); out = {}
    f = lambda xs: [x for x in xs if x is not None]
    for nz in ("False", "streak", "flicker"):
        R = [r for r in rows if r["noise"] == nz]; o = out[nz] = {"runs": len(R), "direction_same": all(r["direction_same"] for r in R),
                                                                     "control_vs_truth": [min(r["control_s_ours_vs_truth"] for r in R), max(r["control_s_ours_vs_truth"] for r in R)]}
        for k in KEYS:
            m = f([r[k]["med_pts"] for r in R]); q = f([r[k]["p90_pts"] for r in R])
            o[k] = {"med_pts": [min(m), max(m)], "p90_pts": [min(q), max(q)], "side_right": sum(r[k]["side_right"] for r in R), "clear": sum(r[k]["clear"] for r in R),
                    "n": sum(r[k]["n"] for r in R), "n_truth": sum(r[k]["n_truth"] for r in R)}
        o["weighted_better_runs"] = sum(1 for r in R if r["poss_roll_weighted"]["p90_pts"] < r["poss_roll_app"]["p90_pts"] and r["poss_roll_weighted"]["side_right"] >= r["poss_roll_app"]["side_right"])
        print(nz, json.dumps(o))
    json.dump(out, open(p.replace("metrica_tilt_windows.json", "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
