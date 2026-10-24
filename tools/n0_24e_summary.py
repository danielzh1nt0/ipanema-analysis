"""N0-24e: summary of results/qa/n0-24e/metrica_restarts.json per variant x ball -> summary.json (sums over the 4 halves, seeds averaged)"""
import json, sys, collections
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-24e/metrica_restarts.json"))
agg = collections.defaultdict(lambda: collections.Counter()); seeds = collections.defaultdict(set)
for r in rows:
    k = (r["variant"], r["noise"]); seeds[k].add(r["seed"])
    for f in ("truth", "truth_out_of_play", "ours", "found", "found_out_of_play", "extra", "kind_right_out_of_play", "team_right"): agg[k][f] += r[f]
    agg[k]["minutes"] += r["minutes"]
out = {}
for (v, nz), c in sorted(agg.items()):
    s = len(seeds[(v, nz)]); c = {f: x / s for f, x in c.items()}
    out[f"{v} / {nz}"] = {"found": f"{c['found']:.0f}/{c['truth']:.0f}", "found_out_of_play": f"{c['found_out_of_play']:.0f}/{c['truth_out_of_play']:.0f}",
                          "extra": round(c["extra"]), "ours_per_90": round(c["ours"] / c["minutes"] * 90), "truth_per_90": round(c["truth"] / c["minutes"] * 90),
                          "real_share_pct": round(100 * c["found"] / max(1, c["ours"])), "kind_right_of_found_out_of_play_pct": round(100 * c["kind_right_out_of_play"] / max(1, c["found_out_of_play"])),
                          "team_right_of_found_pct": round(100 * c["team_right"] / max(1, c["found"]))}
for k, v in out.items(): print(f"{k:32s} {v}")
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "results/qa/n0-24e/summary.json", "w"), indent=1)
