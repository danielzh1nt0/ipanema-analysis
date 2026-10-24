"""N0-24d summary of results/qa/n0-24d/metrica_ppda.json -> summary.json (+ printed table)

    python tools/n0_24d_summary.py [results/qa/n0-24d/metrica_ppda.json]
"""
import sys, json, numpy as np
from collections import defaultdict


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return round(float(np.corrcoef(ra, rb)[0, 1]), 2)


def summarise(rows):
    by = defaultdict(lambda: defaultdict(list))                        # noise -> (game, half, team) -> rows
    for r in rows:
        for tm, t in r["teams"].items(): by[r["noise"]][(r["game"], r["half"], tm)].append(t)
    out = {}
    for noise, d in by.items():
        keys = sorted(d); m = lambda f: [float(np.mean([t[f] for t in d[k] if t.get(f) is not None])) for k in keys]
        pr, tpr = m("ours_pressures"), m("truth_pressures"); op, top = m("ours_opp_passes"), m("truth_opp_passes")
        pp, tsame, tstd, tstrict = m("ours_ppda"), m("truth_ppda_same_rule"), m("truth_ppda_standard"), m("truth_ppda_strict")
        res = {"team_halves": len(keys),
               "pressures_ratio": [round(min(a / b for a, b in zip(pr, tpr)), 2), round(float(np.median([a / b for a, b in zip(pr, tpr)])), 2), round(max(a / b for a, b in zip(pr, tpr)), 2)],
               "opp_passes_ratio": [round(min(a / b for a, b in zip(op, top)), 2), round(float(np.median([a / b for a, b in zip(op, top)])), 2), round(max(a / b for a, b in zip(op, top)), 2)],
               "ppda_ours": [round(min(pp), 1), round(float(np.median(pp)), 1), round(max(pp), 1)],
               "ppda_truth_same_rule": [round(min(tsame), 1), round(float(np.median(tsame)), 1), round(max(tsame), 1)],
               "ppda_truth_standard": [round(min(tstd), 1), round(float(np.median(tstd)), 1), round(max(tstd), 1)],
               "ppda_truth_strict": [round(min(tstrict), 1), round(float(np.median(tstrict)), 1), round(max(tstrict), 1)],
               "spearman_ours_vs_same_rule": spearman(pp, tsame), "spearman_ours_vs_standard": spearman(pp, tstd), "spearman_ours_vs_strict": spearman(pp, tstrict),
               "spearman_same_rule_vs_standard": spearman(tsame, tstd)}
        # within a half: does ours pick the team that presses more (lower PPDA) like the standard figure?
        halves = sorted({(g, h) for g, h, _ in keys}); agree = 0; agree_same = 0
        for g, h in halves:
            ia, ib = keys.index((g, h, "A")), keys.index((g, h, "B"))
            agree += (pp[ia] < pp[ib]) == (tstd[ia] < tstd[ib]); agree_same += (pp[ia] < pp[ib]) == (tsame[ia] < tsame[ib])
        pagree = sum((pr[keys.index((g, h, "A"))] > pr[keys.index((g, h, "B"))]) == (tpr[keys.index((g, h, "A"))] > tpr[keys.index((g, h, "B"))]) for g, h in halves)
        res["which_team_more_pressures_vs_same_rule"] = f"{pagree}/{len(halves)}"
        res["which_team_presses_more_vs_standard"] = f"{agree}/{len(halves)}"; res["which_team_presses_more_vs_same_rule"] = f"{agree_same}/{len(halves)}"
        if all("ours_ppda_zone" in t for k in keys for t in d[k]):
            pz = m("ours_ppda_zone"); prz, tprz = m("ours_pressures_zone"), m("truth_pressures_zone")
            res["lab_zone"] = {"ppda_ours": [round(min(pz), 1), round(float(np.median(pz)), 1), round(max(pz), 1)],
                               "pressures_zone_ratio_median": round(float(np.median([a / b for a, b in zip(prz, tprz)])), 2),
                               "spearman_vs_standard": spearman(pz, tstd), "spearman_vs_strict": spearman(pz, tstrict),
                               "which_team_presses_more_vs_standard": f"{sum((pz[keys.index((g, h, 'A'))] < pz[keys.index((g, h, 'B'))]) == (tstd[keys.index((g, h, 'A'))] < tstd[keys.index((g, h, 'B'))]) for g, h in halves)}/{len(halves)}"}
        res["per_team_half"] = [{"key": "-".join(map(str, k)), "ours_ppda": round(a, 1), "truth_same_rule": round(b, 1), "truth_standard": round(c, 1),
                                 "ours_pressures": round(e, 1), "truth_pressures": round(f, 1)} for k, a, b, c, e, f in zip(keys, pp, tsame, tstd, pr, tpr)]
        out[noise] = res
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-24d/metrica_ppda.json"
    s = summarise(json.load(open(src))); json.dump(s, open(src.replace("metrica_ppda.json", "summary.json"), "w"), indent=1)
    for k, v in s.items(): print(k, json.dumps({a: b for a, b in v.items() if a != "per_team_half"}))
