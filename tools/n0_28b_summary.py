"""N0-28b: table from results/qa/n0-28b/metrica_seqlen.json (per noise: sums / medians over halves and seeds)"""
import json, sys, numpy as np
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-28b/metrica_seqlen.json"))
for noise in ("False", "streak", "flicker"):
    rs = [r for r in R if r["noise"] == noise]; print(f"== ball: {'perfect' if noise == 'False' else noise} ({len(rs)} runs)")
    for v in ("old", "split", "split_edge"):
        x = [r[v] for r in rs]; m = [r[v]["merged_truth"] for r in rs]
        f = lambda k, xs=x: round(float(np.median([y[k] for y in xs if y.get(k) is not None])), 1)
        print(f"  {v:10} count ours/truth {sum(y['ours'] for y in x)}/{sum(y['truth'] for y in x)}  median s ours {f('dur_ours_q') if False else round(float(np.median([y['dur_ours_q'][1] for y in x])),1)}"
              f" truth {round(float(np.median([y['dur_truth_q'][1] for y in x])),1)} (poss. {round(float(np.median([y['dur_truth_q'][1] for y in m])),1)})"
              f"  p90 ours {round(float(np.median([y['dur_ours_q'][3] for y in x])),1)} truth {round(float(np.median([y['dur_truth_q'][3] for y in x])),1)}"
              f"  mean ours {f('dur_ours_mean')} truth {f('dur_truth_mean')}  matched |err| {f('dur_abs_err_matched_median')} s, within 2 s {f('dur_within_2s_pct')}% (poss. {f('dur_within_2s_pct', m)}%)"
              f"  across stop {f('across_stop_pct')}%  swallow {f('swallow_pct')}%"
              f"  gain |err| {f('gain_abs_err_median')} m, within 5 m {f('gain_within_5m_pct')}%, sign {f('gain_sign_right_pct')}%, missing {f('gain_missing_pct')}%")
