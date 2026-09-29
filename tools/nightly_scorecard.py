"""nightly: combine the parallel results into results/nightly/<date>.md + history.tsv"""
import json, glob, os, time
d = time.strftime("%Y-%m-%d"); q = json.load(open("results/nightly/quick.json")) if os.path.exists("results/nightly/quick.json") else {}
lines = [f"# Nightly check {d}", "", "| Check | Result |", "|---|---|"]
t = q.get("tests", {}); lines.append(f"| Tests | {t.get('total', '?') - len(t.get('failed', []))}/{t.get('total', '?')} pass" + (f" — FAILED: {', '.join(t['failed'])}" if t.get("failed") else "") + " |")
b = q.get("ball_34_moments", {}); lines.append(f"| Ball, 34 checked moments | {b.get('right', '?')}/{b.get('of', '?')} (best possible {b.get('best_possible', '?')}) |")
s = q.get("stats_pro_data", {}); lines.append(f"| Stats on pro data (worst half) | passes {s.get('worst_pass_err_pct', '?')}% off, possession {s.get('worst_poss_err_pts', '?')} pts off |")
if "sequences_found_pct" in s: lines.append(f"| Sequences on pro data (all halves) | {s['sequences_found_pct']}% of real ones found, {s['sequences_real_pct']}% of ours real |")
if s.get("possession_model") == "simple": lines.append("| Note | From 29 Sep the pro-data stats use the pipeline's possession model (possession_simple), not the old one; numbers before that date are not comparable (E5) |")
for f in sorted(glob.glob("results/qa/tracktest*/summary.json")):
    r = json.load(open(f)); v = r.get("new, RF-DETR")
    if v: lines.append(f"| Players, {f.split('/')[2].replace('tracktest_', '').replace('tracktest', 'SFKBP1109')[:30]} | seen/frame dark {v['observed_per_frame']['dark']} light {v['observed_per_frame']['light']}, tracked {v['median_track_s']['dark']}/{v['median_track_s']['light']} s |")
os.makedirs("results/nightly", exist_ok=True); open(f"results/nightly/{d}.md", "w").write("\n".join(lines) + "\n")
h = "results/nightly/history.tsv"; new = not os.path.exists(h)
with open(h, "a") as fh:
    if new: fh.write("date\ttests_failed\tball_right\tball_of\tpass_err\tposs_err\n")
    fh.write(f"{d}\t{len(t.get('failed', []))}\t{b.get('right', '')}\t{b.get('of', '')}\t{s.get('worst_pass_err_pct', '')}\t{s.get('worst_poss_err_pts', '')}\n")
print("\n".join(lines))
