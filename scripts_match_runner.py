"""GitHub runner: the full-match calibration on Modal, results committed, report issue opened."""
import os, sys, json, time, base64, traceback, modal
tag = os.environ.get("TAG", time.strftime("%Y%m%d_%H%M")); OUT = f"results/match/{tag}"; os.makedirs(OUT, exist_ok=True)
def log(m): line = f"{time.strftime('%H:%M:%S')} {m}"; print(line, flush=True); open(f"{OUT}/log.txt", "a").write(line + "\n")
try:
    log("full-match calibration on Modal"); res = modal.Function.from_name("ipanema", "match_calibration").remote("SFKBP1109", 12, 1.0)
    if "error" in res: log(res["error"]); sys.exit(1)
    for l in res["log"]: log(l)
    json.dump({"rows": res["rows"], "grades": res["grades"]}, open(f"{OUT}/rows.json", "w"))
    json.dump(res["summary"], open(f"{OUT}/summary.json", "w"), indent=1); open(f"{OUT}/strip.jpg", "wb").write(base64.b64decode(res["strip_b64"]))
    s = res["summary"]
    g = (f"{s['confident_share'] * 100:.0f}% of the match placed confidently ({s['confident']}/{s['frames']} seconds); "
         f"{s['checkpoints_within_10px']}/{s['checkpoints']} of Daniel's clicked moments within 10 px (median {s['checkpoint_median_px']:.1f} px); "
         f"{s['checkpoints_confident_and_wrong']} confident-but-wrong; longest unsure stretch {s['longest_unsure_s']:.0f} s")
    open("/tmp/issue_title", "w").write(f"Ipanema full match {tag}: {g[:150]}")
    open("/tmp/issue_body.md", "w").write(f"**{g}**\n\nHard moments (fast pan / zoom): {s['hard_confident']}/{s['hard_moments']} confident. Unsure stretches over 10 s: {s['unsure_stretches_over_10s']}.\n\nLook at `results/match/{tag}/strip.jpg` (20 pictures across the match, lines drawn, verdict written on each).\n")
    log(g)
except SystemExit: raise
except Exception: log("FAILED:\n" + traceback.format_exc()[-4000:]); sys.exit(1)
