"""GitHub runner: the full-match calibration on Modal, results committed, report issue opened."""
import os, sys, json, time, base64, traceback, modal
tag = os.environ.get("TAG", time.strftime("%Y%m%d_%H%M")); OUT = f"results/match/{tag}"; os.makedirs(OUT, exist_ok=True)
def log(m): line = f"{time.strftime('%H:%M:%S')} {m}"; print(line, flush=True); open(f"{OUT}/log.txt", "a").write(line + "\n")
try:
    trig = open("triggers/match.txt").read().split(); match_id = next((w for w in trig if w.startswith("match=")), "match=SFKBP1109")[6:]
    if "venue" in trig:                                                        # new ground: solve the camera base from lines first
        log(f"venue base for {match_id} on Modal"); vb = modal.Function.from_name("ipanema", "venue_base").remote(match_id, 24)
        if "error" in vb: log(vb["error"]); sys.exit(1)
        for l in vb["log"]: log(l)
        os.makedirs(f"{OUT}/venue", exist_ok=True); json.dump(vb["report"], open(f"{OUT}/venue/report.json", "w"), indent=1); open(f"{OUT}/venue/strip.jpg", "wb").write(base64.b64decode(vb["strip_b64"]))
        if not vb["report"]["accepted"]: log("venue base rejected: stopping before the full match"); open("/tmp/issue_title", "w").write(f"Ipanema {match_id}: venue camera base REJECTED, match not run"); open("/tmp/issue_body.md", "w").write(f"Held-out check: {vb['report']['heldout']}\nStrip: results/match/{tag}/venue/strip.jpg\n"); sys.exit(0)
    log(f"full-match calibration on Modal: {match_id}"); res = modal.Function.from_name("ipanema", "match_calibration").remote(match_id, 12, 1.0)
    if "error" in res: log(res["error"]); sys.exit(1)
    for l in res["log"]: log(l)
    os.makedirs(f"{OUT}/misses", exist_ok=True)
    for g_ in res["grades"]:
        if g_.get("picture"): open(f"{OUT}/misses/t{g_['t']:.1f}_{g_['median_px']:.0f}px{'_confident' if g_['confident'] else ''}.jpg", "wb").write(bytes.fromhex(g_.pop("picture")))
        else: g_.pop("picture", None)
    json.dump({"rows": res["rows"], "grades": res["grades"]}, open(f"{OUT}/rows.json", "w"))
    json.dump(res["summary"], open(f"{OUT}/summary.json", "w"), indent=1); open(f"{OUT}/strip.jpg", "wb").write(base64.b64decode(res["strip_b64"]))
    os.makedirs(f"{OUT}/checkpoints", exist_ok=True)
    for t, b64 in res.get("pictures", {}).items(): open(f"{OUT}/checkpoints/t{float(t):07.1f}.jpg", "wb").write(base64.b64decode(b64))
    s = res["summary"]
    g = (f"{match_id}: {s['confident_share'] * 100:.0f}% of the match placed confidently ({s['confident']}/{s['frames']} seconds); "
         + (f"{s['checkpoints_within_10px']}/{s['checkpoints']} of Daniel's clicked moments within 10 px (median {s['checkpoint_median_px']:.1f} px); {s['checkpoints_confident_and_wrong']} confident-but-wrong; " if s['checkpoints'] else "no click checkpoints for this match; ")
         + f"longest unsure stretch {s['longest_unsure_s']:.0f} s")
    open("/tmp/issue_title", "w").write(f"Ipanema full match {tag}: {g[:150]}")
    open("/tmp/issue_body.md", "w").write(f"**{g}**\n\nHard moments (fast pan / zoom): {s['hard_confident']}/{s['hard_moments']} confident. Unsure stretches over 10 s: {s['unsure_stretches_over_10s']}.\n\nLook at `results/match/{tag}/strip.jpg` (20 pictures across the match, lines drawn, verdict written on each).\n")
    log(g)
except SystemExit: raise
except Exception: log("FAILED:\n" + traceback.format_exc()[-4000:]); sys.exit(1)
