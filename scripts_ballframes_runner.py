"""GitHub runner: pick the ball frames to click (Modal, one GPU pass), commit them to the repo."""
import os, sys, json, time, base64, traceback, modal
tag = os.environ.get("TAG", time.strftime("%Y%m%d_%H%M")); OUT = f"results/ballclicks/{tag}"; os.makedirs(OUT, exist_ok=True)
def log(m): line = f"{time.strftime('%H:%M:%S')} {m}"; print(line, flush=True); open(f"{OUT}/log.txt", "a").write(line + "\n")
try:
    res = modal.Function.from_name("ipanema", "ball_hard_frames").remote("SFKBP1109", 200, 1.0)
    if "error" in res: log(res["error"]); sys.exit(1)
    meta = []
    for r in res["frames"]:
        name = f"t{r['t']:07.1f}.jpg"; open(f"{OUT}/{name}", "wb").write(base64.b64decode(r.pop("jpg"))); meta.append(dict(r, file=name))
    json.dump({"frames": meta, "counts": res["counts"], "sampled": res["sampled"]}, open(f"{OUT}/frames.json", "w"), indent=0)
    g = f"{len(meta)} ball frames to click picked from {res['sampled']} sampled seconds; detector groups over the match: {res['counts']}"
    open("/tmp/issue_title", "w").write(f"Ipanema ball frames {tag}: {len(meta)} frames ready to click"); open("/tmp/issue_body.md", "w").write(g + "\n"); log(g)
except SystemExit: raise
except Exception: log("FAILED:\n" + traceback.format_exc()[-4000:]); sys.exit(1)
