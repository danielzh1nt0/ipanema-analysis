"""GitHub runner: one ball-detector round on Modal from Daniel's clicks; results + exam pictures committed; report issue."""
import os, sys, json, time, base64, traceback, modal
tag = os.environ.get("TAG", time.strftime("%Y%m%d_%H%M")); OUT = f"results/ball/{tag}"; os.makedirs(f"{OUT}/exam", exist_ok=True)
def log(m): line = f"{time.strftime('%H:%M:%S')} {m}"; print(line, flush=True); open(f"{OUT}/log.txt", "a").write(line + "\n")
try:
    res = modal.Function.from_name("ipanema", "ball_round").remote("SFKBP1109", 60)
    if "error" in res: log(res["error"]); sys.exit(1)
    for l in res["log"]: log(l)
    for k, v in res["pictures"].items(): open(f"{OUT}/exam/{k}", "wb").write(base64.b64decode(v))
    s = res["summary"]; json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1); n, o, fz = s["new"], s["old"], s.get("new_farzoom")
    g = (f"ball exam (40 frames Daniel clicked, never trained on): NEW {n['correct']}/{n['of']} correct "
         f"(ball found {n['ball_found']}/{n['ball_frames']}, no-ball right {n['no_ball_right']}/{n['no_ball_frames']}); "
         + (f"NEW + far zoom {fz['correct']}/{fz['of']} (ball found {fz['ball_found']}/{fz['ball_frames']}); " if fz else "") + f"OLD {o['correct']}/{o['of']} (ball found {o['ball_found']}/{o['ball_frames']})")
    open("/tmp/issue_title", "w").write(f"Ipanema ball round {tag}: new {n['correct']}/{n['of']} vs old {o['correct']}/{o['of']}"); open("/tmp/issue_body.md", "w").write(f"**{g}**\n\nPictures: `results/ball/{tag}/exam` (red = old detector, green = new, yellow ring = Daniel's click).\n"); log(g)
except SystemExit: raise
except Exception: log("FAILED:\n" + traceback.format_exc()[-4000:]); sys.exit(1)
