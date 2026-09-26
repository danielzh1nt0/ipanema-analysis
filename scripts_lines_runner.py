"""GitHub runner: download the match from R2, cut the frames, run the line-model round, write results into the repo."""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import linerun
tag = os.environ.get("TAG", time.strftime("%Y%m%d_%H%M")); OUT = f"results/lines/{tag}"; LAB = "/tmp/lab"; os.makedirs(OUT, exist_ok=True)
log_path = f"{OUT}/log.txt"
def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True); open(log_path, "a").write(line + "\n")
url = os.environ["R2_PUBLIC_URL"].rstrip("/"); os.makedirs(LAB, exist_ok=True)
NAMES = ("SFKBP1109_frames_s1.zip", "SFKBP1109_frames_s2.zip", "SFKBP1109_random.zip", "SFKBP1109_random.json", "SFKBP1109_random_review.json")
def fetch_frames():
    for n in NAMES:
        r = subprocess.run(["curl", "-sfL", "-o", f"{LAB}/{n}", f"{url}/SFKBP1109/lines/{n}"])
        if r.returncode != 0: return False
    return True
if not fetch_frames():                                                     # not on R2 yet: Modal cuts them once (cents)
    import modal
    log("frames not on R2: asking Modal to cut them from the full match (one-off)")
    out = modal.Function.from_name("ipanema", "export_line_frames").remote("SFKBP1109"); log(str(out)[:500])
    if not fetch_frames(): log("frames still not on R2"); sys.exit(1)
log("frames fetched from R2")
import traceback
def modal_round():                 # GPU round on Modal (~15-20 min, ~$0.30)
    import modal, base64, io, zipfile
    log("training on Modal's GPU")
    res = modal.Function.from_name("ipanema", "lines_round").remote(80, 25)
    zipfile.ZipFile(io.BytesIO(base64.b64decode(res["zip_b64"]))).extractall(OUT)
    for line in open(f"{OUT}/log.txt"): print(line.rstrip())
    return res["summary"]
trig = open("triggers/lines.txt").read().split()
if "venue" in trig:                                                        # new ground round: Edsberg + venue together, two exams
    try:
        import modal, base64, io, zipfile
        match_id = next(w for w in trig if w.startswith("match="))[6:]; log(f"venue round for {match_id} on Modal's GPU")
        res = modal.Function.from_name("ipanema", "venue_round").remote(match_id, 30, 20)
        zipfile.ZipFile(io.BytesIO(base64.b64decode(res["zip_b64"]))).extractall(OUT); s = res["summary"]
        g = f"{match_id}: new ground held-back frames {s['venue']['before']} -> {s['venue']['after']} within 10 px (median {s['venue']['median_px_after']} px); Edsberg {s['edsberg']['before']} -> {s['edsberg']['after']}; {s['minutes']:.0f} min"
        open("/tmp/issue_title", "w").write(f"Ipanema lines {tag} (venue): {g[:120]}"); open("/tmp/issue_body.md", "w").write(f"**{g}**\n\nPictures: `results/lines/{tag}/eval_venue` (new ground), `results/lines/{tag}/eval_edsberg`.\n")
        log(g); sys.exit(0)
    except SystemExit: raise
    except Exception:
        log("VENUE ROUND FAILED:\n" + traceback.format_exc()[-4000:]); sys.exit(1)
if "modal" in open("triggers/lines.txt").read().lower():
    try: s = modal_round()
    except Exception:
        log("MODAL ROUND FAILED:\n" + traceback.format_exc()[-4000:]); sys.exit(1)   # the error lands in the repo log
else:
    s = linerun.run(LAB, OUT, epochs=int(os.environ.get("EPOCHS", "40")), max_minutes=int(os.environ.get("MAX_MIN", "150")), log=log, work="/tmp")
r = s; g = f"{r['placed_correctly']}/{r['held_back_frames']} held-back frames within 10 px of Daniel's clicks, median {r['median_error_px_1280']:.1f} px; base fix {'accepted' if r['base_fix_accepted'] else 'rejected'}; {r['minutes']:.0f} min"
open("/tmp/issue_title", "w").write(f"Ipanema lines {tag}: {g[:120]}")
open("/tmp/issue_body.md", "w").write(f"**{g}**\n\nErrors per frame (px at 1280): {r['errors_px']}\nPolish options, frames within 10 px: {r.get('per_variant_within_10px')}\nPolish options, median px: {r.get('per_variant_median_px')}\nConfident but wrong: {r['confident_but_wrong']}\n\nPictures: `results/lines/{tag}/eval` and `results/lines/{tag}/base_check` in the repo.\n")
log(g)
