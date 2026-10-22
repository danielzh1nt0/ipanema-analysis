"""N0-22b (22 Oct, free runner): the app's follow-cam calibration (ipanema/mosaic.calibrate_via_mosaic) on the two app
test clips, with the anchor frames kept. Which frames are MEASURED (between anchors <= 1 s apart) and which only get a
guess (inside a registration gap / before the first / after the last anchor)? Every 10th frame is judged against the
painted lines (calcheck.judge_frame: good / bad / unjudged), split measured vs guessed. Writes
results/qa/n0-22b/<clip>.json and a picture sheet: guessed frames (worst gaps first) + a few measured ones, the pitch
model drawn in red, caption = frame, measured/guessed, line verdict.
    R2_PUBLIC_URL=... python tools/n0_22b_mosaic_gaps.py
    dry run: LOCAL_CLIP_SFK=a.mp4 LOCAL_CLIP_AIK=b.mp4 python tools/n0_22b_mosaic_gaps.py out_dir"""
import os, sys, json, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, cv2
from ipanema.mosaic import calibrate_via_mosaic, measured_mask
from ipanema.calibration import draw_model
from ipanema import calcheck

SFK, AIK = "SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"
L, W = 120.0, 70.0                                     # the app's pitch model (sports SoccerPitchConfiguration)


def fetch(clip):
    loc = os.environ.get("LOCAL_CLIP_SFK" if clip == SFK else "LOCAL_CLIP_AIK")
    if loc: return loc
    dst = f"/tmp/n0_22b_{clip}.mp4"
    if os.path.exists(dst): return dst
    url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{clip}/video.mp4"; print("downloading", clip, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    return dst


def gaps(anchors, n, step):
    """[(first guessed frame, last guessed frame)] incl. the clip's start / end"""
    ak = sorted(anchors); out = []
    if not ak: return [(0, n - 1)]
    if ak[0] > step: out.append((0, ak[0] - step - 1))
    for a, b in zip(ak, ak[1:]):
        if b - a > 25: out.append((a + 1, b - 1))
    if n - 1 - ak[-1] > step: out.append((ak[-1] + step + 1, n - 1))
    return out


def run_clip(clip, out_dir):
    t0 = time.time(); video = fetch(clip); info = {}
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; cap.release()
    Hs = calibrate_via_mosaic(video, clip, ROOT, code_dir=ROOT, log=lambda m: print(" ", m, flush=True), anchors_out=info)
    if Hs is None: return {"clip": clip, "error": "no calibration", **{k: v for k, v in info.items() if k != "anchors"}}
    n = len(Hs); step = info["step"]; max_gap = max(step, int(round(fps)))
    ok = measured_mask(info["anchors"], n, step=step, max_gap=max_gap)
    gp = [(a, b) for a, b in gaps(info["anchors"], n, step) if not ok[a:b + 1].all()]
    # line verdict every 10th frame
    verdict = {}; support = {}; cap = cv2.VideoCapture(video); k = 0; frames_keep = {}
    want_pics = set()
    for a, b in sorted(gp, key=lambda g: -(g[1] - g[0]))[:6]:          # the 6 longest gaps: start, middle, end
        want_pics.update({a, (a + b) // 2, b})
    meas = [i for i in range(0, n, 10) if ok[i]]
    want_pics.update(meas[len(meas) // 5::max(1, len(meas) // 4)][:4])
    while k < n:
        okr, f = cap.read()
        if not okr: break
        if k % 10 == 0 or k in want_pics:
            j = calcheck.judge_frame(f, Hs[k], L, W); v = j["verdict"]
            if k % 10 == 0: verdict[k] = v; support[k] = j.get("weakest_support")
            if k in want_pics: frames_keep[k] = (f, v)
        k += 1
    cap.release()
    def tally(sel):
        vs = [verdict[i] for i in verdict if ok[i] == sel]; sp = [support[i] for i in verdict if ok[i] == sel and support.get(i) is not None]
        return {"checked": len(vs), "good": vs.count("good"), "bad": vs.count("bad"), "unjudged": vs.count("unjudged"), "median_line_support": round(float(np.median(sp)), 3) if sp else None}
    res = {"clip": clip, "frames": n, "fps": round(fps, 2), "step": step, "max_gap": max_gap, "anchors": len(info["anchors"]),
           "direct": info["direct"], "chained": info["chained"], "measured": int(ok.sum()), "guessed": int((~ok).sum()),
           "measured_pct": round(100 * float(ok.mean()), 1), "old_count": f"{n}/{n} frames registered",
           "gaps": [{"from": a, "to": b, "s": round((b - a + 1) / fps, 1)} for a, b in gp],
           "lines_measured": tally(True), "lines_guessed": tally(False),
           "per_frame": {int(i): [bool(ok[i]), verdict[i], support.get(i)] for i in verdict}, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(f"{out_dir}/{clip}.json", "w"), indent=1)
    # picture sheet
    tiles = []
    for k in sorted(frames_keep, key=lambda i: (bool(ok[i]), i)):
        f, v = frames_keep[k]; img = f.copy(); draw_model(img, Hs[k], L, W)
        img = cv2.resize(img, (640, 360)); txt = f"f{k} {'MEASURED' if ok[k] else 'GUESSED'} lines:{v}"
        cv2.rectangle(img, (0, 0), (640, 30), (0, 0, 0), -1); cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if not ok[k] else (255, 255, 255), 2)
        tiles.append(img)
    if tiles:
        while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(f"{out_dir}/{clip}_sheet.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(json.dumps({k: v for k, v in res.items() if k != "gaps"}), flush=True)
    return res


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/qa/n0-22b"; os.makedirs(out_dir, exist_ok=True)
    allr = []
    for clip in (SFK, AIK):
        try: allr.append(run_clip(clip, out_dir))
        except Exception as e:
            import traceback; traceback.print_exc(); allr.append({"clip": clip, "error": repr(e)})
        json.dump(allr, open(f"{out_dir}/summary.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
