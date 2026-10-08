"""C3d (8 Oct): wrong TRUSTED camera rows. For every second of a match that has a camera pose, how well do the drawn far
lines (every model line >= 20 m from the camera except the near touchline - what the match fit uses) sit on the painted
lines? 'far cost' = mean capped px distance (c3c_fit.Frame.cost, 1280x720, cap 15 px). Rows as they are, no re-fit.
Also the older validated picture judge (calcheck.judge_frame) for comparison.

    python tools/c3d_farcost.py pick <match> <out dir>      (local) every posed second -> <out>/sample.json
    python tools/c3d_farcost.py run <out dir> [<out dir>]   (free runner) reads the video once, front to back
          -> <out>/costs.json (one line per second) + <out>/look/<id>.jpg (a cost-stratified sample of TRUSTED in-play
             seconds, model lines drawn thin, neutral ids for blind grading; the id -> second key is in <out>/look_key.json)
          VIDEO=/path.mp4 for a local dry run; otherwise R2_PUBLIC_URL/<match>/video.mp4. N_LOOK = pictures per cost bin."""
import os, sys, json, time, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import lines as LN, linecal as LC
from ipanema.calcheck import judge_frame
from c3c_fit import Frame, model_pts, CAP

BINS = [0.0, 3.0, 5.0, 7.0, 9.0, 12.0, CAP + 0.01]

def far_cost(img, camera, pose, far, need=30):
    """(far cost px or None when < need far points are in view / almost no paint, far points in view)"""
    fr = Frame(cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA))
    d = fr.dists(camera, np.asarray(pose, float), far)
    if len(d) < need or fr.paint < 300: return None, int(len(d))
    return round(float(np.minimum(d, CAP).mean()), 2), int(len(d))

def draw(img, camera, pose, colour=(0, 255, 255)):
    h, w = img.shape[:2]
    for seg in LN.projected_segments(camera, np.asarray(pose, float), w, h).values():
        for x0, y0, x1, y1 in seg:
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) < 1e5: cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), colour, 1, cv2.LINE_AA)
    return img

def pick(match, out):
    d = json.load(open(f"calibration/{match}_lines_match.json")); per = json.load(open(f"periods/{match}.json"))["periods_s"]
    rows = [{"t": r["t"], "pose": r["pose"], "trusted": bool(LC.brave(r)), "inplay": any(a + 5 <= r["t"] <= b - 5 for a, b in per),
             "row_cost": r.get("cost"), "why": r.get("why", [])} for r in sorted(d["rows"], key=lambda r: r["t"]) if r.get("pose") is not None]
    os.makedirs(out, exist_ok=True)
    json.dump({"match": match, "src_key": f"{match}/video.mp4", "camera": d["camera"], "rows": rows}, open(f"{out}/sample.json", "w"))
    print(match, len(rows), "posed seconds,", sum(r["trusted"] for r in rows), "trusted,", sum(r["trusted"] and r["inplay"] for r in rows), "trusted in play")

def choose(res, n_per_bin, seed=0):
    """cost-stratified pick of trusted in-play seconds (+ a few unjudged), shuffled neutral ids"""
    rng = np.random.default_rng(seed); ok = [r for r in res if r["trusted"] and r["inplay"]]; out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        b = [r["t"] for r in ok if r["far"] is not None and lo <= r["far"] < hi]
        out += list(rng.choice(b, min(n_per_bin, len(b)), replace=False)) if b else []
    u = [r["t"] for r in ok if r["far"] is None]; out += list(rng.choice(u, min(max(2, n_per_bin // 2), len(u)), replace=False)) if u else []
    out = [float(t) for t in out]; rng.shuffle(out)
    return {f"x{i:03d}": t for i, t in enumerate(out)}

def run(d, n_per_bin=None):
    n_per_bin = n_per_bin or int(os.environ.get("N_LOOK", "10")); t0 = time.time()
    S = json.load(open(f"{d}/sample.json")); cam = S["camera"]; far = model_pts(cam, 106.0, 64.0)[0]
    src = os.environ.get("VIDEO") or f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{S['src_key']}"
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    print(d, "video", "ok" if cap.isOpened() else "NOT OPEN", fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), flush=True)
    want = {}
    for r in S["rows"]: want.setdefault(int(round(r["t"] * fps)), r)
    last = max(want) if want else -1; k = 0; res = []; small = {}
    while k <= last:
        if k not in want:
            if not cap.grab(): break
            k += 1; continue
        ok, f = cap.read()
        if not ok: break
        r = want[k]; c, npts = far_cost(f, cam, r["pose"], far)
        j = judge_frame(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA), LC.homography(cam, r["pose"], 1280, 720), 106.0, 64.0)
        res.append({"t": r["t"], "trusted": r["trusted"], "inplay": r["inplay"], "far": c, "far_pts": npts, "judge": j["verdict"],
                    "support": j.get("weakest_support"), "row_cost": r["row_cost"], "why": r["why"]})
        if r["trusted"] and r["inplay"]: small[r["t"]] = cv2.imencode(".jpg", cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        if len(res) % 500 == 0: print(" ", len(res), "seconds", round(time.time() - t0), "s", flush=True)
        k += 1
    cap.release()
    key = choose(res, n_per_bin); os.makedirs(f"{d}/look", exist_ok=True); pose = {r["t"]: r["pose"] for r in S["rows"]}
    for i, t in key.items():
        img = cv2.imdecode(np.frombuffer(small[t], np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(f"{d}/look/{i}.jpg", draw(img, cam, pose[t]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(key, open(f"{d}/look_key.json", "w"), indent=0)
    json.dump({"match": S["match"], "fps": fps, "seconds": len(res), "minutes": round((time.time() - t0) / 60, 1), "rows": res}, open(f"{d}/costs.json", "w"))
    print(d, "seconds", len(res), "of", len(S["rows"]), "look", len(key), "in", round((time.time() - t0) / 60, 1), "min", flush=True)
    return len(res)

def main(argv=sys.argv[1:]):
    if argv[0] == "pick": pick(argv[1], argv[2])
    else:
        for d in (argv[1:] if argv[0] == "run" else argv): run(d)

if __name__ == "__main__": main()
