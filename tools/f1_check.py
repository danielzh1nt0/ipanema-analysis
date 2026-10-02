"""F1 (2 Oct, free runner, CPU, no Modal): why the full AIK match has 6+6 players per frame when the 5-min clip has 7+7,
plus by-eye pictures of both full matches as the app shows them.

Part A - AIK team colours. The full run removed 794,212 'not a team kit' rows (3.8 per frame) vs 9,086 in the clip
(1.0 per frame). The full run learns ONE kit model for the whole match (modal_app.rf_kits, 36 frames). It ran before the
AIK periods file existed, so it sampled the default spans (15-45% and 55-90% of the video), which include half-time.
The clip learned its own model from 36 frames of its 5 minutes. Here the three models are rebuilt from the same frames
(RF-DETR on CPU) and used to label the same people: 20 frames inside the clip window and 20 over both halves. Counts
are of people whose feet are on the pitch (kits.feet_on_pitch). Picture: 6 clip moments, match model left, clip model right.

Part B - the app's view: 10 moments of each full match (5 per half), drawn from the app's own frame files (Supabase
public storage) on the R2 video: players by team, ball. Skipped with a note if the files can't be read.

    R2_PUBLIC_URL=... python tools/f1_check.py      -> results/qa/f1/
    DRY=1 LOCAL_VIDEO=x.mp4 python tools/f1_check.py   (stand-in detector, a short local video, no network)"""
import os, sys, json, time, subprocess, urllib.request, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRY = os.environ.get("DRY") == "1"
if not DRY:
    try: import rfdetr  # noqa
    except ImportError: subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
from ipanema import tracking as TR, kits as KT

OUT = os.environ.get("F1_OUT", "results/qa/f1"); os.makedirs(OUT, exist_ok=True)
R2 = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
SUPA = "https://savbsnvusqbogdzvkjaf.supabase.co/storage/v1/object/public/matches"
AIK = "p15u-vs-aik-2026-09-21-bd09"; CLIP_S = 2520.0
PERIODS = {AIK: [[366, 3390], [3950, 6899]], "SFKBP1109": [[0, 3060], [3733, 5940]]}
t0 = time.time(); LOG = []
def log(*a):
    m = " ".join(str(x) for x in a); LOG.append(f"{time.time() - t0:6.0f}s {m}"); print(m, flush=True)

class StandIn:
    """dry run: 'people' = fixed boxes, so the plumbing runs without RF-DETR"""
    def detect_batch(self, fs, conf, tiles):
        import supervision as sv
        h, w = fs[0].shape[:2]; xy = np.array([[w * a, h * 0.5, w * a + 30, h * 0.5 + 70] for a in np.linspace(0.1, 0.85, 10)], float)
        return [(sv.Detections(xyxy=xy, confidence=np.ones(len(xy)), class_id=np.zeros(len(xy), int)), {0: "person"})]

def open_video(match):
    for src in ([os.environ["LOCAL_VIDEO"]] if os.environ.get("LOCAL_VIDEO") else [f"{R2}/{match}/video.mp4", f"{R2}/{match}/full.mp4", f"{R2}/{match}/video_cropped.mp4"]):
        cap = cv2.VideoCapture(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cap.isOpened() and n > 0: log(f"{match}: video {src.split('/')[-2:]} {n} frames @ {cap.get(cv2.CAP_PROP_FPS):.3f}"); return cap, n, cap.get(cv2.CAP_PROP_FPS) or 29.97
        cap.release()
    log(f"{match}: no video found on R2"); return None, 0, 29.97

def frame(cap, n, fps, t):
    k = int(round(t * fps))
    if DRY or os.environ.get("WRAP") == "1": k %= max(1, n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); return f if ok else None

det = StandIn() if DRY else TR.RFDetrPerson("medium"); _cache = {}
def people(cap, n, fps, t):
    key = round(t, 2)
    if key not in _cache:
        f = frame(cap, n, fps, t)
        _cache[key] = None if f is None else (f, np.array([b for b in det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0].xyxy]).reshape(-1, 4))
    return _cache[key]

def part_a():
    cap, n, fps = open_video(AIK)
    if cap is None: return {"error": "AIK full video not on R2"}
    T = n / fps
    def spread(spans):   # the exact sampling of modal_app.rf_kits
        return np.concatenate([np.linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), 18) for a, b in spans])
    clip_n = int(round(300 * fps)); clip_ts = CLIP_S + np.linspace(0, clip_n - 1, 36).astype(int) / fps   # the exact sampling of modal_app.players_rf
    plans = {"match model as used (default spans)": spread([[0.15 * T, 0.45 * T], [0.55 * T, 0.9 * T]]),
             "match model from the halves (periods file)": spread(PERIODS[AIK]),
             "clip model (36 frames of the 5-min clip)": clip_ts}
    half_time = [t for t in plans["match model as used (default spans)"] if PERIODS[AIK][0][1] <= t < PERIODS[AIK][1][0]]
    log(f"default spans: {len(half_time)} of 36 kit frames fall in half-time ({[round(x) for x in half_time]})")
    models = {}
    for name, ts in plans.items():
        fb = [p for p in (people(cap, n, fps, t) for t in ts) if p is not None]
        models[name] = KT.KitTeamModel().fit_frames(fb, log=log); log(f"  {name}: {len(fb)} frames, {len(_cache)} frames detected so far")
    tests = {"inside the clip window": CLIP_S + np.linspace(7, 293, 20), "over both halves": spread(PERIODS[AIK])[1::2][:20] + 11.0}
    res = {"half_time_kit_frames": len(half_time), "counts": {}}
    for tname, ts in tests.items():
        acc = {m: {"A": 0, "B": 0, "K": 0, "none": 0} for m in models}; nf = 0
        for t in ts:
            p = people(cap, n, fps, t)
            if p is None: continue
            f, b = p; top = KT.pitch_top(f); keep = np.array([bb for bb in b if bb[3] - bb[1] >= 22 and KT.feet_on_pitch(f, bb, top)]).reshape(-1, 4); nf += 1
            for m, tm in models.items():
                for c in (tm.predict_batch(f, keep) if len(keep) else []): acc[m][c if c in ("A", "B", "K") else "none"] += 1
        res["counts"][tname] = {m: {k: round(v / max(1, nf), 1) for k, v in c.items()} for m, c in acc.items()}; res["counts"][tname]["frames"] = nf
        log(tname, json.dumps(res["counts"][tname]))
    # picture: 6 clip moments, left = match model as used, right = clip model
    col = {"A": (0, 0, 230), "B": (255, 200, 0), "K": (200, 0, 200), None: (128, 128, 128)}
    rows = []
    for t in CLIP_S + np.linspace(20, 280, 6):
        p = people(cap, n, fps, t)
        if p is None: continue
        f, b = p; top = KT.pitch_top(f); pair = []
        for m in ("match model as used (default spans)", "clip model (36 frames of the 5-min clip)"):
            g = f.copy(); labs = models[m].predict_batch(f, b) if len(b) else []
            for bb, c in zip(b, labs):
                on = KT.feet_on_pitch(f, bb, top); cv2.rectangle(g, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), col.get(c, col[None]), 3 if on else 1)
            cnt = {k: sum(1 for bb, c in zip(b, labs) if c == k and KT.feet_on_pitch(f, bb, top)) for k in ("A", "B", "K")}
            g = cv2.resize(g, (960, 540)); cv2.rectangle(g, (0, 0), (960, 30), (0, 0, 0), -1)
            cv2.putText(g, f"{t:.0f}s {m.split(' (')[0]}: dark {cnt['A']} light {cnt['B']} neither {cnt['K']} (on pitch)", (6, 21), 0, 0.6, (255, 255, 255), 2); pair.append(g)
        rows.append(np.hstack(pair))
    if rows: cv2.imwrite(f"{OUT}/aik_kits_match_vs_clip.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    for m, tm in models.items():
        cv2.imwrite(f"{OUT}/aik_kit_strips_{m.split(' (')[0].replace(' ', '_')}.jpg", np.vstack([cv2.resize(tm.strips[k], (576, 144)) for k in ("A", "B")]))
    res["dark_share"] = {m: tm.dark_share for m, tm in models.items()}
    cap.release(); return res

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as r: return json.loads(r.read())

def part_b(match):
    if DRY: return {"skipped": "dry run"}
    try: md = fetch_json(f"{SUPA}/{match}/match_data.json")
    except Exception as e: log(f"{match}: app files not readable ({e!r})"); return {"error": f"match_data.json not readable: {e!r}"[:300]}
    cap, n, fps = open_video(match)
    if cap is None: return {"error": "video not on R2"}
    chunks = md.get("frame_chunks") or []; got = {}; tiles = []; rows = []
    ts = np.concatenate([np.linspace(a + 0.1 * (b - a), b - 0.1 * (b - a), 5) for a, b in PERIODS[match]])
    for t in ts:
        ch = next((c for c in chunks if c["t_start"] <= t <= c["t_end"]), None)
        if ch is None: continue
        if ch["key"] not in got:
            try: got[ch["key"]] = fetch_json(f"{SUPA}/{match}/{ch['key']}.json")["frames"]
            except Exception as e: log(f"{match}: {ch['key']} not readable ({e!r})"); got[ch["key"]] = []
        fr = min(got[ch["key"]], key=lambda x: abs(x["t"] - t), default=None)
        if fr is None: continue
        f = frame(cap, n, fps, fr["t"])
        if f is None: continue
        for p in fr["players"]:
            c = (0, 0, 230) if p["team"] == "A" else (255, 200, 0); x, y = map(int, p["px"])
            cv2.circle(f, (x, y), 14, c, 3 if p.get("state") == "observed" else 1)
        if fr.get("ball"): cv2.circle(f, tuple(map(int, fr["ball"]["px"])), 18, (0, 255, 255), 3)
        a_ = sum(p["team"] == "A" for p in fr["players"]); b_ = sum(p["team"] == "B" for p in fr["players"])
        g = cv2.resize(f, (960, 540)); cv2.rectangle(g, (0, 0), (960, 30), (0, 0, 0), -1)
        cv2.putText(g, f"{match} {fr['t']:.0f}s  A {a_}  B {b_}  ball {'yes' if fr.get('ball') else 'no'}  lines {'yes' if fr.get('cal_ok', True) else 'unsure'}", (6, 21), 0, 0.6, (255, 255, 255), 2)
        tiles.append(g); rows.append({"t": fr["t"], "A": a_, "B": b_, "ball": bool(fr.get("ball")), "cal_ok": fr.get("cal_ok")})
    cap.release()
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2] + ([np.zeros_like(tiles[0])] if len(tiles[i:i + 2]) == 1 else [])
        cv2.imwrite(f"{OUT}/{match}_app_{i // 2}.jpg", np.hstack(pair), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"moments": rows}

if __name__ == "__main__":
    out = {"A_aik_kits": part_a()}
    for m in (AIK, "SFKBP1109"): out[f"B_app_{m}"] = part_b(m)
    out["log"] = LOG[-60:]; out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1, default=str); log("done", out["minutes"], "min")
