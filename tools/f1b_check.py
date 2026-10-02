"""F1b (2 Oct, free runner, CPU, no Modal): a kit model per 5-min piece of a full match, its teams named like the match
model (kits.piece_model), vs the one match model the full run uses now (modal_app.rf_kits).

For each piece inside the playing time: 36 frames spread over the piece (the exact sampling of the per-piece RF-DETR job)
-> piece model, named like the match model. Test frames (none of them used for fitting): 3 per piece, plus the 20
frames of F1's 'inside the clip window' test (AIK, with the clip's own model too). Counts are of people whose feet are
on the pitch. 'swapped' = people both models put in a team but in different teams (a high share = an A/B swap).
Pictures: match model left, piece model right; a strip sheet of each piece's team A / team B (A must be the same kit
on every row).

    R2_PUBLIC_URL=... python tools/f1b_check.py            -> results/qa/f1b/
    DRY=1 LOCAL_VIDEO=x.mp4 python tools/f1b_check.py      (stand-in detector, a short local video, no network)"""
import os, sys, json, time, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRY = os.environ.get("DRY") == "1"
if not DRY:
    try: import rfdetr  # noqa
    except ImportError: subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
from ipanema import tracking as TR, kits as KT

OUT = os.environ.get("F1B_OUT", "results/qa/f1b"); os.makedirs(OUT, exist_ok=True)
R2 = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
AIK = "p15u-vs-aik-2026-09-21-bd09"; CLIP_S = 2520.0; PIECE_S = float(os.environ.get("F1B_PIECE_S", "300"))
MATCHES = [m for m in os.environ.get("F1B_MATCHES", f"{AIK},SFKBP1109").split(",") if m]
MAX_PIECES = int(os.environ.get("F1B_MAX_PIECES", "99"))
t0 = time.time(); LOG = []
def log(*a):
    m = " ".join(str(x) for x in a); LOG.append(f"{time.time() - t0:6.0f}s {m}"); print(m, flush=True)
Q = lambda *a: None

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

det = StandIn() if DRY else TR.RFDetrPerson("medium")

class Match:
    def __init__(self, match):
        self.match = match; self.cap, self.n, self.fps = open_video(match); self.cache = {}
    def people(self, k, keep=True):
        """detections of frame k; only test frames are kept in memory (a full match's fitting frames would not fit)"""
        k = int(k) % max(1, self.n) if DRY else int(k)
        if k in self.cache: return self.cache[k]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = self.cap.read()
        p = None if not ok else (f, np.array([b for b in det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0].xyxy]).reshape(-1, 4))
        if keep: self.cache[k] = p
        return p
    def fb(self, ks): return [p for p in (self.people(k, keep=False) for k in ks) if p is not None]

def on_pitch(f, b):
    top = KT.pitch_top(f); return np.array([bb for bb in b if bb[3] - bb[1] >= 22 and KT.feet_on_pitch(f, bb, top)]).reshape(-1, 4)

def count(models, frames):
    """per model: A/B/K per frame; per pair (other, 'piece'): same / swapped team readings"""
    acc = {m: {"A": 0, "B": 0, "K": 0} for m in models}; pair = {m: {"same": 0, "swapped": 0} for m in models if m != "piece"}; nf = 0
    for (f, b), piece in frames:
        keep = on_pitch(f, b); nf += 1
        if not len(keep): continue
        labs = {m: (piece if m == "piece" else tm).predict_batch(f, keep) for m, tm in models.items()}
        for m, ls in labs.items():
            for c in ls:
                if c in acc[m]: acc[m][c] += 1
        for m in pair:
            for a, p in zip(labs[m], labs["piece"]):
                if a in ("A", "B") and p in ("A", "B"): pair[m]["same" if a == p else "swapped"] += 1
    out = {m: {k: round(v / max(1, nf), 2) for k, v in c.items()} for m, c in acc.items()}
    out["agreement_with_piece_model"] = pair; out["frames"] = nf; return out

COL = {"A": (0, 0, 230), "B": (255, 200, 0), "K": (200, 0, 200), "O": (90, 90, 90)}
def tile(f, b, tm, title):
    g = f.copy(); top = KT.pitch_top(f); labs = tm.predict_batch(f, b) if len(b) else []; cnt = {"A": 0, "B": 0, "K": 0}
    for bb, c in zip(b, labs):
        on = bb[3] - bb[1] >= 22 and KT.feet_on_pitch(f, bb, top)
        cv2.rectangle(g, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), COL.get(c, COL["O"]), 3 if on else 1)
        if on and c in cnt: cnt[c] += 1
    g = cv2.resize(g, (960, int(960 * f.shape[0] / f.shape[1]))); cv2.rectangle(g, (0, 0), (960, 30), (0, 0, 0), -1)
    cv2.putText(g, f"{title}: A(red) {cnt['A']} B(blue) {cnt['B']} neither(pink) {cnt['K']}", (6, 21), 0, 0.6, (255, 255, 255), 2); return g

def run(match):
    M = Match(match)
    if M.cap is None: return {"error": "video not on R2"}
    fps, T = M.fps, M.n / M.fps
    pf = f"periods/{match}.json"
    spans = json.load(open(pf))["periods_s"] if os.path.exists(pf) and not DRY else [[0.15 * T, 0.45 * T], [0.55 * T, 0.9 * T]]
    log(f"{match}: playing time {spans}")
    ts = np.concatenate([np.linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), 18) for a, b in spans])      # modal_app.rf_kits
    match_tm = KT.KitTeamModel().fit_frames(M.fb([t * fps for t in ts]), log=log)
    pieces = []; i = 0
    while i * PIECE_S < T:
        s, e = i * PIECE_S, min(T, (i + 1) * PIECE_S)
        if any(min(e, b) - max(s, a) > min(60, 0.2 * PIECE_S) for a, b in spans): pieces.append((i, s, e))
        i += 1
    pieces = pieces[:MAX_PIECES]; res = {"spans": spans, "match_model_groups": [int(x) for x in match_tm.model["sizes"]], "pieces": [], "tests": {}}
    piece_tm = {}; strips = []
    for i, s, e in pieces:
        k0, k1 = int(round(s * fps)), int(round(e * fps)) - 1
        fb = M.fb(np.linspace(k0, k1, 36).astype(int))                      # modal_app.players_rf sampling on the piece
        tm, info = KT.piece_model(fb, match_tm, log=Q); piece_tm[i] = tm; del fb
        info.update(i=i, start_s=s); res["pieces"].append(info)
        log(f"{match} piece {i} ({s:.0f}s): {json.dumps(info)}")
        if tm is not match_tm:
            row = np.hstack([cv2.resize(tm.strips[t], (576, 144)) if tm.strips[t].shape[0] else np.zeros((144, 576, 3), np.uint8) for t in ("A", "B")])
        else: row = np.zeros((144, 1152, 3), np.uint8)
        row = row.copy(); cv2.rectangle(row, (0, 0), (380, 24), (0, 0, 0), -1)
        cv2.putText(row, f"piece {i} {s/60:.0f}min {'flip' if info['flip'] else ''}{'match model: ' + info['fallback'] if info['fallback'] else ''}", (4, 17), 0, 0.5, (255, 255, 255), 1)
        strips.append(row)
        if tm is not match_tm: tm.samples = []                              # frees the piece's 36 frames (predict does not need them)
    if strips:
        head = np.hstack([cv2.resize(match_tm.strips[t], (576, 144)) for t in ("A", "B")]).copy(); cv2.rectangle(head, (0, 0), (380, 24), (0, 0, 0), -1)
        cv2.putText(head, "MATCH model (left A, right B)", (4, 17), 0, 0.5, (255, 255, 255), 1)
        cv2.imwrite(f"{OUT}/{match}_strips.jpg", np.vstack([head] + strips), [cv2.IMWRITE_JPEG_QUALITY, 75])
    # test frames: 3 per piece, offset by 0.37 s from any fitting frame
    tf = []
    for i, s, e in pieces:
        for x in (0.2, 0.5, 0.8):
            t = s + x * (e - s) + 0.37
            if any(a <= t <= b for a, b in spans):
                p = M.people(int(round(t * fps)))
                if p is not None: tf.append((p, piece_tm[i]))
    res["tests"]["3 per piece"] = count({"match": match_tm, "piece": None}, tf); log(match, "3 per piece", json.dumps(res["tests"]["3 per piece"]))
    if match == AIK and not DRY or (DRY and match == MATCHES[0]):
        clip = CLIP_S if not DRY else 0.0
        clip_n = int(round(300 * fps)); clip_tm = KT.KitTeamModel().fit_frames(M.fb(int(round(clip * fps)) + np.linspace(0, clip_n - 1, 36).astype(int)), log=log)
        pof = lambda t: piece_tm.get(int(t // PIECE_S), match_tm)
        cw = [(M.people(int(round(t * fps))), pof(t)) for t in clip + np.linspace(7, 293, 20)]
        res["tests"]["inside the clip window (F1 frames)"] = count({"match": match_tm, "clip": clip_tm, "piece": None}, [x for x in cw if x[0] is not None])
        log(match, "clip window", json.dumps(res["tests"]["inside the clip window (F1 frames)"]))
        rows = []
        for t in clip + np.linspace(20, 280, 6):                            # F1's 6 picture moments
            p = M.people(int(round(t * fps)))
            if p is not None: rows.append(np.hstack([tile(*p, match_tm, f"{t:.0f}s match"), tile(*p, pof(t), f"{t:.0f}s piece")]))
        if rows: cv2.imwrite(f"{OUT}/{match}_clip_moments.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 75])
    rows = []                                                               # 6 moments over the halves
    for (p, tm) in tf[1::max(1, len(tf) // 6)][:6]:
        rows.append(np.hstack([tile(*p, match_tm, "match"), tile(*p, tm, "piece")]))
    if rows: cv2.imwrite(f"{OUT}/{match}_half_moments.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 75])
    M.cap.release(); return res

if __name__ == "__main__":
    out = {}
    for m in MATCHES:
        try: out[m] = run(m)
        except Exception as e:
            import traceback; out[m] = {"error": traceback.format_exc()[-2000:]}; log(m, "FAILED", out[m]["error"])
        json.dump({**out, "log": LOG[-80:]}, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
    out["log"] = LOG[-80:]; out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1, default=str); log("done", out["minutes"], "min")
