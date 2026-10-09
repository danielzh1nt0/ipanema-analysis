"""K4 (9 Oct, Kaggle GPU, free): people the Vallentuna export is MISSING. In the pipeline a track whose kit read 'neither
team' most of the time was dropped (sun-washed dark shirts): 216k rows in the join, and by eye the SFK player on the ball
is often simply not there, which hands possession to Vallentuna or to nobody. On every exported frame (10 per s) this
detects people (RF-DETR, pipeline tiles), labels every box with the K3c kit rule, and writes the boxes that match NO
exported player and read A or B, with pitch metres from the frame's own calibration (or the nearest calibrated frame
within 1 s). The join adds them as extra rows (overrides/<match>_extras.json via tools/make_extras.py).
    VIDEO=<file or url> PYTHONPATH=. python tools/vall_extras.py -> $OUT/{extras.jsonl, summary.json, qa/*.jpg}"""
import os, sys, json, glob, time, bisect, collections, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = os.environ.get("MATCH", "p15u-vs-vallentuna-2026-10-03-6cce")
OUT = os.environ.get("OUT", "results/qa/k4/extras"); STEP = int(os.environ.get("STEP", "1")); BATCH = int(os.environ.get("BATCH", "8"))
QA_T = [549.4, 1332.8, 1851.8, 3363.3, 3537.8, 4910.3, 5542.5, 724.9, 1146.0, 3415.4, 3910.1, 4015.4]   # by-eye moments where SFK's possession was missed
from vall_relabel import match_boxes

def main():
    import v3lab as L
    from ipanema import tracking as TR
    os.makedirs(f"{OUT}/qa", exist_ok=True); t0 = time.time()
    frames = []
    for fn in sorted(glob.glob(f"results/volume/runs/matches/{M}/frames_*.json")): frames += json.load(open(fn))["frames"]
    md = json.load(open(f"results/volume/runs/matches/{M}/match_data.json")); L_m, W_m = md["pitch"]["length"], md["pitch"]["width"]
    ts = [f["t"] for f in frames]; Hs = [(f["t"], np.array(f["pitch_lines"], float).reshape(3, 3)) for f in frames if f.get("pitch_lines")]
    Ht = [h[0] for h in Hs]
    def H_at(t):
        i = bisect.bisect_left(Ht, t); c = [Hs[j] for j in (i - 1, i) if 0 <= j < len(Hs)]
        if not c: return None
        tt, H = min(c, key=lambda x: abs(x[0] - t)); return H if abs(tt - t) <= 1.0 else None
    model = L.fit_model(); print("kit model: hue mode", __import__("ipanema.kits", fromlist=["x"]).hue_mode(model.model), flush=True)
    det = TR.RFDetrPerson("medium")
    cap = cv2.VideoCapture(os.environ["VIDEO"]); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    want = {}
    for j, fr in enumerate(frames):
        if j % STEP == 0: want[int(round(fr["t"] * fps))] = fr
    qa_k = {int(round(t * fps)): t for t in QA_T}
    out = open(f"{OUT}/extras.jsonl", "w"); stats = collections.Counter(); k = -1; last = max(list(want) + list(qa_k)); buf = []; done = 0
    def flush():
        nonlocal done
        if not buf: return
        res = det.detect_batch([b[1] for b in buf], 0.3, TR.FOLLOW_TILES)
        for (kk, f, fr, qa_t), (d, _) in zip(buf, res):
            bx = [list(map(float, x)) for x in d.xyxy]
            if not bx: continue
            labs = model.predict_batch(f, bx); mb = match_boxes(fr["players"], bx); matched = set(mb.values())
            H = H_at(fr["t"]); Hi = np.linalg.inv(H) if H is not None else None
            stats["boxes"] += len(bx); stats["matched"] += len(matched)
            for i, (b, lab) in enumerate(zip(bx, labs)):
                if i in matched: continue
                stats[f"unmatched_{lab}"] += 1
                if lab not in ("A", "B") or Hi is None: continue
                fx, fy = (b[0] + b[2]) / 2, b[3]; p = Hi @ np.array([fx, fy, 1.0]); mx, my = float(p[0] / p[2]), float(p[1] / p[2])
                if not (-1.0 <= mx <= L_m + 1.0 and -1.0 <= my <= W_m + 1.0): stats["unmatched_offpitch"] += 1; continue
                stats["extra"] += 1
                out.write(json.dumps([round(fr["t"], 3), lab, round(fx, 1), round(fy, 1), round(mx, 2), round(my, 2), [round(v, 1) for v in b]]) + "\n")
            if qa_t is not None:
                g = f.copy()
                for p in fr["players"]:
                    if p.get("px"): cv2.circle(g, (int(p["px"][0]), int(p["px"][1])), 9, (0, 0, 255) if p["team"] == "A" else (255, 255, 255), 2)
                for i, (b, lab) in enumerate(zip(bx, labs)):
                    if i in matched: continue
                    col = (0, 255, 0) if lab in ("A", "B") else (0, 200, 255)
                    cv2.rectangle(g, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2); cv2.putText(g, f"+{lab}", (int(b[0]), int(b[1]) - 4), 0, 0.6, col, 2)
                cv2.putText(g, f"t={qa_t} red=export SFK white=export VAL green box=added", (10, 30), 0, 0.8, (0, 255, 0), 2)
                cv2.imwrite(f"{OUT}/qa/t{qa_t}.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 85])
        done += len(buf); buf.clear()
    while k < last:
        ok = cap.grab(); k += 1
        if not ok: break
        if k not in want and k not in qa_k: continue
        ok, f = cap.retrieve()
        if not ok: continue
        if f.shape[1] != 1920: f = cv2.resize(f, (1920, 1080))
        fr = want.get(k) or frames[min(bisect.bisect_left(ts, k / fps), len(frames) - 1)]
        buf.append((k, f, fr, qa_k.get(k)))
        if len(buf) >= BATCH: flush()
        if done and done % 3000 < BATCH and not buf:
            print(f"{time.time() - t0:6.0f}s frame {k} ({done} frames) {dict(stats)}", flush=True); out.flush()
    flush(); cap.release(); out.close()
    s = dict(stats); s.update({"frames": done, "minutes": round((time.time() - t0) / 60, 1)})
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1); print(s, flush=True)

if __name__ == "__main__":
    main()
