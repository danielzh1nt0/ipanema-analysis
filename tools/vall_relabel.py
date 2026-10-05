"""K3c free route (5 Oct, Kaggle GPU, no Modal): new team for every exported Vallentuna player, by the K3c kit rule.
Daniel said no more paid runs, so instead of a GPU re-track on Modal this goes through the exported frames (the app's
frames_*.json, 10 per s): on every STEP-th exported frame it detects people (RF-DETR, pipeline tiles), finds the box at
each exported player's feet and labels the box with the kit model (hue mode + kit-colour pixel share, ipanema/kits.py).
Votes per player id -> override.json {id: "A"|"B"} for the ids whose majority differs or agrees (all are written).
Applying it to the app needs one join (Modal CPU) reading the override - not done here.
    VIDEO=<file or url> PYTHONPATH=. python tools/vall_relabel.py   -> $OUT/{votes.json, override.json, summary.json, qa/*.jpg}"""
import os, sys, json, glob, time, collections, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = "p15u-vs-vallentuna-2026-10-03-6cce"
OUT = os.environ.get("OUT", "results/qa/k3c/relabel"); STEP = int(os.environ.get("STEP", "2")); BATCH = int(os.environ.get("BATCH", "8"))
QA_T = [638, 1045, 1452, 1858, 2265, 2672, 3408, 3815, 4222, 4628, 5035, 5442]     # the qa-frames moments

def match_boxes(players, boxes):
    """exported player (feet px) -> index of the detection box whose bottom centre is at his feet, or None"""
    out = {}
    for p in players:
        if not p.get("px"): continue
        fx, fy = p["px"]; best, bd = None, 1e9
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b[:4]; h = y2 - y1; d = abs((x1 + x2) / 2 - fx) + abs(y2 - fy)
            if d <= max(25.0, 0.4 * h) and d < bd: best, bd = i, d
        if best is not None: out[p["id"]] = best
    return out

def label_frame(model, f, players, boxes, detail=None):
    """{player id: label} for one frame; detail (list) gets (id, label, red share, dark share, box height) per player"""
    mb = match_boxes(players, boxes)
    if not mb: return {}
    ids = list(mb); bx = [list(boxes[mb[i]][:4]) for i in ids]; labs = model.predict_batch(f, bx)
    if detail is not None:
        from ipanema import kits as K
        _, hue = K._kit_hue(model.model)
        for i, b, lab in zip(ids, bx, labs):
            r, d = K.kit_share(f, b, hue); detail.append((i, lab, None if r is None else round(r, 3), None if d is None else round(d, 3), round(b[3] - b[1], 1)))
    return dict(zip(ids, labs))

def decide(votes, old):
    """id -> team: majority of its A/B readings (at least 3), else its old team"""
    out = {}
    for pid, team in old.items():
        v = votes.get(pid, {}); a, b = v.get("A", 0), v.get("B", 0)
        out[pid] = ("A" if a >= b else "B") if a + b >= 3 else team
    return out

def main():
    import v3lab as L
    from ipanema import tracking as TR
    os.makedirs(f"{OUT}/qa", exist_ok=True); t0 = time.time()
    frames = []
    for fn in sorted(glob.glob(f"results/volume/runs/matches/{M}/frames_*.json")): frames += json.load(open(fn))["frames"]
    old = {}
    for fr in frames:
        for p in fr["players"]: old.setdefault(p["id"], collections.Counter())[p["team"]] += 1
    old = {k: v.most_common(1)[0][0] for k, v in old.items()}
    model = L.fit_model(); print("kit model: hue mode", __import__("ipanema.kits", fromlist=["x"]).hue_mode(model.model), "cls", len(model.cls), flush=True)
    det = TR.RFDetrPerson("medium")
    cap = cv2.VideoCapture(os.environ["VIDEO"]); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    want = {}
    for j, fr in enumerate(frames):
        if j % STEP == 0 and fr["players"]: want[int(round(fr["t"] * fps))] = fr
    qa_k = {int(round(t * fps)): t for t in QA_T}
    perlab = open(f"{OUT}/labels.jsonl", "w")      # [t, id, label, red share, dark share, box height] per labelled player
    votes = collections.defaultdict(collections.Counter); k = -1; last = max(list(want) + list(qa_k)); buf = []; done = 0
    def flush():
        nonlocal done
        if not buf: return
        res = det.detect_batch([b[1] for b in buf], 0.3, TR.FOLLOW_TILES)
        for (kk, f, fr), (d, _) in zip(buf, res):
            bx = [list(map(float, x)) for x in d.xyxy]; det_ = []
            for pid, lab in label_frame(model, f, fr["players"], bx, det_).items():
                if lab in ("A", "B", "K"): votes[pid][lab] += 1
            for row in det_: perlab.write(json.dumps([round(fr["t"], 2)] + list(row)) + "\n")
        done += len(buf); buf.clear()
    while k < last:
        ok = cap.grab(); k += 1
        if not ok: break
        if k not in want and k not in qa_k: continue
        ok, f = cap.retrieve()
        if not ok: continue
        if f.shape[1] != 1920: f = cv2.resize(f, (1920, 1080))
        if k in qa_k: cv2.imwrite(f"{OUT}/qa/t{qa_k[k]}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if k in want:
            buf.append((k, f, want[k]))
            if len(buf) >= BATCH: flush()
        if done and done % 2000 < BATCH and not buf:
            print(f"{time.time() - t0:6.0f}s frame {k} ({done} labelled frames)", flush=True)
            json.dump(votes, open(f"{OUT}/votes.json", "w"))
    flush(); cap.release(); perlab.close()
    new = decide(votes, old)
    changed = sum(new[i] != old[i] for i in old)
    json.dump(votes, open(f"{OUT}/votes.json", "w")); json.dump({str(k): v for k, v in new.items()}, open(f"{OUT}/override.json", "w"))
    s = {"players": len(old), "with_votes": sum(1 for i in old if sum(votes.get(i, {}).values()) >= 3), "changed": changed,
         "old": collections.Counter(old.values()), "new": collections.Counter(new.values()), "labelled_frames": done, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1, default=dict); print(s, flush=True)

if __name__ == "__main__":
    main()
