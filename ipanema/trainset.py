"""Training material from a whole Veo match (27 Sep), built from the cached output of both ball finders (ball_piece).

1. WASB guesses (every frame) are linked into tracks (autolabel.link); a track that moves like a ball for a second or more
   gives labels. Where the click model (every 3rd frame) agrees within 15 px, the label is marked 'agreed'.
2. Disagreements: frames where both finders are sure (>= 0.4) but more than 40 px apart -> candidates for a question.
3. A review sample for Daniel: n_check labels (yes/no: is the circle on the ball?) + n_dis disagreements. A match's labels
   are used for training only if the checked labels are >= 95% right; wrong ones tell us what to filter next.
No camera / pitch model is needed for this, so it works on any ground."""
import os, glob, json, pickle, random, numpy as np
from . import autolabel as AL

def piece_caches(root, match_id, fps=29.97):
    """[(start_s, frame_offset, clicks, wasb)] for every finished piece of a match"""
    out = []
    for d in sorted(glob.glob(f"{root}/cache/{match_id}_s*_d*")):
        if not os.path.exists(f"{d}/ball_piece_done.json"): continue
        info = json.load(open(f"{d}/ball_piece_done.json")); wf = sorted(glob.glob(f"{d}/ball_cands_wasb*.pkl"), key=os.path.getmtime)
        cf = f"{d}/ball_cands_clicks_st3_fz0.pkl"
        if not wf or not os.path.exists(cf): continue
        out.append((info["start_s"], int(round(info["start_s"] * fps)), pickle.load(open(cf, "rb")), pickle.load(open(wf[-1], "rb"))))
    return sorted(out, key=lambda z: z[0])

def piece_labels(clicks, wasb, offset, fps=29.97, agree_px=15.0, sure=0.4, far_px=40.0):
    tr = AL.link(wasb, fps=fps, min_len=30, min_conf=0.2, start_conf=0.35, min_travel_px=60.0)
    labs = []
    for k, x, y, c in AL.labels(tr, fps=fps, every=1):
        if k % 3: continue                                                     # the frames the click model also looked at
        ag = any(np.hypot(x - a, y - b) <= agree_px and cc >= 0.1 for a, b, cc in clicks.get(k, []))
        labs.append({"frame": offset + k, "x": round(x, 1), "y": round(y, 1), "wasb": round(c, 3), "agreed": bool(ag)})
    dis = []
    for k in sorted(clicks):
        a = max(clicks.get(k, []), key=lambda z: z[2], default=None); b = max(wasb.get(k, []), key=lambda z: z[2], default=None)
        if a and b and a[2] >= sure and b[2] >= sure and np.hypot(a[0] - b[0], a[1] - b[1]) > far_px:
            dis.append({"frame": offset + k, "click": [round(a[0], 1), round(a[1], 1), round(a[2], 3)], "wasb": [round(b[0], 1), round(b[1], 1), round(b[2], 3)]})
    return labs, dis, len(tr)

def match_trainset(pieces, fps=29.97):
    labs, dis, ntr = [], [], 0
    for start_s, off, clk, wb in pieces:
        l, d, t = piece_labels(clk, wb, off, fps); labs += l; dis += d; ntr += t
    return {"labels": labs, "disagreements": dis, "tracks": ntr, "pieces": len(pieces),
            "agreed_share": round(sum(l["agreed"] for l in labs) / max(1, len(labs)), 3)}

def review_sample(ts, n_check=28, n_dis=12, seed=0, min_gap_frames=150):
    """spread over the match: labels to check (yes/no) and disagreements to settle (which circle, or neither)"""
    rng = random.Random(seed)
    def spread(items, n):
        items = sorted(items, key=lambda z: z["frame"]); pick = []
        for it in rng.sample(items, len(items)):
            if all(abs(it["frame"] - p["frame"]) >= min_gap_frames for p in pick): pick.append(it)
            if len(pick) >= n: break
        return sorted(pick, key=lambda z: z["frame"])
    return {"check": spread(ts["labels"], n_check), "disagree": spread(ts["disagreements"], n_dis)}

# ---- 28 Sep: labels from the click-trained finder (Daniel's Vasalund check: WASB-track labels 3/28 right -> rejected;
# in disagreements the click model was right 6/12, WASB 0/12). The click model looks at every 3rd frame.
CONF_BINS = ((0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01))

def click_labels(clicks, wasb, offset, fps=29.97, min_conf=0.3, min_steps=4, max_step_px=80.0, min_travel_px=25.0, agree_px=15.0):
    """a label = a click-model detection that belongs to a run of >= min_steps consecutive looks (every 3rd frame) that
    link up like a ball; conf = the detection's own score (the review is split by it to find the threshold that is 95% right)"""
    tr = AL.link(clicks, fps=fps, stride=3, max_step_px=max_step_px, min_len=min_steps, max_gap=1, min_conf=min_conf,
                 min_travel_px=min_travel_px, start_conf=min_conf)
    labs = []
    for t in tr:
        for k, x, y, c in t:
            if c < min_conf: continue
            ag = any(np.hypot(x - a, y - b) <= agree_px and cc >= 0.1 for a, b, cc in wasb.get(k, []))
            labs.append({"frame": offset + int(k), "x": round(float(x), 1), "y": round(float(y), 1), "conf": round(float(c), 3), "run": len(t), "wasb_agrees": bool(ag)})
    return labs, len(tr)

def match_click_trainset(pieces, fps=29.97, **kw):
    labs, ntr = [], 0
    for start_s, off, clk, wb in pieces:
        l, t = click_labels(clk, wb, off, fps, **kw); labs += l; ntr += t
    by_bin = {f"{a:.2f}-{min(b, 1):.2f}": sum(a <= l["conf"] < b for l in labs) for a, b in CONF_BINS}
    return {"source": "clicks", "labels": sorted(labs, key=lambda z: z["frame"]), "tracks": ntr, "pieces": len(pieces), "by_conf": by_bin,
            "wasb_agrees_share": round(sum(l["wasb_agrees"] for l in labs) / max(1, len(labs)), 3)}

def review_by_conf(ts, per_bin=10, seed=0, min_gap_frames=150):
    """per_bin labels from each confidence band, spread over the match"""
    rng = random.Random(seed); out = []
    for a, b in CONF_BINS:
        items = [l for l in ts["labels"] if a <= l["conf"] < b]; pick = []
        for it in rng.sample(items, len(items)):
            if all(abs(it["frame"] - p["frame"]) >= min_gap_frames for p in pick): pick.append(dict(it, band=f"{a:.2f}-{min(b, 1):.2f}"))
            if len(pick) >= per_bin: break
        out += pick
    return sorted(out, key=lambda z: z["frame"])
