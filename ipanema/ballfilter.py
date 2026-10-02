"""T1 (2 Oct): second opinion on each ball guess BEFORE the picker. A small 3-frame crop scorer (ipanema/ballscorer.py)
gives every finder guess a ball probability p; this module turns p into a new guess list for the picker:
  mode "drop":  remove guesses with p < t (the picker never sees shoes / lines / cones the scorer is sure about)
  mode "mult":  conf * p**t (soft: keeps every guess, lowers the ones the scorer doubts; t = strength)
  mode "blend": conf * (1 - t) + p * t
The scores come from tools/t1_score.py (free runner, needs the video); grading is tools/t1lab.py."""
import numpy as np


def rescore(cands, scores, mode="drop", t=0.2):
    """cands: {frame: [(x, y, conf), ...]}; scores: {frame: array of p, same order}. Frames without scores are kept as they are.
    Returns a new dict; never touches the input."""
    out = {}
    for f, rows in cands.items():
        p = scores.get(f)
        if p is None or len(p) != len(rows): out[f] = list(rows); continue
        new = []
        for (x, y, c), pi in zip(rows, p):
            pi = float(pi)
            if mode == "drop":
                if pi >= t: new.append((x, y, c))
            elif mode == "mult": new.append((x, y, float(c) * pi ** t))
            elif mode == "blend": new.append((x, y, float(c) * (1 - t) + pi * t))
            else: raise ValueError(mode)
        out[f] = new
    return out


def pack(scores_by_frame):
    """{frame: list of p} -> flat arrays for npz (frame index, p) so a 9,000-frame clip stays small"""
    fr, p = [], []
    for f in sorted(scores_by_frame):
        fr += [f] * len(scores_by_frame[f]); p += list(scores_by_frame[f])
    return np.asarray(fr, np.int32), np.asarray(p, np.float16)


def unpack(fr, p):
    out = {}
    for f, pi in zip(fr.tolist(), p.astype(np.float32).tolist()): out.setdefault(f, []).append(pi)
    return {f: np.asarray(v, np.float32) for f, v in out.items()}
