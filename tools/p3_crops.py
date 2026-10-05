"""P3 (5 Oct, free GitHub runner, CPU, no Modal, no detector): cut player crops for the tracklet-joining check.
For every piece in results/qa/p3/p3lab.json, every track that is part of a join candidate gets up to 8 crops over its
life (always its first 2 and last 2 observed frames). Box from the saved foot point + box height (SFK-BP export has no
box height: h = 0.276 * foot_y - 67.7 px, fitted on the Veo pieces, +-17%).
    R2_PUBLIC_URL=... python tools/p3_crops.py              -> results/qa/p3/crops/<piece>.jpg + <piece>.json
    DRY=1 LOCAL_VIDEO=x.mp4 P3_ONLY=reym_726 python tools/p3_crops.py   (local video, k0 = 0, no network)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from tools.p3lab import load_piece, PIECES

OUT = os.environ.get("P3_OUT", "results/qa/p3/crops"); os.makedirs(OUT, exist_ok=True)
CW, CH, PER_ROW = 64, 144, 16
R2 = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
LAB = json.load(open(os.environ.get("P3_LAB", "results/qa/p3/p3lab.json")))


def pick_frames(ks, m=8):
    if len(ks) <= m: return list(ks)
    mid = [ks[i] for i in np.linspace(2, len(ks) - 3, m - 4).astype(int)]
    return sorted(set(ks[:2] + mid + ks[-2:]))


def crop(f, px, h):
    x, y = px; h = max(40.0, h); w = 0.42 * h
    x0, x1, y0, y1 = int(x - w / 2), int(x + w / 2), int(y - 1.08 * h), int(y + 0.06 * h)
    H, W = f.shape[:2]
    c = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0: c[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = f[sy0:sy1, sx0:sx1]
    return cv2.resize(c, (CW, CH), interpolation=cv2.INTER_AREA)


def run(name):
    rows, fps_rows, t0, use_m = load_piece(name)
    R = LAB[name]; need = {int(i) for c in R["cands"] for i in c[:2]}
    want = {}                                                          # row k -> [(id, px, h)]
    for tid in need:
        obs = [(k, r) for k in sorted(rows) for r in rows[k] if r[0] == tid and not r[3] and r[2] is not None]
        ks = pick_frames([k for k, _ in obs]); byk = dict(obs)
        for k in ks:
            r = byk[k]; h = r[4] if r[4] else 0.276 * r[2][1] - 67.7
            want.setdefault(k, []).append((tid, r[2], h))
    match = PIECES[name][0] if name in PIECES else "SFKBP1109"
    src = os.environ.get("LOCAL_VIDEO") or f"{R2}/{match}/video.mp4"
    cap = cv2.VideoCapture(src); vfps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    vk = lambda k: int(round((0 if os.environ.get("DRY") else t0) * vfps + k * vfps / fps_rows))
    order = sorted(want, key=vk); cells = []; imgs = []
    if not order: return
    cap.set(cv2.CAP_PROP_POS_FRAMES, vk(order[0])); cur = vk(order[0]); f = None
    for k in order:
        tgt = vk(k)
        if tgt - cur > 300: cap.set(cv2.CAP_PROP_POS_FRAMES, tgt); cur = tgt; f = None
        while cur <= tgt:
            ok, f = cap.read(); cur += 1
            if not ok: f = None; break
        if f is None: continue
        for tid, px, h in want[k]:
            imgs.append(crop(f, px, h)); cells.append({"id": tid, "k": k, "cell": len(cells)})
    rowsn = (len(imgs) + PER_ROW - 1) // PER_ROW
    sheet = np.zeros((rowsn * CH, PER_ROW * CW, 3), np.uint8)
    for i, im in enumerate(imgs): sheet[(i // PER_ROW) * CH:(i // PER_ROW + 1) * CH, (i % PER_ROW) * CW:(i % PER_ROW + 1) * CW] = im
    cv2.imwrite(f"{OUT}/{name}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump({"cw": CW, "ch": CH, "per_row": PER_ROW, "video_fps": vfps, "cells": cells}, open(f"{OUT}/{name}.json", "w"))
    print(name, len(cells), "crops of", len(need), "tracks", flush=True)


if __name__ == "__main__":
    only = [x for x in os.environ.get("P3_ONLY", "").split(",") if x]
    bad = 0
    for name in LAB:
        if only and name not in only: continue
        try: run(name)
        except Exception as e: print(name, "FAILED", repr(e), flush=True); bad += 1
    sys.exit(1 if bad else 0)
