"""BALL-EDGE (23 Oct, free runner): picture sheets where the picker's ball differs between old and IPANEMA_BALL_EDGE=1
(results/qa/ball-edge/grade.json spells, picks.json). Per spell, the middle frame: left cell around the OLD pick (red ring),
right cell around the NEW pick (green ring); both rings drawn in both cells when they fit. No pick = grey 'none'.
    R2_PUBLIC_URL=... python tools/ball_edge_sheet.py
    dry run: LOCAL_CLIP=v.mp4 python tools/ball_edge_sheet.py grade.json picks.json out_dir"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from tools.ball_edge_shard import CLIPS, fetch

OLD, NEW = "old (re-detected)", "new (BALL_EDGE=1)"


def crop(fr, c, rings, label, half=80, scale=2):
    h, w = fr.shape[:2]; cell = np.zeros((2 * half * scale, 2 * half * scale, 3), np.uint8)
    if c is not None:
        x, y = int(round(c[0])), int(round(c[1])); p = cv2.copyMakeBorder(fr, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
        cell = cv2.resize(p[y:y + 2 * half, x:x + 2 * half], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        for r, col in rings:
            if r is None: continue
            bx, by = (r[0] - x + half) * scale, (r[1] - y + half) * scale
            if 0 <= bx < cell.shape[1] and 0 <= by < cell.shape[0]: cv2.circle(cell, (int(bx), int(by)), 18, col, 2)
    else: cv2.putText(cell, "none", (half * scale - 30, half * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    cv2.rectangle(cell, (0, 0), (cell.shape[1], 16), (0, 0, 0), -1); cv2.putText(cell, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return cell


def main(grade="results/qa/ball-edge/grade.json", picks="results/qa/ball-edge/picks.json", out="results/qa/ball-edge"):
    G, PK = json.load(open(grade)), json.load(open(picks)); os.makedirs(out, exist_ok=True); done = {}
    for key, r in G.items():
        want = {}
        for i, (a, b) in enumerate(r["pick_differs_spells"]): want[(a + b) // 2] = i
        cap = cv2.VideoCapture(fetch(CLIPS[key])); k = 0; rows = []; last = max(want) if want else -1
        while k <= last:
            if k in want:
                ok, fr = cap.read()
                if not ok: break
                o, n = PK[key][OLD].get(str(k)), PK[key][NEW].get(str(k)); rings = [(o, (0, 0, 255)), (n, (0, 255, 0))]
                a, b = r["pick_differs_spells"][want[k]]; t = f"{key}{want[k]} f{k} ({a}-{b})"
                rows.append(np.hstack([crop(fr, o, rings, t + " OLD"), crop(fr, n, rings, t + " NEW")]))
            elif not cap.grab(): break
            k += 1
        cap.release()
        for s in range(0, len(rows), 12):
            part = rows[s:s + 12]
            if len(part) % 2: part.append(np.zeros_like(part[0]))
            img = np.vstack([np.hstack(part[i:i + 2]) for i in range(0, len(part), 2)])
            cv2.imwrite(f"{out}/diff_{key}_{s // 12:02d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        done[key] = len(rows); print(key, len(rows), "spells drawn", flush=True)
    return done


if __name__ == "__main__":
    main(*sys.argv[1:4])
