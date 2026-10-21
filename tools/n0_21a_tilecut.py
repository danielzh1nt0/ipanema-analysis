"""N0-21a: player boxes cut at a tile's inner edge. Runs the app's RF-DETR person detector (FOLLOW_TILES, conf 0.3) on saved
raw frames, merges the tiles the old way (IPANEMA_TILE_CUT=0) and the new way, and writes numbers + a by-eye sheet of every
changed box. Usage: python tools/n0_21a_tilecut.py [out_dir] [glob ...]"""
import os, sys, glob, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from ipanema import tracking as TR

def frames_list(globs):
    fs = []
    for g in globs: fs += sorted(glob.glob(g))
    return fs

class Memo:
    def __init__(self, m): self.m, self.c = m, {}
    def predict(self, crops, threshold):
        out = []
        for c in crops:
            k = hashlib.md5(c.tobytes()).hexdigest() + f"{threshold}"
            if k not in self.c: self.c[k] = self.m.predict([c], threshold=threshold)
            r = self.c[k]; out.append(r[0] if isinstance(r, list) else r)
        return out

def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    i = iw * ih; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

def main(out, globs, det=None):
    det = det or TR.RFDetrPerson(os.environ.get("IPANEMA_RFDETR_SIZE", "medium"))
    det.m = Memo(det.m)
    rows, tiles = [], []; tot = {"frames": 0, "old": 0, "new": 0, "old_on_edge": 0, "new_on_edge": 0, "moved": 0, "lost": 0, "gained": 0}
    for p in frames_list(globs):
        f = cv2.imread(p)
        if f is None: continue
        h, w = f.shape[:2]; edge = int(0.65 * h)
        res = {}
        for mode in ("0", "1"):
            os.environ["IPANEMA_TILE_CUT"] = mode
            d, _ = det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0]; res[mode] = d.xyxy.tolist() if len(d) else []
        os.environ.pop("IPANEMA_TILE_CUT", None)
        old, new = res["0"], res["1"]; tot["frames"] += 1; tot["old"] += len(old); tot["new"] += len(new)
        tot["old_on_edge"] += sum(abs(b[3] - edge) < 2.5 for b in old); tot["new_on_edge"] += sum(abs(b[3] - edge) < 2.5 for b in new)
        un_new = set(range(len(new)))
        for b in old:
            j = max(range(len(new)), key=lambda j: iou(b, new[j]), default=None)
            if j is not None and iou(b, new[j]) > 0.95: un_new.discard(j); continue
            if j is not None and iou(b, new[j]) > 0.3 and j in un_new:
                un_new.discard(j); tot["moved"] += 1; kind = "moved"; nb = new[j]
            else: tot["lost"] += 1; kind = "lost"; nb = None
            rows.append({"frame": p, "kind": kind, "old": [round(x, 1) for x in b], "new": [round(x, 1) for x in nb] if nb else None})
            tiles.append((f, b, nb, kind))
        for j in un_new:
            tot["gained"] += 1; rows.append({"frame": p, "kind": "gained", "old": None, "new": [round(x, 1) for x in new[j]]}); tiles.append((f, None, new[j], "gained"))
    os.makedirs(out, exist_ok=True)
    json.dump({"totals": tot, "changes": rows}, open(f"{out}/changes.json", "w"), indent=1)
    sheet(tiles, f"{out}/sheet.jpg")
    print(json.dumps(tot)); return tot, rows

def sheet(tiles, path, per_row=8, tw=150):
    cells = []
    for i, (f, ob, nb, kind) in enumerate(tiles):
        bs = [b for b in (ob, nb) if b is not None]
        x0 = int(min(b[0] for b in bs)); y0 = int(min(b[1] for b in bs)); x1 = int(max(b[2] for b in bs)); y1 = int(max(b[3] for b in bs))
        hh = y1 - y0; H, W = f.shape[:2]
        a, b_, c, d = max(0, x0 - 25), max(0, y0 - 15), min(W, x1 + 25), min(H, y1 + hh // 2 + 15)
        cr = f[b_:d, a:c].copy()
        if ob is not None: cv2.rectangle(cr, (int(ob[0]) - a, int(ob[1]) - b_), (int(ob[2]) - a, int(ob[3]) - b_), (0, 0, 255), 1)       # red = old
        if nb is not None: cv2.rectangle(cr, (int(nb[0]) - a - 1, int(nb[1]) - b_ - 1), (int(nb[2]) - a + 1, int(nb[3]) - b_ + 1), (0, 255, 0), 1)   # green = new
        cr = cv2.resize(cr, (tw, max(1, int(tw * cr.shape[0] / cr.shape[1]))))
        cr = cr[:260] if cr.shape[0] > 260 else np.pad(cr, ((0, 260 - cr.shape[0]), (0, 0), (0, 0)))
        cv2.putText(cr, f"{i} {kind}", (3, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cells.append(cr)
    if not cells: return
    while len(cells) % per_row: cells.append(np.zeros_like(cells[0]))
    cv2.imwrite(path, np.vstack([np.hstack(cells[i:i + per_row]) for i in range(0, len(cells), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 85])

if __name__ == "__main__":
    try: import rfdetr  # noqa
    except ImportError:
        import subprocess
        subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
    out = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-21a"
    main(out, sys.argv[2:] or ["results/qa/p2/*/raw_*.jpg", "results/qa/players/SFKBP1109/f*.jpg", "results/qa/players/p15u-vs-aik-2026-09-21-bd09/f*.jpg"])
