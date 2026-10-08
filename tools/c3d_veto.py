"""C3d (8 Oct): can a picture check veto the wrong TRUSTED camera rows? On the 60 by-eye graded Vallentuna seconds
(results/qa/c3d/vall/grades.json: g = lines on the paint, r = rough 10-30 px, o = clearly wrong, ? = can't tell) and their raw
frames (results/qa/c3d/vall_raw/<id>.jpg), three scores per second:
  far_white  - the far cost as measured over the whole match (white-line mask; misses paint in the long shadows)
  far_shade  - the same with the shade-aware paint mask (c3c_nearline.paint_mask: shaded paint reads sky-blue)
  move_px    - re-fit the pose to the far paint (shade-aware, small local search) and measure how far the drawn far lines
               move (median px over the far model points in view). A right pose in shade moves little; a wrong one jumps.
    python tools/c3d_veto.py results/qa/c3d/vall_raw results/qa/c3d/vall/grades.json -> results/qa/c3d/veto.json"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import lines as LN
from c3c_fit import Frame, model_pts, refit, CAP
from c3c_nearline import paint_mask

class ShadeFrame(Frame):
    """Frame whose far-line distance map comes from the shade-aware paint mask (whole picture below the top 15%)"""
    def __init__(self, img):
        super().__init__(img); m = paint_mask(img); m[: int(0.15 * self.h)] = 0
        self.dt_white = self.dt; self.dt = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)

def scores(img, camera, pose, far, near):
    fr = ShadeFrame(cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)); pose = np.asarray(pose, float)
    dw = fr.dists(camera, pose, far, fr.dt_white); ds = fr.dists(camera, pose, far)
    q = refit(fr, camera, pose, far, near, 0.0)
    a = LN.project(camera, pose, far, 1280, 720); b = LN.project(camera, np.asarray(q, float), far, 1280, 720)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (a[:, 0] >= 0) & (a[:, 0] < 1280) & (a[:, 1] >= 0) & (a[:, 1] < 720)
    mv = float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1))) if ok.sum() >= 30 else None
    return {"far_white": round(float(np.minimum(dw, CAP).mean()), 2) if len(dw) >= 30 else None,
            "far_shade": round(float(np.minimum(ds, CAP).mean()), 2) if len(ds) >= 30 else None,
            "far_shade_after": round(float(np.minimum(fr.dists(camera, np.asarray(q, float), far), CAP).mean()), 2),
            "move_px": round(mv, 1) if mv is not None else None}

def separation(rows, key):
    """smallest score among clearly wrong rows, how many good / rough rows reach it, and the share of good rows below it"""
    o = [r[key] for r in rows if r["grade"] == "o" and r[key] is not None]; g = [r[key] for r in rows if r["grade"] == "g" and r[key] is not None]
    rr = [r[key] for r in rows if r["grade"] == "r" and r[key] is not None]
    if not o: return None
    thr = min(o); return {"wrong": sorted(o), "thr": thr, "good_vetoed": sum(x >= thr for x in g), "good": len(g),
                          "rough_vetoed": sum(x >= thr for x in rr), "rough": len(rr)}

def main(argv=sys.argv[1:]):
    d, gpath = argv[0], argv[1]; S = json.load(open(f"{d}/sample.json")); G = json.load(open(gpath)); cam = S["camera"]
    far, near = model_pts(cam, 106.0, 64.0); rows = []
    for q in S["rows"]:
        p = f"{d}/{q['id']}.jpg"
        if not os.path.exists(p): continue
        s = scores(cv2.imread(p), cam, q["pose"], far, near); s.update({"id": q["id"], "t": q["t"], "grade": G.get(q["id"], "?")}); rows.append(s)
        print(s, flush=True)
    out = {"rows": rows, "separation": {k: separation(rows, k) for k in ("far_white", "far_shade", "move_px")}}
    json.dump(out, open(os.path.join(os.path.dirname(d), "veto.json"), "w"), indent=1); print(json.dumps(out["separation"], indent=1))

if __name__ == "__main__": main()
