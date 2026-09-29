"""P1 (30 Sep): load a piece saved by kaggle/kitcrops_p1.py and feed it to the kit step exactly as tracktest does, offline.
Each person becomes a tiny 'frame' (its crop); the frame-wide values the kit step needs (grass colour, pitch-edge row) are
looked up from what Kaggle saved. use(): with P.patched(): K.KitTeamModel().fit_frames(P.frames_boxes(piece))."""
import os, sys, gzip, pickle, contextlib, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
META = {}
def load(path):
    with gzip.open(path, "rb") as fh: return pickle.load(fh)
def pack(f, boxes):
    """per person: jpg crop of the box widened by a quarter each side + the strip below the feet; box in crop coords;
    pitch-edge row at the box centre (full-frame coords)"""
    H, Wd = f.shape[:2]; g = K.grass_lab(f); top = K.pitch_top(f); ppl = []
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b]; w = max(4, x2 - x1); h = y2 - y1
        cx1, cy1 = int(max(0, x1 - w / 4 - 2)), int(max(0, y1 - 2)); cx2, cy2 = int(min(Wd, x2 + w / 4 + 3)), int(min(H, y2 + max(3, 0.08 * h) + 3))
        ok, jpg = cv2.imencode(".png", f[cy1:cy2, cx1:cx2], [cv2.IMWRITE_PNG_COMPRESSION, 9])                    # lossless: the kit step reads exact colours
        ppl.append({"jpg": jpg.tobytes(), "box": [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], "full_box": [x1, y1, x2, y2], "origin": [cx1, cy1],
                    "top": float(top[int(min(len(top) - 1, max(0, (x1 + x2) / 2)))]), "at_bottom": y2 >= H - 1})
    return {"H": H, "W": Wd, "g": g.tolist(), "people": ppl}
def frames_boxes(piece):
    """-> list of (crop, [box]) (one per person) + the flat list of people in the same order"""
    out, flat = [], []
    for fi, fr in enumerate(piece["frames"]):
        for pi, p in enumerate(fr["people"]):
            im = cv2.imdecode(np.frombuffer(p["jpg"], np.uint8), cv2.IMREAD_COLOR)
            META[id(im)] = {"g": np.array(fr["g"]), "top": p["top"], "oy": p["origin"][1], "H": fr["H"], "keep": im}
            out.append((im, np.array([p["box"]], float))); flat.append((fi, pi, p))
    return out, flat
@contextlib.contextmanager
def patched():
    og, op, of = K.grass_lab, K.pitch_top, K.feet_on_pitch
    K.grass_lab = lambda f: META[id(f)]["g"] if id(f) in META else og(f)
    K.pitch_top = lambda f, *a, **k: np.zeros(1) if id(f) in META else op(f, *a, **k)
    def feet(frame, box, top=None, margin=0.011):
        m = META.get(id(frame))
        if m is None: return of(frame, box, top, margin)
        return bool(box[3] + m["oy"] - m["top"] >= margin * m["H"])
    K.feet_on_pitch = feet
    try: yield
    finally: K.grass_lab, K.pitch_top, K.feet_on_pitch = og, op, of
def fit(piece, **kw):
    fb, flat = frames_boxes(piece); lines = []
    with patched():
        tm = K.KitTeamModel().fit_frames(fb, log=lambda m: lines.append(m), **kw); tm.offpitch = piece["match"] != "SFKBP1109_s1200"
        reading = [tm.predict_batch(im, b)[0] for im, b in fb]
    return tm, lines, reading, flat, fb
