"""V3 (5 Oct, local, $0): why is the player on the ball missing on Vallentuna? On the 20 by-eye carrier moments
(results/qa/v3/carriers from tools/v3_frames.py, free runner) find, for each detector setting, the box of the player on
the ball (the ball sits at his feet: inside the box widened by half its width, between 30% of its height and 35% below
its feet), whether the app's export has a player there (results/volume/.../frames_*.json, 10 per s), and which team the
match kit model (fitted on the 36 Vallentuna fit frames of results/qa/f1c/frames, as in the app) gives that box, with the
5 Oct hue mode on and off. Extra people a lower threshold adds = boxes with no base box over them (IoU < 0.3).
Sheets: 2x zoom around the ball per moment, boxes per setting (base green, low >= 0.15 yellow, gamma cyan, clahe magenta,
conf printed), export dots (red A dark, blue B red kit).
    python tools/v3lab.py  -> results/qa/v3/v3lab.json + results/qa/v3/sheets/*.jpg"""
import os, sys, json, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K

M = "p15u-vs-vallentuna-2026-10-03-6cce"; D = os.environ.get("V3_DIR", "results/qa/v3/carriers"); OUT = os.environ.get("V3_LAB_OUT", "results/qa/v3")
EXPORT = f"results/volume/runs/matches/{M}"
LOW = (0.15, 0.2)
# By eye on the sheets (5 Oct): the player on the ball when the ball is a stride ahead (c03, c05), the key's ball point is
# ~25 px off (b00) or two players overlap at the ball (c26: the red #14, not the dark player behind): his foot point, full res.
EYE = {"c03": (1133, 582), "b00": (265, 524), "c05": (1099, 540), "c26": (1057, 500)}
COL = {"base": (0, 220, 0), "low": (0, 230, 255), "gamma": (255, 255, 0), "clahe": (255, 0, 255)}

def carrier(boxes, ball, conf=0.0):
    """index of the box whose player has the ball at his feet, else None"""
    bx, by = ball; best, bd = None, 1e9
    for i, b in enumerate(boxes):
        x1, y1, x2, y2, c = b[:5]; w, h = x2 - x1, y2 - y1
        if c < conf: continue
        if x1 - 0.5 * w <= bx <= x2 + 0.5 * w and y1 + 0.3 * h <= by <= y2 + 0.35 * h:
            d = abs(bx - (x1 + x2) / 2) + abs(by - y2)
            if d < bd: best, bd = i, d
    return best

def carrier_eye(boxes, foot, conf=0.0, r=30):
    c = [(abs((b[0] + b[2]) / 2 - foot[0]) + abs(b[3] - foot[1]), i) for i, b in enumerate(boxes) if b[4] >= conf]
    c = [x for x in c if x[0] <= r]
    return min(c)[1] if c else None

def find(boxes, m, conf=0.0):
    return carrier_eye(boxes, EYE[m["id"]], conf) if m["id"] in EYE else carrier(boxes, m["ball"], conf)

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

def extras(base, other, conf):
    return [b for b in other if b[4] >= conf and all(iou(b, a) < 0.3 for a in base)]

def export_players(t):
    best = None
    for fn in sorted(glob.glob(f"{EXPORT}/frames_*.json")):
        F = json.load(open(fn))
        if not (F["t_start"] - 1 <= t <= F["t_end"] + 1): continue
        for fr in F["frames"]:
            if best is None or abs(fr["t"] - t) < abs(best["t"] - t): best = fr
    return best

def exported_near(fr, box):
    """the exported player whose foot point is on this box's feet (within 40% of its height)"""
    if not fr or box is None: return None
    x1, y1, x2, y2 = box[:4]; fx, fy = (x1 + x2) / 2, y2; h = y2 - y1
    c = [(abs(p["px"][0] - fx) + abs(p["px"][1] - fy), p) for p in fr["players"] if p.get("px")]
    c = [x for x in c if x[0] <= max(25, 0.4 * h)]
    return min(c, key=lambda x: x[0])[1] if c else None

def fit_model():
    d = f"results/qa/f1c/frames/{M}/fit"; B = json.load(open(f"{d}/boxes.json"))
    fb = [(cv2.imread(f"{d}/{fn}"), np.array(B[fn]).reshape(-1, 4)) for fn in sorted(B)]
    return K.KitTeamModel().fit_frames(fb, log=lambda *a: None)

def team_of(m, f, box, hue):
    old = {k: os.environ.get(k) for k in ("IPANEMA_KIT_HUE", "IPANEMA_KIT_HUEMODE")}
    for k in old: os.environ[k] = "1" if hue else "0"
    try: return m.predict_batch(f, [list(box[:4])])[0]
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

def sheet(f, m, B, fr, row, path):
    g = f.copy()
    for s in ("base", "low", "gamma", "clahe"):
        for b in B[s]:
            if s != "base" and (b[4] < LOW[0] or any(iou(b, a) >= 0.3 for a in B["base"])): continue
            cv2.rectangle(g, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COL[s], 1)
            cv2.putText(g, f"{b[4]:.2f}", (int(b[0]), int(b[1]) - 2), 0, 0.35, COL[s], 1)
    for p in (fr or {}).get("players", []):
        if p.get("px"): cv2.circle(g, (int(p["px"][0]), int(p["px"][1])), 5, (0, 0, 255) if p["team"] == "A" else (255, 120, 0), -1)
    bx, by = m["ball"]; cv2.circle(g, (int(bx), int(by)), 14, (255, 255, 255), 1)
    R = 220; x0 = int(min(max(bx - R, 0), 1920 - 2 * R)); y0 = int(min(max(by - R, 0), 1080 - 2 * R))
    z = cv2.resize(g[y0:y0 + 2 * R, x0:x0 + 2 * R], (880, 880), interpolation=cv2.INTER_LINEAR)
    txt = f"{m['id']} t={m['t']:.1f} owner={m['owner']} | " + " ".join(f"{k}={v}" for k, v in row.items() if k in ("base", "low15", "gamma", "clahe", "exported", "team_hue", "team_old"))
    cv2.rectangle(z, (0, 0), (880, 26), (0, 0, 0), -1); cv2.putText(z, txt[:110], (6, 18), 0, 0.5, (255, 255, 255), 1)
    cv2.imwrite(path, z, [cv2.IMWRITE_JPEG_QUALITY, 88])

def main():
    MS = json.load(open(f"{D}/moments.json"))["moments"]; BX = json.load(open(f"{D}/boxes.json"))
    model = fit_model(); os.makedirs(f"{OUT}/sheets", exist_ok=True); rows = []; ext = {f"{s}{c}": [] for s in ("low", "gamma", "clahe") for c in LOW}
    for m in MS:
        if m.get("file") not in BX: continue
        f = cv2.imread(f"{D}/{m['file']}"); B = BX[m["file"]]; fr = export_players(m["t"])
        ib = find(B["base"], m); bb = B["base"][ib] if ib is not None else None
        row = {"id": m["id"], "t": m["t"], "owner": m["owner"], "by_eye": m["id"] in EYE, "base": ib is not None, "base_conf": bb[4] if bb else None,
               "low15": find(B["low"], m, 0.15) is not None, "low20": find(B["low"], m, 0.2) is not None,
               "gamma": find(B["gamma"], m, 0.15) is not None, "clahe": find(B["clahe"], m, 0.15) is not None}
        for s in ("low", "gamma", "clahe"):
            i = find(B[s], m); row[f"{s}_conf"] = B[s][i][4] if i is not None else None
        e = exported_near(fr, bb); row["exported"] = (e["team"] if e else "-") if bb else "no box"
        row["team_hue"] = team_of(model, f, bb, True) if bb else None; row["team_old"] = team_of(model, f, bb, False) if bb else None
        row["export_t"] = fr["t"] if fr else None; row["export_n"] = len(fr["players"]) if fr else 0
        for s in ("low", "gamma", "clahe"):
            for c in LOW: ext[f"{s}{c}"].append(len(extras(B["base"], B[s], c)))
        rows.append(row); sheet(f, m, B, fr, row, f"{OUT}/sheets/{m['id']}.jpg"); print(row, flush=True)
    n = len(rows)
    summ = {"moments": n,
            "carrier boxed": {k: sum(r[k] for r in rows) for k in ("base", "low15", "low20", "gamma", "clahe")},
            "carrier exported (base box, any team)": sum(r["exported"] in ("A", "B") for r in rows),
            "carrier exported in the right team": sum(r["exported"] == r["owner"] for r in rows),
            "kit model right on the carrier box (hue on / off)": [sum(r["team_hue"] == r["owner"] for r in rows), sum(r["team_old"] == r["owner"] for r in rows)],
            "extra people per frame vs base": {k: round(float(np.mean(v)), 2) for k, v in ext.items()}}
    json.dump({"summary": summ, "rows": rows}, open(f"{OUT}/v3lab.json", "w"), indent=1); print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()
