"""N1a (6 Oct, worker, local $0): on Solberga at night the nearest white players read far brighter than the white team's
centre and are removed as a 'non-team kit' (referee / staff). New rule kits.bright_team: on grounds without calibration a
'neither' reading with a kit's colour (a*b*) that is at least as light as that kit joins it (env IPANEMA_KIT_BRIGHT =
share of the kits' colour gap, 0 = old).
For every P2 piece (results/qa/p2/<piece>/: the 8 key frames + the detector's boxes; Solberga 1500 = the nightly spot, its
frames from results/qa/tracktest_solberga-...) the kit model is fitted on the key frames like tools/p2clab.py, then every
person is read with the rule off and at a few settings. People whose reading changes are cut into a sheet to check by eye.
Graded: Reymersholm 252 labelled night players (piece models), Reymersholm 4227 88 labelled people.
python tools/n1alab.py -> results/qa/n1a/n1alab.json + results/qa/n1a/changed_<setting>.jpg"""
import os, sys, json, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
P2 = "results/qa/p2"; OUT = "results/qa/n1a"; os.makedirs(OUT, exist_ok=True)
SETTINGS = [s for s in os.environ.get("N1A_SETTINGS", "0,0.4,0.5,0.6").split(",")]
RAWDIR = {"solberga-vs-p09-norrviken-2026-09-11_1500": "results/qa/tracktest_solberga-vs-p09-norrviken-2026-09-11"}

def frames(d, piece):
    kd = json.load(open(d + "keydets.json")); rd = RAWDIR.get(piece, d.rstrip("/"))
    return [(int(j), cv2.imread(f"{rd}/raw_{int(j):04d}.jpg"), np.array(bs)) for j, bs in kd.items() if bs]

def read_all(m, fb, s):
    old = os.environ.get("IPANEMA_KIT_BRIGHT"); os.environ["IPANEMA_KIT_BRIGHT"] = s
    try: return [m.predict_batch(f, b) for _, f, b in fb]
    finally:
        os.environ.pop("IPANEMA_KIT_BRIGHT", None)
        if old is not None: os.environ["IPANEMA_KIT_BRIGHT"] = old

def tile(f, b, txt):
    x1, y1, x2, y2 = [int(v) for v in b]; h = y2 - y1; pad = max(6, h // 3)
    c = f[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
    c = cv2.resize(c, (90, 150)); c = np.vstack([c, np.zeros((22, 90, 3), np.uint8)])
    cv2.putText(c, txt, (2, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1); return c

def sheet(tiles, name, per_row=12):
    if not tiles: return
    while len(tiles) % per_row: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(name, np.vstack([np.hstack(tiles[r:r + per_row]) for r in range(0, len(tiles), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 88])

def team_names(m):
    """which letter is the light kit (tracktest calls B the lighter team)"""
    return {"A": "dark", "B": "light"}

if __name__ == "__main__":
    os.environ.pop("IPANEMA_KIT_BRIGHT", None)
    res = {"settings": SETTINGS, "pieces": {}}; changed = {s: [] for s in SETTINGS[1:]}; listing = {s: [] for s in SETTINGS[1:]}; refs = []
    only = os.environ.get("N1A_ONLY", "")
    for d in sorted(glob.glob(f"{P2}/*{only}*/")):
        piece = d.rstrip("/").split("/")[-1]; fb = frames(d, piece)
        logs = []; m = K.KitTeamModel().fit_frames([(f, b) for _, f, b in fb], log=logs.append); m.offpitch = True
        reads = {s: read_all(m, fb, s) for s in SETTINGS}; r = {"model": [l[6:] for l in logs if "learned" in l]}
        base = reads[SETTINGS[0]]
        for s in SETTINGS:
            flat = [x for p in reads[s] for x in p]
            r[s] = {"per frame A/B/neither/off": [float(np.median([p.count(t) for p in reads[s]])) for t in "ABKO"],
                    "changed vs off": sum(a != b for a, b in zip([x for p in base for x in p], flat))}
            if "reymersholm" in piece:
                import tools.p2blab as B
                os.environ["IPANEMA_KIT_BRIGHT"] = s
                r[s]["252 labelled night players"] = B.grade([m.predict_batch(B.KF[fn], np.array([B.KB[fn][j]]))[0] for (fn, j), _ in B.KEY])["team right"]
                os.environ.pop("IPANEMA_KIT_BRIGHT", None)
            if piece.endswith("_4227"):
                import tools.p2clab as C
                r[s]["88 labelled people"] = C.grade(flat)
        for s in SETTINGS[1:]:
            for (j, f, b), p0, p1 in zip(fb, base, reads[s]):
                for i, (a, c) in enumerate(zip(p0, p1)):
                    if a != c:
                        n = len(changed[s]); changed[s].append(tile(f, b[i], f"{n} {a}->{c}"))
                        listing[s].append({"n": n, "piece": piece, "frame": j, "box": [round(float(v), 1) for v in b[i]], "from": a, "to": c})
        if any(r[s]["changed vs off"] for s in SETTINGS[1:]):   # reference: what each team looks like in this piece (rule off, tallest people)
            for t in "AB":
                ex = sorted([(b[i][3] - b[i][1], j, i) for (j, f, b), p in zip(fb, base) for i, x in enumerate(p) if x == t], reverse=True)[:6]
                fr = {j: (f, b) for j, f, b in fb}
                refs.append([tile(fr[j][0], fr[j][1][i], f"{piece[4:10]} {t}") for _, j, i in ex])
        res["pieces"][piece] = r
        print(piece, "|", json.dumps({s: r[s] for s in SETTINGS}), flush=True)
    for s in SETTINGS[1:]:
        sheet(changed[s], f"{OUT}/changed_{s}.jpg"); res[f"changed people {s}"] = listing[s]
    sheet([t for row in refs for t in row + [np.zeros_like(row[0])] * (6 - len(row))] if refs else [], f"{OUT}/team_reference.jpg")
    json.dump(res, open(f"{OUT}/n1alab.json", "w"), indent=1)
