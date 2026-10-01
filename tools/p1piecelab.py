"""P1 (1 Oct): striped kits on the REAL tracking pieces. The P1 v2 switch (try far-pair + mean colour only when the default
fit leaves out a group >= 1.0 x the team gap) never fired on the 21 real 60-s pieces (their 'missed' is 0.40-0.90), so near
striped Spånga players stayed 'neither'. This compares the default reading with far-pair + mean on every piece saved by
kaggle/kitcrops_p1.py (the kit step's own input, offline copy proven identical by tests/test_p1crops.py) and writes, per
piece, a sheet of only the people whose reading changes, to grade by eye:
  rows 'neither -> team', 'team -> neither', 'A -> B', 'B -> A' (A = the darker team).
Counts are people on the pitch (off-pitch 'O' left out). Free, local: python tools/p1piecelab.py [piece-substring]"""
import os, sys, glob, json, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import p1crops as P
SRC = f"{ROOT}/results/kaggle/kitcrops_p1"; OUT = f"{ROOT}/results/qa/p1/pieces"
COMBOS = {"default": {"IPANEMA_KIT_PAIR": "big", "IPANEMA_TORSO_STAT": "median", "IPANEMA_KIT_AUTO": "0"},
          "far+mean": {"IPANEMA_KIT_PAIR": "far", "IPANEMA_TORSO_STAT": "mean", "IPANEMA_KIT_AUTO": "0"},
          "auto (v3 gate)": {"IPANEMA_KIT_PAIR": "big", "IPANEMA_TORSO_STAT": "median", "IPANEMA_KIT_AUTO": "1"}}

def reading(piece, env):
    keep = {k: os.environ.get(k) for k in list(env) + ["IPANEMA_KIT_AUTO"]}
    os.environ.update(env)
    try:
        tm, lines, rd, flat, fb = P.fit(piece)
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return rd, flat, lines

def crop(p, w=40, h=80):
    im = cv2.imdecode(np.frombuffer(p["jpg"], np.uint8), cv2.IMREAD_COLOR); x1, y1, x2, y2 = [int(v) for v in p["box"]]
    return cv2.resize(im[max(0, y1):y2, max(0, x1):x2], (w, h), interpolation=cv2.INTER_NEAREST)

def changes(a, b):
    """people (index) by kind of change between two readings"""
    out = {"neither -> team": [], "team -> neither": [], "A -> B": [], "B -> A": []}
    for i, (x, y) in enumerate(zip(a, b)):
        if x == "O" or y == "O" or x == y: continue
        if x == "K": out["neither -> team"].append(i)
        elif y == "K": out["team -> neither"].append(i)
        else: out[f"{x} -> {y}"].append(i)
    return out

def sheet(flat, ch, path, per_row=24):
    rows = []
    for name, idx in ch.items():
        sel = [idx[j] for j in np.linspace(0, len(idx) - 1, min(len(idx), per_row)).astype(int)] if idx else []
        cs = [crop(flat[i][2]) for i in sel]
        while len(cs) < per_row: cs.append(np.zeros((80, 40, 3), np.uint8))
        hdr = np.zeros((20, 40 * per_row, 3), np.uint8); cv2.putText(hdr, f"{name}: {len(idx)}", (4, 15), 0, 0.5, (255, 255, 255), 1)
        rows += [hdr, np.hstack(cs)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

def run(filt=""):
    os.makedirs(OUT, exist_ok=True); res = {}
    for path in sorted(glob.glob(f"{SRC}/*.pkl.gz")):
        tag = os.path.basename(path)[:-len(".pkl.gz")]
        if filt not in tag: continue
        piece = P.load(path); R = {}
        for name, env in COMBOS.items():
            rd, flat, lines = reading(piece, env); on = [r for r in rd if r != "O"]
            R[name] = rd; res.setdefault(tag, {})[name] = {"A": on.count("A"), "B": on.count("B"), "neither": on.count("K"), "fit": lines[-1]}
        ch = changes(R["default"], R["far+mean"]); res[tag]["changed"] = {k: len(v) for k, v in ch.items()}
        res[tag]["auto switched"] = R["auto (v3 gate)"] == R["far+mean"] and R["far+mean"] != R["default"]
        res[tag]["auto = default"] = R["auto (v3 gate)"] == R["default"]
        sheet(flat, ch, f"{OUT}/{tag}.jpg"); print(tag, res[tag]['changed'], 'switched' if res[tag]['auto switched'] else ('default' if res[tag]['auto = default'] else 'OTHER'), flush=True)
    json.dump(res, open(f"{OUT}/piecelab.json", "w"), indent=1)
    return res

if __name__ == "__main__": run(sys.argv[1] if len(sys.argv) > 1 else "")
