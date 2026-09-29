"""P1 (30 Sep): the offline copy of a piece (kaggle/kitcrops_p1.py -> tools/p1crops.py) must give the kit step exactly the
same answer as the real frames, else the striped-kit fix would again be tuned on data that isn't what the pipeline sees."""
import os, sys, json, gzip, pickle, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from ipanema import kits as K
import p1crops as P

def test_offline_copy_gives_the_same_kit_fit(tmp_path):
    R = f"{ROOT}/results/qa/kitprobe/p15u-vs-spanga-2026-09-25"; B = json.load(open(f"{R}/boxes.json"))
    fb = [(cv2.imread(f"{R}/{fn}"), np.array([b for b in B[fn] if b[3] - b[1] >= 22], float)) for fn in sorted(B)[:12]]
    lines = []; tm = K.KitTeamModel().fit_frames(fb, log=lines.append); tm.offpitch = True
    real = [x for f, b in fb for x in tm.predict_batch(f, b)]
    piece = {"match": "p15u-vs-spanga-2026-09-25", "frames": [P.pack(f, b) for f, b in fb]}
    p = tmp_path / "x.pkl.gz"
    with gzip.open(p, "wb") as fh: pickle.dump(piece, fh)
    tm2, lines2, reading, flat, _ = P.fit(P.load(p))
    assert lines2 == lines and reading == real and len(flat) == len(real)

def test_patch_is_undone():
    og = K.grass_lab
    with P.patched(): assert K.grass_lab is not og
    assert K.grass_lab is og
