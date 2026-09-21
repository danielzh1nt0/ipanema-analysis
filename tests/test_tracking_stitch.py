"""The linear track stitcher must give exactly the old all-pairs result (verified 40/40 on 21 Sep)."""
import numpy as np
from ipanema.tracking import _stitch, reposition
from ipanema.calibration import to_m

def _old(segs, fps):
    remap = {}
    def root(t):
        while t in remap: t = remap[t]
        return t
    for tid in sorted(segs, key=lambda t: segs[t]["start"]):
        B = segs[tid]; best = None
        for aid, A in segs.items():
            if aid == tid or A["team"] != B["team"]: continue
            ra = root(aid); Aend = max(segs[q]["end"] for q in segs if root(q) == ra); gap = (B["start"] - Aend) / fps
            if not (0 < gap <= 2.5): continue
            dist = np.linalg.norm(A["p1"] - B["p0"]) if A["end"] == Aend else 99
            if dist <= 2.0 + 6.0 * gap and (best is None or dist < best[1]): best = (ra, dist)
        if best: remap[tid] = best[0]
    return remap

def test_stitch_identical_to_old():
    for seed in range(25):
        rng = np.random.RandomState(seed); segs = {}
        for p in range(10):
            t = rng.randint(0, 60); pos = rng.uniform(0, 100, 2)
            while t < 900:
                L = rng.randint(3, 120); tid = int(rng.randint(0, 10**6))
                while tid in segs: tid += 1
                segs[tid] = {"team": "A" if p < 5 else "B", "start": t, "end": t + L, "p0": pos.copy(), "p1": pos + rng.normal(0, 1.5, 2)}
                pos = segs[tid]["p1"] + rng.normal(0, 0.8, 2); t = t + L + rng.randint(1, 90)
        for _ in range(40):
            tid = int(rng.randint(10**6, 2 * 10**6)); st = rng.randint(0, 900)
            segs[tid] = {"team": rng.choice(["A", "B"]), "start": st, "end": st + rng.randint(0, 40), "p0": rng.uniform(0, 100, 2), "p1": rng.uniform(0, 100, 2)}
        assert _stitch(segs, 29.97) == _old(segs, 29.97), f"seed {seed}"

def test_reposition_exact():
    Hold = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1.0]]); Hnew = np.array([[12.0, 1.0, 30.0], [0.5, 9.0, 20.0], [0, 0.001, 1.0]])
    per = {k: [[j, "A", to_m(Hold, [[100.0 + 50 * j + k, 300.0 + 20 * j]])[0], np.array([100.0 + 50 * j + k, 300.0 + 20 * j]), np.zeros(4), j == 0] for j in range(4)] for k in range(3)}
    per[3] = []
    out = reposition(per, {k: Hnew for k in range(4)})
    for k in range(3):
        for j in range(4):
            assert np.allclose(out[k][j][2], to_m(Hnew, [per[k][j][3]])[0])
            assert out[k][j][0] == per[k][j][0] and out[k][j][1] == per[k][j][1] and out[k][j][5] == per[k][j][5]
    assert out[3] == []
