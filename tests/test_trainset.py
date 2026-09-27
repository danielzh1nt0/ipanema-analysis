import os, json, pickle, tempfile, numpy as np
from ipanema import trainset as TS

def _fake_piece(root, match, start_s, fps=29.97, seed=0):
    rng = np.random.RandomState(seed); d = f"{root}/cache/{match}_s{start_s}_d300"; os.makedirs(d)
    wasb, clk = {}, {}
    for k in range(600):
        w = []
        if 50 <= k < 400 and rng.rand() > 0.1: w.append((400 + 3.0 * k, 600 - 0.5 * k, 0.6))         # a moving ball
        w.append((1700 + rng.randn(), 150 + rng.randn(), 0.5))                                        # static clutter
        wasb[k] = w
        if k % 3 == 0:
            clk[k] = [(400 + 3.0 * k + 2, 600 - 0.5 * k, 0.5)] if 50 <= k < 400 else []
            if k == 450: clk[k] = [(100, 100, 0.9)]                                                  # a sure disagreement
    pickle.dump(clk, open(f"{d}/ball_cands_clicks_st3_fz0.pkl", "wb")); pickle.dump(wasb, open(f"{d}/ball_cands_wasb_x.pkl", "wb"))
    json.dump({"start_s": start_s}, open(f"{d}/ball_piece_done.json", "w"))

def test_labels_from_pieces():
    root = tempfile.mkdtemp(); _fake_piece(root, "m", 0); _fake_piece(root, "m", 300, seed=1)
    ps = TS.piece_caches(root, "m"); assert [p[0] for p in ps] == [0, 300] and ps[1][1] == round(300 * 29.97)
    ts = TS.match_trainset(ps)
    assert ts["labels"] and all(abs(l["x"] - 1700) > 50 for l in ts["labels"])                        # no clutter labels
    assert ts["agreed_share"] > 0.5
    assert any(d["frame"] in (450, 450 + round(300 * 29.97)) for d in ts["disagreements"])
    s = TS.review_sample(ts, n_check=5, n_dis=2); assert 1 <= len(s["check"]) <= 5 and len(s["disagree"]) >= 1
