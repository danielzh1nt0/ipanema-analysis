"""P1 (29 Sep): striped kits. A striped shirt's MEDIAN colour jumps between its stripes, so many striped players read as
'neither team' (Spånga: 79/142). The kit fit now also tries a mean colour and the far-pair rule and keeps the reading that
leaves the fewest people as 'neither' (only if it wins by 3 points) - but only when the default fit leaves out a big
colour group (model["missed"] >= 1), so plain-kit matches keep their old reading."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K

def person(rng, kind):
    im = np.zeros((120, 30, 3), np.uint8); im[:] = (40, 150, 40)
    if kind == "dark": im[14:40, 6:24] = np.clip(np.array((30, 30, 35)) + rng.normal(0, 8, 3), 0, 255)
    elif kind == "ref": im[14:40, 6:24] = np.clip(np.array((20, 190, 235)) + rng.normal(0, 8, 3), 0, 255)
    else:                                                        # white/navy vertical stripes, share of white varies with pose
        share = rng.uniform(0.3, 0.7); w = 6; x = np.arange(6, 24); white = ((x - 6 + rng.integers(0, w)) % w) < share * w
        for i, xx in enumerate(x): im[14:40, xx] = np.clip(np.array((200, 225, 235) if white[i] else (90, 35, 25)) + rng.normal(0, 6, 3), 0, 255)
    return im

def fit_and_read(auto):
    rng = np.random.default_rng(1); os.environ["IPANEMA_KIT_AUTO"] = "1" if auto else "0"
    try:
        kinds = ["dark"] * 60 + ["striped"] * 60 + ["ref"] * 5; ims = [person(rng, k) for k in kinds]
        tm = K.KitTeamModel().fit_frames([(im, [(0, 0, 30, 120)]) for im in ims], log=lambda *a: None, player_cls=False)
        labs = [tm.predict_batch(im, [(0, 0, 30, 120)])[0] for im in ims]
        return tm, labs[:60], labs[60:120], labs[120:]
    finally: os.environ.pop("IPANEMA_KIT_AUTO", None)

def test_mean_reading_is_steadier_on_stripes():
    rng = np.random.default_rng(2); ims = [person(rng, "striped") for _ in range(40)]
    sd = {s: np.std([K.torso_feature(im, (0, 0, 30, 120), stat=s)[0] for im in ims]) for s in ("median", "mean")}
    assert sd["mean"] < 0.6 * sd["median"], sd

def test_spanga_striped_players_read_as_a_team():
    """real saved Spånga frames + the by-eye key (results/qa/kitprobe): 181/279 before P1"""
    import json, cv2
    R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/qa/kitprobe"); g = "p15u-vs-spanga-2026-09-25"
    B = json.load(open(f"{R}/{g}/boxes.json")); fr = {fn: cv2.imread(f"{R}/{g}/{fn}") for fn in sorted(B)}
    L = json.load(open(f"{R}/spanga_labels_p8.json"))
    tm = K.KitTeamModel().fit_frames([(fr[fn], np.array(B[fn])) for fn in sorted(B)], log=lambda *a: None)
    pred = {fn: tm.predict_batch(fr[fn], np.array(B[fn])) for fn in sorted(B)}
    pairs = [(l, pred[fn][j]) for (fn, j), l in zip(L["items"], L["labels"]) if l in "DS"]
    d = max("AB", key=lambda t: sum(l == "D" and p == t for l, p in pairs)); s_ = "B" if d == "A" else "A"
    right = sum((l == "D" and p == d) or (l == "S" and p == s_) for l, p in pairs)
    assert tm.choice_missed >= 1.0 and tm.pair == "far" and right >= 225, (right, tm.choice, tm.choice_missed)

def test_plain_kits_keep_the_default_fit():
    rng = np.random.default_rng(3); ims = [person(rng, "dark") for _ in range(60)] + [person(rng, "ref") for _ in range(5)]
    def white(r):
        im = np.zeros((120, 30, 3), np.uint8); im[:] = (40, 150, 40); im[14:40, 6:24] = np.clip(np.array((230, 230, 230)) + r.normal(0, 8, 3), 0, 255); return im
    ims += [white(rng) for _ in range(55)]
    tm = K.KitTeamModel().fit_frames([(im, [(0, 0, 30, 120)]) for im in ims], log=lambda *a: None, player_cls=False)
    assert len(tm.choice) == 1 and tm.stat is None and tm.choice_missed < 1.0, (tm.choice, tm.choice_missed)

def test_auto_off_is_the_old_reading():
    tm, dark, striped, ref = fit_and_read(False)
    assert tm.stat is None and len(tm.choice) == 1

def test_reymersholm_and_sfk_keep_their_fit():
    import json, cv2
    R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/qa/kitprobe")
    for g in ("p15u-vs-reymersholm-2026-09-18", "SFKBP1109_s1200"):
        B = json.load(open(f"{R}/{g}/boxes.json"))
        tm = K.KitTeamModel().fit_frames([(cv2.imread(f"{R}/{g}/{fn}"), np.array(B[fn])) for fn in sorted(B)], log=lambda *a: None, player_cls=False)
        if "reymersholm" in g:
            # P2b (4 Oct): the pitch-edge fallback keeps 305 people instead of 93; the switch now fires and reads green vs white (by eye)
            ta, tb = sorted(tm.model["teams"], key=lambda c: c[0]); assert ta[1] < -10 and tb[0] > 75, (g, tm.model["teams"])
        else: assert len(tm.choice) == 1 and tm.choice_missed < 1.0, (g, tm.choice, tm.choice_missed)
