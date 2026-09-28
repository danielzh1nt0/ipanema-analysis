"""28 Sep: kits learned per match - two teams + a referee in ANY colours, also under a colour cast (floodlights)"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
rng = np.random.default_rng(0)
def person(bgr, cast=(0, 0, 0)):
    im = np.zeros((120, 30, 3), np.uint8); im[:] = (40, 150, 40)                  # grass (lower third = pitch sample)
    shirt = np.clip(np.array(bgr) + np.array(cast) + rng.normal(0, 8, 3), 0, 255); im[14:40, 6:24] = shirt; return im
for name, A, B, REF, cast in (("orange vs white, blue ref", (40, 120, 240), (235, 235, 235), (220, 160, 60), (0, 0, 0)),
                              ("black vs white, orange ref, floodlight cast", (30, 30, 30), (230, 230, 230), (20, 140, 250), (-40, 10, 40)),
                              ("red vs dark-green kit, black ref", (40, 40, 200), (40, 95, 20), (25, 25, 25), (0, 0, 0))):
    ppl = [("A", person(A, cast)) for _ in range(60)] + [("B", person(B, cast)) for _ in range(55)] + [("other", person(REF, cast)) for _ in range(6)]
    feats = [K.torso_feature(im, (0, 0, 30, 80)) for _, im in ppl]; m = K.fit(feats)
    got = [K.classify(m, f) for f in feats]
    # team names are arbitrary: map by majority
    a_lab = max("AB", key=lambda t: sum(g == t for (w, _), g in zip(ppl, got) if w == "A")); b_lab = "B" if a_lab == "A" else "A"
    ok = sum((w == "A" and g == a_lab) or (w == "B" and g == b_lab) or (w == "other" and g == "other") for (w, _), g in zip(ppl, got))
    print(f"{name}: {ok}/{len(ppl)}"); assert ok >= len(ppl) - 2, name
print("OK")
# KitTeamModel: A = darker team, B = lighter, K = neither
ppl = [person((30, 30, 30)) for _ in range(40)] + [person((230, 230, 230)) for _ in range(40)] + [person((220, 160, 60)) for _ in range(4)]
tm = K.KitTeamModel().fit_frames([(im, [(0, 0, 30, 120)]) for im in ppl], log=lambda *a: None)
labs = [tm.predict_batch(im, [(0, 0, 30, 120)])[0] for im in ppl]
assert labs[:40].count("A") >= 39 and labs[40:80].count("B") >= 39 and labs[80:].count("K") >= 3, (labs[:3], labs[40:43], labs[80:])
assert set(tm.strips) == {"A", "B"} and tm.dark_share["A"] < tm.dark_share["B"]
print("KitTeamModel OK")
