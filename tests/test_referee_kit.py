"""28 Sep: referee vs keeper vs players on real SFK-BP crops (tests/data/kits, cut from the saved clip frames)"""
import os, sys, glob, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import teams as T
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kits"); res = {}
HELD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kits_heldout")
for f in sorted(glob.glob(f"{D}/*.png")):
    im = cv2.imread(f); h, w = im.shape[:2]; res[os.path.basename(f)] = T.referee_kit(im, (0, 0, w, h))
kind = lambda n: n.split("_")[0]
refs = [v for n, v in res.items() if kind(n) == "ref"]; others = {n: v for n, v in res.items() if kind(n) != "ref"}
print(res)
assert sum(refs) >= 4, f"referees caught {sum(refs)}/5"
assert not any(others.values()), f"a keeper or player was called referee: {[n for n, v in others.items() if v]}"
print("OK", f"referees {sum(refs)}/{len(refs)}, keepers and players never")
# held-out crops from 6 other frames, never used to set the thresholds: 22/22 on 28 Sep
hr = {os.path.basename(f): T.referee_kit(cv2.imread(f), (0, 0, cv2.imread(f).shape[1], cv2.imread(f).shape[0])) for f in sorted(glob.glob(f"{HELD}/*.png"))}
bad = [n for n, v in hr.items() if v != (n.startswith("ref"))]
assert len(hr) == 23 and not bad, bad
print("held-out OK 23/23")
# 28 Sep all-footage check: the kit rule is SFK-BP-specific (caught players on 6 other matches) -> must stay OFF by default
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ipanema", "teams.py")).read()
assert "IPANEMA_REFEREE_KIT\", \"0\") == \"1\" and (_is_referee_bib(c) or referee_kit" in src
print("kit rule off by default: OK")
