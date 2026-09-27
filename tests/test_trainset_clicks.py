import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import numpy as np, random
from ipanema import trainset as TS
rng = random.Random(1)
# a ball rolling for 90 frames seen every 3rd frame + a static cone + random noise
clicks = {}
for k in range(0, 9000, 3):
    d = []
    if 300 <= k < 390: d.append((500 + (k - 300) * 4.0, 600 + (k - 300) * 1.0, 0.4 + 0.5 * rng.random()))
    d.append((1200, 800, 0.9))                                     # static cone
    if rng.random() < 0.1: d.append((rng.uniform(0, 1920), rng.uniform(0, 1080), rng.uniform(0.3, 0.9)))
    clicks[k] = d
wasb = {k: [(500 + (k - 300) * 4.0, 600 + (k - 300) * 1.0, 0.5)] for k in range(300, 390)}
ts = TS.match_click_trainset([(0, 1000, clicks, wasb)])
labs = ts["labels"]
print("labels", len(labs), "tracks", ts["tracks"], ts["by_conf"], "wasb agrees", ts["wasb_agrees_share"])
assert all(1300 <= l["frame"] < 1390 for l in labs), "only the rolling ball should give labels"
assert len(labs) >= 25
r = TS.review_by_conf({"labels": labs * 1}, per_bin=10, min_gap_frames=0)
print("review", len(r), sorted({x["band"] for x in r}))
print("OK")
