import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import statreview as SR
fps = 30.0; frames = [{"t": k / fps, "possession": ("A" if (k // 300) % 2 else "B") if k % 7 else None} for k in range(9000)]
md = {"fps": fps, "frames": frames, "restarts": [{"t": 1.0, "kind": "throw-in", "team": None}, {"t": 150.0, "kind": "corner", "team": "A"}]}
stops, poss, dur = SR.plan(md, 40)
assert len(stops) == 2 and len(poss) == 40 and all(b["frame"] - a["frame"] >= 4 * fps for a, b in zip(poss, poss[1:]))
items = SR.pictures(lambda k: np.full((1080, 1920, 3), k % 255, np.uint8), md, stops, poss, fps)
assert len(items) == 42 and all(len(i["img"]) > 1000 for i in items) and items[0]["kind"] == "stoppage"
print("OK", len(items))
