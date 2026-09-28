import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import playerbench as PB
class D:  # stand-in for supervision Detections
    def __init__(s, xyxy, conf, cls): s.xyxy = np.array(xyxy, float); s.confidence = np.array(conf, float); s.class_id = np.array(cls)
    def __len__(s): return len(s.xyxy)
f = np.zeros((1080, 1920, 3), np.uint8)
det = D([[0, 0, 10, 30], [20, 0, 30, 30], [40, 0, 50, 30], [60, 0, 70, 30], [80, 0, 90, 30]], [0.8, 0.2, 0.12, 0.9, 0.5], [0, 0, 0, 3, 0])
b = PB.classify(det, {0: "player", 3: "referee"}, f, lambda fr, box: box[0] == 80)
assert [x[5] for x in b] == ["today", "at 0.15", "at 0.10", "model-referee", "referee"], b
fr = PB.pick_frames(9000, 10); assert len(fr) == 10 and fr[0] > 700 and fr[-1] < 8300
im = PB.draw(f, b, "test"); assert im.shape == (720, 1280, 3) and len(PB.jpg(im)) > 1000
print("OK")
