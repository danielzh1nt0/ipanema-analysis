"""'Don't guess': frames whose calibration is far from the painted lines are untrusted; correct ones are kept."""
import numpy as np, cv2, tempfile, os
from ipanema.calcheck import confidence_mask, draw_model
from ipanema.fullmatch import apply_unknown

def test_confidence_mask_and_apply():
    L, W = 120.0, 70.0; H = np.array([[14.0, 3.0, -100.0], [0.0, 6.0, 380.0], [0.0, 0.004, 1.0]])
    img = np.zeros((1080, 1920, 3), np.uint8); img[:] = (40, 130, 60); img = draw_model(img, H, L, W, (235, 235, 235), 3)
    p = os.path.join(tempfile.mkdtemp(), "v.mp4"); vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for _ in range(200): vw.write(img)
    vw.release()
    wrong = np.array([[1, 0, 90.0], [0, 1, 40.0], [0, 0, 1.0]]) @ H                     # a badly misplaced stretch: frames 80-119
    Hs = {k: (wrong if 80 <= k < 120 else H) for k in range(200)}; del Hs[199]           # and one frame with no calibration
    ok, summary = confidence_mask(p, Hs, 200, L, W, every=10)
    assert ok[:70].all() and ok[130:199].all() and not ok[80:120].any() and not ok[199]
    assert summary["bad"] == 4 and summary["good"] >= 15
    per = {k: [[1]] for k in range(200)}; cands = {k: [(1.0, 1.0, 0.9)] for k in range(200)}
    per, cands = apply_unknown(per, cands, ok)
    assert per[100] == [] and cands[100] == [] and per[10] == [[1]]
