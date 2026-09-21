"""The calibration score must be ~0 for a correct calibration and grow with error; the piece report must run end to end."""
import numpy as np, cv2, tempfile, os
from ipanema.calcheck import score_frame, report_piece
from ipanema.calibration import pitch_segments

def _frame(H, L=120.0, W=70.0):
    img = np.zeros((1080, 1920, 3), np.uint8); img[:] = (40, 130, 60)
    for a, b in pitch_segments(L, W):
        p = cv2.perspectiveTransform(np.float32([[a], [b]]), np.float32(H)).reshape(-1, 2)
        if np.isfinite(p).all() and np.abs(p).max() < 1e5: cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (235, 235, 235), 3)
    return img

H = np.array([[14.0, 3.0, -100.0], [0.0, 6.0, 380.0], [0.0, 0.004, 1.0]])

def test_score_grows_with_error():
    img = _frame(H); shift = lambda dx: np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1.0]]) @ H
    s0, s30 = score_frame(img, H, 120, 70), score_frame(img, shift(30), 120, 70)
    assert s0["p80_px"] < 1.0 and s0["within8_pct"] > 99
    assert s30["p80_px"] > 15 and s30["within8_pct"] < 70

def test_report_piece():
    d = tempfile.mkdtemp(); path = os.path.join(d, "p.mp4"); img = _frame(H)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for _ in range(300): vw.write(img)
    vw.release()
    per = {k: [[1, "A", np.array([60.0, 35.0]), np.zeros(2), np.zeros(4), False], [2, "B", np.array([130.0, 35.0]), np.zeros(2), np.zeros(4), False]] for k in range(300)}
    out, jpg = report_piece(path, {k: H for k in range(300)}, per, 120.0, 70.0, every=50)
    assert out["frames_scored"] == 6 and out["off_pitch_pct"] == 50.0 and out["players_per_frame"] == 2.0
    assert out["p80_px_median"] < 3.0 and jpg is not None and len(jpg) > 1000
