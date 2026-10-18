"""18 Oct (worker, N0): tests for ipanema/link.py (follow-cam frame -> pitch metres through the same-moment panorama
frame; used by modal_app's follow-cam/panorama proof, was untested). Made-up textured 'pitch' picture, a known
panorama camera (pixels -> metres) and a known follow-cam view (zoomed, shifted crop of the panorama)."""
import cv2, numpy as np
import pytest

from ipanema import link as LK

L, W = 105.0, 68.0
PW, PH = 1400, 500                                    # panorama size


class Cam:
    """panorama camera stand-in: a fixed homography from panorama pixels to pitch metres"""
    def __init__(self, H): self.H = H
    def to_m(self, px): return cv2.perspectiveTransform(np.float32(px).reshape(-1, 1, 2), self.H).reshape(-1, 2)


def _pano(seed=0):
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur(rng.integers(0, 255, (PH, PW), dtype=np.uint8), (0, 0), 2)    # grass-like texture with corners
    img = cv2.merge([g // 3, g, g // 3])
    for x in range(0, PW, 140): cv2.line(img, (x, 0), (x, PH), (255, 255, 255), 2)
    for y in range(0, PH, 100): cv2.line(img, (0, y), (PW, y), (255, 255, 255), 2)
    for _ in range(300):
        c = tuple(int(v) for v in rng.integers(0, [PW, PH])); cv2.circle(img, c, int(rng.integers(3, 12)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def _cam():
    src = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]]); dst = np.float32([[0, 0], [L, 0], [L - 5, W], [5, W]])    # slight perspective
    return Cam(cv2.getPerspectiveTransform(src, dst))


def _fc(pano, x0=500, y0=150, zoom=2.0, size=(640, 360)):
    """follow-cam frame: a zoomed crop of the panorama; A maps follow-cam pixels -> panorama pixels"""
    A = np.float64([[1 / zoom, 0, x0], [0, 1 / zoom, y0], [0, 0, 1]])
    return cv2.warpPerspective(pano, np.linalg.inv(A), size), A


def test_fc_to_metres_recovers_the_known_view():
    pano = _pano(); cam = _cam(); fc, A = _fc(pano)
    H, info = LK.fc_to_metres(fc, pano, cam, L, W)
    assert H is not None, info
    assert info["inliers"] >= 12 and info["residual_m"] < 0.2
    truth = cam.H @ A                                   # follow-cam px -> pano px -> metres
    pts = np.float32([[50, 50], [600, 40], [320, 180], [30, 330], [620, 340]])
    got = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    want = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), truth).reshape(-1, 2)
    assert np.abs(got - want).max() < 0.5             # metres, whole frame incl. corners


def test_unrelated_frame_gives_no_calibration_with_a_reason():
    pano = _pano(0); other = _pano(7)[100:460, 300:940]
    H, info = LK.fc_to_metres(other, pano, _cam(), L, W)
    assert H is None and info["why"]


def test_blank_frame_gives_no_matches():
    pano = _pano(); blank = np.full((360, 640, 3), (40, 120, 40), np.uint8)
    a, b = LK.match(blank, pano)
    assert a.shape == (0, 2) and b.shape == (0, 2)
    H, info = LK.fc_to_metres(blank, pano, _cam(), L, W)
    assert H is None and info["why"] == "too few matches"


def test_match_shapes_stay_two_column_when_nothing_passes_the_ratio_test():
    pano = _pano()
    a, b = LK.match(pano, pano, ratio=0.0)            # ratio 0: no match can pass
    assert a.shape == (0, 2) and b.shape == (0, 2)


def test_matches_off_the_pitch_are_refused():
    pano = _pano(); fc, _ = _fc(pano)
    far = Cam(np.float64([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]]))     # every panorama pixel lands 1 km off the pitch
    H, info = LK.fc_to_metres(fc, pano, far, L, W)
    assert H is None and info["why"] == "too few matches on the pitch"


def test_best_frame_picks_the_frame_with_the_most_ground_matches():
    pano = _pano(); fc, A = _fc(pano)
    covered = pano.copy(); covered[150:330, 500:820] = _pano(3)[150:330, 500:820]        # the follow-cam's area replaced
    half = pano.copy(); half[150:330, 500:660] = _pano(4)[150:330, 500:660]               # half of it replaced
    n, i, H = LK.best_frame(fc, [covered, half, pano], _cam(), L, W)
    assert i == 2 and n >= 12 and H is not None
    _, info = LK.fc_to_metres(fc, half, _cam(), L, W)
    assert info.get("inliers", 0) < n


def test_best_frame_none_found():
    pano = _pano(); blank = np.full((360, 640, 3), 90, np.uint8)
    assert LK.best_frame(blank, [pano, pano], _cam(), L, W) == (-1, None, None)
