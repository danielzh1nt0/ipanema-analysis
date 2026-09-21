"""Panorama registration must not crash when a piece starts on a view the panorama doesn't contain (the 21 Sep bug)."""
import os, cv2, numpy as np, tempfile
from ipanema.mosaic import calibrate_via_mosaic
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _clip(path, bad_start):
    pano = cv2.imread(f"{ROOT}/results/mosaic_SFKBP1109_c4.jpg"); H0 = pano.shape[0]
    other = cv2.imread(f"{ROOT}/results/prepare/p15u-vs-aik-2026-09-21-bd09/frame_004496.jpg")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for k in range(300):
        if bad_start and k < 60: f = cv2.resize(other, (1920, 1080))
        else:
            x = int(200 + k * 5); y = max(0, (H0 - 1080) // 2); f = pano[y:y + 1080, x:x + 1920]
            if f.shape[:2] != (1080, 1920): f = cv2.resize(f, (1920, 1080))
        vw.write(f)
    vw.release()

def test_unmatched_start_registers_all_frames():
    p = os.path.join(tempfile.mkdtemp(), "bad.mp4"); _clip(p, True)
    Hs = calibrate_via_mosaic(p, "SFKBP1109_c003", tempfile.mkdtemp(), code_dir=ROOT, log=lambda *a: None)
    assert Hs is not None and len(Hs) == 300

def test_normal_pan_registers_all_frames():
    p = os.path.join(tempfile.mkdtemp(), "ok.mp4"); _clip(p, False)
    Hs = calibrate_via_mosaic(p, "SFKBP1109_s1200", tempfile.mkdtemp(), code_dir=ROOT, log=lambda *a: None)
    assert Hs is not None and len(Hs) == 300
