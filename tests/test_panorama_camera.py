"""Curved panorama camera: exact conversions, automatic refit on a differently-sized recording, and the analysis end to end."""
import os, json, types, tempfile, numpy as np, cv2
from ipanema.cylcam import CylCam, fit, NAMES
from ipanema.calibration import to_m, pitch_segments
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(f"{ROOT}/calibration/panorama/SFKBP1109_pano.json"))
P0 = [SPEC["params"][k] for k in NAMES]

def test_conversions_and_to_m():
    cam = CylCam(P0); P = np.column_stack([np.random.RandomState(1).uniform(0, 106, 500), np.random.RandomState(2).uniform(0, 64, 500)])
    assert np.abs(to_m(cam, cam.project(P)) - P).max() < 1e-3

def test_refit_on_resized_recording():
    """a recording of a different size with a border (browser edges): the automatic fit must still put the lines on the paint"""
    shot = cv2.imread(f"{ROOT}/results/panorama/SFKBP1109/veo_panorama_screenshot_1411.png")
    small = cv2.resize(shot, (1800, int(1800 * shot.shape[0] / shot.shape[1])))
    frame = np.zeros((1080, 1920, 3), np.uint8); oy, ox = 60, 60; frame[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    s = 1800 / shot.shape[1]
    _, stats = fit(frame, P0, 106.0, 64.0, mask_top=oy + int(100 * s), mask_bottom=oy + int(1225 * s), log=lambda *a: None)
    assert stats["median_px"] <= 2.0 and stats["within6_pct"] >= 85, stats

def test_full_analysis_through_curved_camera():
    import sys
    from ipanema import fullmatch as FM
    from ipanema.config import Settings
    from ipanema.run import analyse
    cam = CylCam(P0); fps = 10.0; n = 1200; L, W = 106.0, 64.0; rng = np.random.RandomState(0)
    per, H, cands = {}, {}, {}
    for k in range(n):
        rows = []
        for j in range(20):
            team = "A" if j < 10 else "B"
            m = np.array([10 + 4 * j + 3 * np.sin(k / 30 + j), 8 + 2.3 * j + 2 * np.cos(k / 40 + j)]); feet = cam.project(m)[0]
            rows.append([j + 1, team, m, feet, np.array([feet[0] - 5, feet[1] - 30, feet[0] + 5, feet[1]]), False])
        per[k] = rows; H[k] = cam
        c = rows[k // 200 % 20]; cands[k] = [(float(c[3][0]) + 2, float(c[3][1]) - 3, 0.9)]
    per, H, cands, play, periods = FM.apply_periods(per, H, cands, [(0, 50), (60, 120)], fps, L, W)
    d = tempfile.mkdtemp(); vp = f"{d}/v.mp4"; vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    for _ in range(n): vw.write(np.full((180, 320, 3), 60, np.uint8))
    vw.release()
    ctx = {"match_id": "SYN_pano", "video": vp, "vi": {"n": n, "fps": fps, "width": 2450, "height": 1364}, "H": H, "L": L, "W": W,
           "cal": {"coverage": 1.0, "frozen": 0}, "tm": types.SimpleNamespace(dark_share={"A": 0.5, "B": 0.2}, strips=None), "per": per, "fps": fps,
           "cands": cands, "t0": 0, "play_mask": play, "periods": periods}
    summary, folder, _ = analyse(ctx, Settings(root=d, sports_dir="/tmp/x", work=f"{d}/w"), log=lambda *a: None,
                                 export_kw={"frame_stride": 2, "split_s": 60, "copy_video": False, "make_zip": False}, gt_path="/nonexistent")
    md = json.load(open(f"{folder}/match_data.json")); f0 = json.load(open(f"{folder}/frames_000.json"))["frames"]
    assert md["camera"]["camera"] == "cylindrical" and md["pitch"] == {"length": 106.0, "width": 64.0}
    assert all(fr["pitch_lines"] is None for fr in f0)
    pl = [p for fr in f0 for p in fr["players"]]; assert pl and all(0 <= p["m"][0] <= 106 and 0 <= p["m"][1] <= 64 for p in pl if p.get("m"))
    assert summary["players_per_frame_median"]["A"] >= 9 and summary["match_seconds"] == 110.0

def test_find_video_rect_on_real_recording():
    """today's screen recording: Veo's player sits inside the browser; the crop must find it (checked by eye: ~154-1701 x 185-1056)"""
    from ipanema.cylcam import find_video_rect
    fs = [cv2.imread(f"{ROOT}/results/check/p15u-vs-bp-2026-09-22-2000_test/test_piece2_f{k:05d}.jpg") for k in (1800, 4500, 7200)]
    x0, y0, x1, y1 = find_video_rect(fs)
    assert abs(x0 - 154) <= 8 and abs(x1 - 1701) <= 8 and abs(y0 - 185) <= 8 and abs(y1 - 1056) <= 8, (x0, y0, x1, y1)

def test_impossible_camera_rejected():
    from ipanema.cylcam import plausible
    assert plausible(P0, 106.0, 64.0)[0]
    assert not plausible([52.919, 64.494, 0.685, -0.123, 464.1, 1034.1, 1607.3, 532.1, 0.111, 0.016], 106.0, 64.0)[0]   # 21 Sep wrong fit
