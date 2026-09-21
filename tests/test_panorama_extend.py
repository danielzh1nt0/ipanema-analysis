"""Panorama extension on real texture: views cut from the verified panorama must be placed where they truly belong,
and the map must grow beyond the seed (including via later passes)."""
import os, cv2, numpy as np
from ipanema.panorama import extend
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_extend_real_texture():
    world = cv2.imread(f"{ROOT}/results/mosaic_SFKBP1109_c4.jpg"); Hw, Ww = world.shape[:2]
    x0, x1 = 1400, 2800; seed = world[:, x0:x1].copy(); rng = np.random.RandomState(0); views, truth = {}, {}
    while len(views) < 25:
        cx, cy = rng.uniform(700, Ww - 700), rng.uniform(560, Hw - 560)
        quad = np.float32([[cx - 625, cy - 350], [cx + 625, cy - 350], [cx + 625, cy + 350], [cx - 625, cy + 350]]) + rng.uniform(-50, 50, (4, 2)).astype(np.float32)
        if quad[:, 0].min() < 0 or quad[:, 0].max() >= Ww or quad[:, 1].min() < 0 or quad[:, 1].max() >= Hw: continue
        Hv = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]])); f = cv2.warpPerspective(world, Hv, (1920, 1080))
        if (f.max(2) < 8).mean() > 0.03: continue
        views[len(views)] = f; truth[len(truth)] = np.linalg.inv(Hv)
    canvas, covered, T, placed = extend(seed, list(views), lambda k: views[k], pad=(1600, 100, 1600, 100), log=lambda *a: None)
    W2C = T @ np.array([[1, 0, -x0], [0, 1, 0], [0, 0, 1.0]]); c = np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]]).reshape(-1, 1, 2)
    errs = [float(np.abs(cv2.perspectiveTransform(c, H) - cv2.perspectiveTransform(c, W2C @ truth[k])).max()) for k, H, g in placed]
    assert len(placed) >= 5 and max(errs) < 8.0, (len(placed), max(errs))
    wc = cv2.warpPerspective((world.max(2) > 8).astype(np.uint8), W2C, (canvas.shape[1], canvas.shape[0])) > 0
    assert (covered & wc).sum() / wc.sum() > 0.6
