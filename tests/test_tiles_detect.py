"""Tiled player detection: every player found exactly once at its true full-frame position (overlaps and frame edge included)."""
def test_tiled_detection(monkeypatch):
    import sys, numpy as np; 
    import supervision as sv
    import ipanema.tracking as T
    H, W = 1364, 2450; truth = np.array([[300, 420], [1250, 500], [2200, 450], [700, 900], [1800, 1100], [1225, 800], [2440, 1300]], float)
    monkeypatch.setattr(sv.Detections, "from_ultralytics", staticmethod(lambda r: sv.Detections(xyxy=r.boxes, confidence=np.full(len(r.boxes), 0.9), class_id=np.zeros(len(r.boxes), int)) if len(r.boxes) else sv.Detections.empty()))
    class M:
        def __init__(self): self.i = 0
        def __call__(self, crops, conf=0.3, verbose=False):
            assert isinstance(crops, list) and len(crops) == len(T.PANO_TILES), "all tiles must go to the model in one batch"
            return [self.one(c) for c in crops]
        def one(self, crop):
            x0, y0, x1, y1 = T.PANO_TILES[self.i]; self.i += 1; ox, oy = int(x0 * W), int(y0 * H); h, w = crop.shape[:2]
            bx = [[fx - 10 - ox, fy - 40 - oy, fx + 10 - ox, fy - oy] for fx, fy in truth if ox <= fx - 10 and fx + 10 <= ox + w and oy <= fy - 40 and fy <= oy + h]
            class Res: pass
            Res.names = {0: "player", 1: "referee"}; Res.boxes = np.array(bx, float).reshape(-1, 4); return Res
    det, names = T.detect_tiled(M(), np.zeros((H, W, 3), np.uint8), 0.3, T.PANO_TILES)
    feet = np.c_[(det.xyxy[:, 0] + det.xyxy[:, 2]) / 2, det.xyxy[:, 3]]
    found = [bool(np.min(np.linalg.norm(feet - t, axis=1)) < 1e-6) for t in truth]
    assert len(det) == len(truth) and all(found), found