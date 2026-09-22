"""Tracking stops itself when it is too slow to finish (it burned two paid runs on 22 Sep before this existed)."""
import os, numpy as np, cv2, tempfile, types, sys, pytest

def test_watchdog_raises(monkeypatch):
    import ipanema.tracking as T
    monkeypatch.setenv("IPANEMA_MIN_FPS", "1000000")                       # nothing can be this fast: the watchdog must fire
    d = tempfile.mkdtemp(); p = f"{d}/v.mp4"
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for _ in range(1600): vw.write(np.zeros((180, 320, 3), np.uint8))
    vw.release()
    class Det:
        xyxy = np.zeros((0, 4)); confidence = np.zeros(0); class_id = np.zeros(0, int); tracker_id = np.zeros(0, int)
        def __len__(self): return 0
        def with_nms(self, *a, **k): return self
    class Res:
        names = {0: "player"}
    monkeypatch.setattr(T, "detect_tiled", lambda *a, **k: (Det(), {0: "player"}))
    import supervision as sv
    monkeypatch.setattr(sv.Detections, "from_ultralytics", staticmethod(lambda r: Det()))
    monkeypatch.setattr(sv, "ByteTrack", lambda *a, **k: types.SimpleNamespace(update_with_detections=lambda d: d))
    monkeypatch.setattr(T, "YOLO", lambda w: (lambda *a, **k: [Res()]), raising=False)
    sys.modules.setdefault("ultralytics", types.ModuleType("ultralytics")).YOLO = lambda w: (lambda *a, **k: [Res()])
    with pytest.raises(RuntimeError, match="too slow"):
        T.track(p, "weights.pt", {k: np.eye(3) for k in range(1600)}, None, 0.3, log=lambda *a: None, imgsz=1920)
