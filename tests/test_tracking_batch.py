import numpy as np, cv2, os
from ipanema import tracking as TR

class _Model:
    """stands in for YOLO: returns an ultralytics Results with no boxes per image"""
    def __call__(self, imgs, **kw):
        from ultralytics.engine.results import Results
        return [Results(orig_img=im, path="", names={0: "player"}, boxes=__import__("torch").zeros((0, 6))) for im in imgs]

def test_batched_frames_exact_multiple_of_batch(tmp_path):
    p = str(tmp_path / "t.mp4"); w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    for i in range(16): w.write(np.full((180, 320, 3), i * 10, np.uint8))
    w.release()
    try: import supervision, ultralytics, torch  # noqa
    except ImportError: return
    ks = [k for k, f, d, n in TR._batched_frames(p, _Model(), 0.1, TR.FOLLOW_TILES, 320, batch=8)]
    assert ks == list(range(16))
