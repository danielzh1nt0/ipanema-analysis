"""Pitch-point click sessions: frames spread over the playing time, no overlap between sessions, half-time skipped."""
import os, tempfile, zipfile, numpy as np, cv2
from ipanema.label import sample_session

def test_sessions_cover_play_without_overlap():
    d = tempfile.mkdtemp(); v = f"{d}/m.mp4"; vw = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (320, 180))
    for _ in range(2 * 100 * 60 // 10): vw.write(np.zeros((180, 320, 3), np.uint8))     # 20 min at 2 fps, scaled play windows below
    vw.release()
    play = ((0.0, 5 * 60.0), (7 * 60.0, 10 * 60.0))                                          # "half-time" 5:00-7:00 must be skipped
    seen = []
    for s in (1, 2, 3):
        z = zipfile.ZipFile(sample_session(v, f"{d}/s{s}.zip", s, n=10, width=160, play=play, min_minutes=5))
        ts = [float(n[3:-4]) for n in z.namelist()]; seen += ts
        assert len(ts) == 10 and not any(300 <= t < 420 for t in ts), ts
        assert cv2.imdecode(np.frombuffer(z.read(z.namelist()[0]), np.uint8), 1).shape[1] == 160
    assert len(set(seen)) == 30 and min(seen) < 30 and max(seen) > 570                       # distinct, spread over both halves
