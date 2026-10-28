"""N0-28d: OpenCV 4.11.0 ignores OPENCV_FFMPEG_CAPTURE_OPTIONS (upside-down check), so the R2 reconnect settings never
reached ffmpeg on the runner and a stalled big clip sat silent for 2 min. Workflows pin 4.10.0.84; clipopen warns on 4.11."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pytest
from tools import clipopen


def _cv(version):
    class CV:
        __version__ = version
        CAP_FFMPEG, CAP_PROP_OPEN_TIMEOUT_MSEC, CAP_PROP_READ_TIMEOUT_MSEC = 1900, 53, 54
        def VideoCapture(self, *a): return "cap"
    return CV()


def test_only_411_is_flagged():
    assert not clipopen.honours_ffmpeg_options(_cv("4.11.0"))
    for v in ("4.10.0", "4.12.0", "4.13.0", "5.0.0"):
        assert clipopen.honours_ffmpeg_options(_cv(v)), v


def test_capture_warns_once_on_411_for_urls(monkeypatch):
    monkeypatch.setattr(clipopen, "_warned", [False])
    logs = []
    clipopen.capture(_cv("4.11.0"), "/tmp/v.mp4", log=logs.append); assert logs == []          # local file: no warning
    clipopen.capture(_cv("4.10.0"), "https://r2/v.mp4", log=logs.append); assert logs == []
    clipopen.capture(_cv("4.11.0"), "https://r2/v.mp4", log=logs.append)
    clipopen.capture(_cv("4.11.0"), "https://r2/v.mp4", log=logs.append)
    assert len(logs) == 1 and "4.10.0.84" in logs[0]


@pytest.mark.parametrize("wf", ["nightly.yml", "free.yml"])
def test_workflows_pin_opencv_410(wf):
    txt = open(os.path.join(ROOT, ".github/workflows", wf)).read()
    installs = re.findall(r"opencv-python-headless[^\s\"]*", txt)
    assert installs and all(i == "opencv-python-headless==4.10.0.84" for i in installs), installs


def test_tracktest_logs_opencv_version():
    assert 'log(f"opencv {cv2.__version__}")' in open(os.path.join(ROOT, "tools/tracktest.py")).read()


def test_stall_server_reconnects_with_a_good_opencv(tmp_path):
    """The runner check's own stall test, with this machine's OpenCV: header read goes silent -> reconnect, all frames."""
    cv2 = pytest.importorskip("cv2")
    if not clipopen.honours_ffmpeg_options(cv2): pytest.skip("this OpenCV ignores the options")
    from tools import n0_28d_runner_check as RC
    clip = RC.make_clip(str(tmp_path / "c.mp4"), n=200, size=(320, 180))
    url, reqs, srv = RC.stall_server(clip, stalls={2}, stall_after={2: 0})
    res = RC.probe(url, 50, 100, timeout=90); srv.shutdown()
    assert res.get("frames") == 100, res
    assert any("STALL" in (q or "") for q in reqs) and len(reqs) >= 3      # it stalled and asked again


def test_piece_writer_without_file_counts_zero_not_crash(tmp_path):
    """Dry run 28 Oct: a stalled read gave frames but the writer left no file and os.replace crashed the job."""
    import numpy as np
    class Cap:
        left = 5
        def get(self, p): return 25.0
        def set(self, p, v): pass
        def read(self):
            if self.left <= 0: return False, None
            self.left -= 1; return True, np.zeros((4, 6, 3), np.uint8)
        def release(self): pass
    class CV:
        CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES = 5, 7, 1
        def VideoCapture(self, src): return Cap()
        def VideoWriter_fourcc(self, *a): return 0
        def VideoWriter(self, *a):
            class W:
                def write(self, f): pass
                def release(self): pass
            return W()
    logs = []
    got, fps, tries = clipopen.read_piece("x.mp4", CV(), 0, 5, str(tmp_path / "p.mp4"), tries=2, sleep=lambda s: None, log=logs.append, cap=Cap())
    assert got == 0 and tries == 2 and any("no file" in m for m in logs)
