"""17 Oct (worker, N0): tests for ipanema/segments.py (cuts full games into clips before every run; was untested).
Made-up 4-s videos in a temp root, real ffmpeg (skipped if missing)."""
import json, os, shutil
import cv2, numpy as np
import pytest

from ipanema.segments import prepare_segments

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _video(path, seconds=4, fps=10, size=(160, 90)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for k in range(seconds * fps): vw.write(np.full((size[1], size[0], 3), (40, 100 + k, 40), np.uint8))
    vw.release()


def _seconds(path):
    c = cv2.VideoCapture(str(path)); n, fps = c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FPS); c.release()
    return n / fps


def test_cuts_segments_from_config_and_biggest_file(tmp_path):
    m = tmp_path / "videos" / "M1"; m.mkdir(parents=True)
    _video(m / "small.mp4", seconds=1); _video(m / "full.mp4", seconds=4)          # the big one is the full game
    json.dump([[1, 2], [3, 5]], open(m / "segments.json", "w"))
    made = prepare_segments(str(tmp_path), log=lambda s: None)
    assert sorted(os.path.basename(p) for p in made) == ["M1_s1.mp4", "M1_s3.mp4"]
    assert abs(_seconds(tmp_path / "videos" / "M1_s1.mp4") - 2) < 0.3            # cut from full.mp4, not small.mp4 (1 s)
    assert abs(_seconds(tmp_path / "videos" / "M1_s3.mp4") - 1) < 0.3            # runs to the end of the video
    assert prepare_segments(str(tmp_path), log=lambda s: None) == []              # second run: already there, nothing redone
    assert not [f for f in os.listdir(tmp_path / "videos") if f.endswith(".tmp.mp4")]


def test_default_segment_and_no_video_folder(tmp_path):
    assert prepare_segments(str(tmp_path), log=lambda s: None) == []              # no videos/ at all: quietly nothing
    m = tmp_path / "videos" / "M2"; m.mkdir(parents=True); (tmp_path / "videos" / "M2_clip.mp4").write_bytes(b"")
    (tmp_path / "videos" / "empty").mkdir()                                         # a match folder without a video is skipped
    _video(m / "game.mp4", seconds=2)
    made = prepare_segments(str(tmp_path), log=lambda s: None)
    assert [os.path.basename(p) for p in made] == ["M2_s0.mp4"]                    # default (0, 300)


def test_failed_cut_leaves_no_half_file(tmp_path):
    m = tmp_path / "videos" / "M3"; m.mkdir(parents=True)
    (m / "full.mp4").write_bytes(b"not a video" * 100)
    logs = []
    assert prepare_segments(str(tmp_path), log=logs.append) == []
    assert any("ffmpeg failed" in s for s in logs)
    # nothing ending in .mp4 is left in videos/ (the run loop would take it for a clip)
    assert not [f for f in os.listdir(tmp_path / "videos") if f.lower().endswith(".mp4")]


def test_ffmpeg_dying_mid_cut_removes_its_partial_output(tmp_path, monkeypatch):
    m = tmp_path / "videos" / "M4"; m.mkdir(parents=True); _video(m / "full.mp4", seconds=1)
    import subprocess, types
    def dies(cmd, **kw):                                  # ffmpeg wrote part of the output, then failed (disk full, killed, broken input)
        open(cmd[-1], "wb").write(b"partial"); return types.SimpleNamespace(returncode=1, stderr=b"No space left on device")
    monkeypatch.setattr(subprocess, "run", dies)
    assert prepare_segments(str(tmp_path), log=lambda s: None) == []
    assert not [f for f in os.listdir(tmp_path / "videos") if f.lower().endswith(".mp4")]
