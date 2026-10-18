"""18 Oct (worker, N0): tests for two pieces every app run calls and that had no tests:
ipanema/upload.py (sends a finished match to Supabase + R2 at the end of run()) and ipanema/video.py
(normalise = the first re-encode of every run; info / frames / frame_at). Supabase and boto3 are faked, nothing is sent."""
import json, os, shutil, subprocess, sys, types
import cv2, numpy as np
import pytest

from ipanema import upload as U
from ipanema import video as V

KEYS = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET", "R2_PUBLIC_URL")


# ---------------- fakes ----------------
class _Calls:
    def __init__(self): self.storage, self.rows, self.r2_put, self.r2_head = [], [], [], []

class _Bucket:
    def __init__(self, c, name): self.c, self.name = c, name
    def upload(self, dest, data, opts): self.c.storage.append((self.name, dest, len(data), opts["content-type"], opts["upsert"]))

class _Table:
    def __init__(self, c, name): self.c, self.name, self.row = c, name, None
    def upsert(self, row): self.row = row; return self
    def execute(self): self.c.rows.append((self.name, self.row))

class _SB:
    def __init__(self, c):
        self.c = c; self.storage = types.SimpleNamespace(from_=lambda n: _Bucket(c, n))
    def table(self, n): return _Table(self.c, n)

class _R2:
    def __init__(self, c, have): self.c, self.have = c, have
    def head_object(self, Bucket, Key):
        self.c.r2_head.append(Key)
        if Key not in self.have: raise RuntimeError("404")
    def upload_file(self, local, bucket, dest, ExtraArgs): self.c.r2_put.append((local, bucket, dest, ExtraArgs["ContentType"]))


@pytest.fixture
def fake(monkeypatch):
    c = _Calls(); c.have = set()
    monkeypatch.setitem(sys.modules, "supabase", types.SimpleNamespace(create_client=lambda url, key: _SB(c)))
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *a, **k: _R2(c, c.have)))
    monkeypatch.setitem(sys.modules, "google.colab", None)          # no Colab: credentials come from the environment
    for k in KEYS: monkeypatch.setenv(k, f"x-{k.lower()}")
    monkeypatch.setenv("R2_PUBLIC_URL", "https://cdn.example/")
    return c


def _match(root, mid="M1", video=True, video_url=None, thumb=True):
    d = root / "runs" / "matches" / mid; d.mkdir(parents=True)
    json.dump({"fps": 25, "width": 1280, "height": 720, "schema_version": 3, "contract": "v1", "frames": []}, open(d / "match_data.json", "w"))
    json.dump({"metrics": {}}, open(d / "stats.json", "w"))
    (d / "kit_A.png").write_bytes(b"png")                               # kit_B.png missing on purpose
    (d / "frames_000.json").write_text("[]")
    if video: (d / "video.mp4").write_bytes(b"0" * 100)
    if thumb: (d / "thumb.jpg").write_bytes(b"jpg")
    entry = {"id": mid, "duration_s": 300.0, "video_url": video_url, "summary": {"passes": 3}, "attack_right": {"A": True}, "attack_right_confidence": 0.8,
             "files": {"video": f"matches/{mid}/video.mp4", "match_data": f"matches/{mid}/match_data.json", "stats": f"matches/{mid}/stats.json",
                       "kit_A": f"matches/{mid}/kit_A.png", "kit_B": f"matches/{mid}/kit_B.png", "frames_000": f"matches/{mid}/frames_000.json"}}
    lib = root / "runs" / "library.json"
    old = json.load(open(lib))["matches"] if lib.exists() else []
    json.dump({"matches": old + [entry]}, open(lib, "w"))
    return d


# ---------------- upload ----------------
def test_upload_sends_files_and_row(tmp_path, fake):
    _match(tmp_path); _match(tmp_path, "OTHER")
    files = U.upload_match(str(tmp_path), "M1", log=lambda s: None)
    assert files["video"] == "https://cdn.example/M1/video.mp4"         # public R2 address, no double slash
    assert fake.r2_put == [(str(tmp_path / "runs/matches/M1/video.mp4"), "x-r2_bucket", "M1/video.mp4", "video/mp4")]
    up = {dest: (bucket, ctype, ups) for bucket, dest, n, ctype, ups in fake.storage}
    assert set(up) == {"M1/match_data.json", "M1/stats.json", "M1/kit_A.png", "M1/frames_000.json", "M1/thumb.jpg"}   # missing kit_B skipped
    assert up["M1/match_data.json"] == ("matches", "application/json", "true") and up["M1/kit_A.png"][1] == "image/png"
    assert "kit_B" not in files and files["thumb"] == "M1/thumb.jpg" and files["stats"] == "M1/stats.json"
    (tname, row), = fake.rows
    assert tname == "matches" and row["id"] == "M1" and row["status"] == "ready" and row["fps"] == 25 and row["duration_s"] == 300.0
    assert row["files"] == files and row["summary"] == {"passes": 3} and row["attack_right"] == {"A": True}
    assert not any("OTHER" in d for _, d, *_ in fake.storage)            # only the asked match


def test_upload_skips_video_already_on_r2(tmp_path, fake):
    _match(tmp_path); fake.have.add("M1/video.mp4")
    files = U.upload_match(str(tmp_path), "M1", log=lambda s: None)
    assert fake.r2_head == ["M1/video.mp4"] and fake.r2_put == [] and files["video"] == "https://cdn.example/M1/video.mp4"


def test_upload_keeps_video_url_when_no_local_video(tmp_path, fake):
    _match(tmp_path, video=False, video_url="https://r2.example/full/M1.mp4", thumb=False)
    files = U.upload_match(str(tmp_path), "M1", log=lambda s: None)
    assert files["video"] == "https://r2.example/full/M1.mp4" and fake.r2_put == [] and fake.r2_head == []
    assert "thumb" not in files and not any(d.endswith("thumb.jpg") for _, d, *_ in fake.storage)


def test_upload_quiet_without_credentials_or_match(tmp_path, fake, monkeypatch):
    _match(tmp_path); msgs = []
    assert U.upload_match(str(tmp_path), "NOPE", log=msgs.append) is None and "not in library" in msgs[-1]
    monkeypatch.delenv("R2_BUCKET")
    assert U.upload_match(str(tmp_path), "M1", log=msgs.append) is None and "no Supabase/R2 credentials" in msgs[-1]
    assert fake.rows == [] and fake.storage == [] and fake.r2_put == []


# ---------------- video ----------------
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _video(path, n=20, fps=10, size=(160, 90)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for k in range(n): vw.write(np.full((size[1], size[0], 3), (10 * k, 120, 40), np.uint8))
    vw.release()


@needs_ffmpeg
def test_normalise_info_frames(tmp_path):
    src = tmp_path / "in.mp4"; _video(src)
    dst = tmp_path / "work" / "M" / "video.mp4"
    assert V.normalise(str(src), str(dst)) == str(dst) and dst.exists()
    i = V.info(str(dst)); assert (i["width"], i["height"], i["n"]) == (160, 90, 20) and abs(i["fps"] - 10) < 0.01
    ks = [k for k, f in V.frames(str(dst))]; assert ks == list(range(20))
    f = V.frame_at(str(dst), 15); assert f is not None and abs(int(f[45, 80, 0]) - 150) < 12   # the right frame (blue = 10 k)
    assert V.frame_at(str(dst), 500) is None
    m = dst.stat().st_mtime_ns; V.normalise(str(src), str(dst)); assert dst.stat().st_mtime_ns == m   # done once, not redone
    assert not list(dst.parent.glob("*.part.mp4"))


@needs_ffmpeg
def test_normalise_failure_leaves_no_file(tmp_path, monkeypatch):
    bad = tmp_path / "broken.mp4"; bad.write_bytes(b"not a video")
    dst = tmp_path / "work" / "video.mp4"
    with pytest.raises(subprocess.CalledProcessError): V.normalise(str(bad), str(dst))
    assert not dst.exists() and not list(dst.parent.glob("*"))
    # ffmpeg dies half way (error, timeout, Ctrl-C) after writing part of its output: the old code left that
    # half video at dst, and the next call (and modal_app's 'if not os.path.exists(video)') took it as done
    real = subprocess.run
    for err in (subprocess.CalledProcessError(1, "ffmpeg"), KeyboardInterrupt()):
        def dies(cmd, **kw):
            open(cmd[-1], "wb").write(b"half a video"); raise err
        monkeypatch.setattr(V.subprocess, "run", dies)
        with pytest.raises(type(err)): V.normalise(str(bad), str(dst))
        assert not dst.exists() and not list(dst.parent.glob("*"))
    monkeypatch.setattr(V.subprocess, "run", real)
    src = tmp_path / "in.mp4"; _video(src)
    V.normalise(str(src), str(dst)); assert V.info(str(dst))["n"] == 20  # a later good call still works
