"""F1 tools/app_push.py (Kaggle export -> app storage, no Modal): checked with a fake Supabase client, never the real app."""
import json, os, sys, types, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "fake-match"


def _load():
    spec = importlib.util.spec_from_file_location("app_push", os.path.join(ROOT, "tools/app_push.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class _Fake:
    def __init__(self): self.uploads, self.rows = {}, []
    def create(self, url, key):
        fake = self
        class Bucket:
            def upload(self, dest, data, opts): fake.uploads[dest] = (len(data), opts["content-type"])
        class Table:
            def __init__(self, name): self.name = name
            def upsert(self, row): fake.rows.append((self.name, row)); return self
            def execute(self): return None
        return types.SimpleNamespace(storage=types.SimpleNamespace(from_=lambda b: Bucket()), table=Table)


def _export(tmp, with_thumb=True, drop=None):
    d = tmp / M; d.mkdir()
    files = {"match_data": f"{M}/match_data.json", "stats": f"{M}/stats.json", "frames": f"{M}/frames_000.json", "video": f"{M}/video.mp4"}
    json.dump({"fps": 25, "width": 1920, "height": 1080, "schema_version": 3, "contract": "v1"}, open(d / "match_data.json", "w"))
    json.dump({"possession": [50, 50]}, open(d / "stats.json", "w"))
    json.dump({"frames": []}, open(d / "frames_000.json", "w"))
    if drop: os.remove(d / drop)
    if with_thumb: (d / "thumb.jpg").write_bytes(b"\xff\xd8jpg")
    lib = {"matches": [{"id": M, "files": files, "duration_s": 3000, "summary": {"goals": [1, 0]}, "attack_right": "A",
                        "attack_right_confidence": 0.8, "video_url": "https://r2.example/fake-match/video.mp4"}]}
    json.dump(lib, open(tmp / "library.json", "w"))


def _fake_supabase(monkeypatch):
    fake = _Fake(); monkeypatch.setitem(sys.modules, "supabase", types.SimpleNamespace(create_client=fake.create)); return fake


def test_no_secrets_pushes_nothing(tmp_path, monkeypatch):
    _export(tmp_path); fake = _fake_supabase(monkeypatch)
    monkeypatch.delenv("SUPABASE_URL", raising=False); monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    out = []
    assert _load().push(str(tmp_path), log=out.append) is None
    assert fake.uploads == {} and fake.rows == [] and "not set" in out[0]


def test_push_uploads_files_and_row_video_stays_on_r2(tmp_path, monkeypatch):
    _export(tmp_path); fake = _fake_supabase(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co"); monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
    files = _load().push(str(tmp_path), log=lambda *a: None)
    assert set(fake.uploads) == {f"{M}/match_data.json", f"{M}/stats.json", f"{M}/frames_000.json", f"{M}/thumb.jpg"}
    assert not any(k.endswith(".mp4") for k in fake.uploads)          # the video is never uploaded
    assert files["video"] == "https://r2.example/fake-match/video.mp4" and files["thumb"] == f"{M}/thumb.jpg"
    assert fake.uploads[f"{M}/match_data.json"][1] == "application/json"
    (table, row), = fake.rows
    assert table == "matches" and row["id"] == M and row["status"] == "ready"
    assert row["fps"] == 25 and row["width"] == 1920 and row["duration_s"] == 3000 and row["files"] == files


def test_unknown_match_and_missing_file(tmp_path, monkeypatch):
    _export(tmp_path, with_thumb=False, drop="stats.json"); fake = _fake_supabase(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co"); monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
    mod = _load()
    assert mod.push(str(tmp_path), "other-match", log=lambda *a: None) is None and fake.rows == []
    out = []; files = mod.push(str(tmp_path), M, log=out.append)
    assert "stats" not in files and "thumb" not in files and any("missing" in s for s in out)
