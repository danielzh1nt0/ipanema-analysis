"""18 Oct (worker, N0): tests for ipanema/progress.py (the app's live 'analysing...' bar, fed every log line of a Modal
run; was untested). No Supabase: a fake table records what would be written. Also replays three real run logs from
results/modal/ (skipped if missing): the bar must never go backwards and must end near 100%."""
import os
import pytest

from ipanema import progress as PR
from ipanema.progress import Progress, STAGES


class _Q:
    def __init__(self, sink): self.sink = sink; self.row = {}
    def update(self, row): self.row["update"] = row; return self
    def eq(self, k, v): self.row[k] = v; return self
    def execute(self): self.sink.append(self.row)


class _DB:
    def __init__(self): self.writes = []; self.tables = []
    def table(self, name): self.tables.append(name); return _Q(self.writes)


def _prog(match_id="M1"):
    p = Progress(match_id, every_s=1e9); p.db = _DB(); return p


def test_weights_add_to_one_and_ids():
    assert abs(sum(w for _, w in STAGES) - 1.0) < 1e-9
    assert Progress("M1_s1200").row == "M1" and Progress("M1_s1200").segment == "M1_s1200"
    assert Progress("M1").row == "M1" and Progress("M1").segment is None


def test_stages_advance_and_never_go_back():
    p = _prog()
    p.feed("=== M1 full match === video: 1000 frames (33 s) @ 30.000 fps -> 1 pieces")
    assert p.frames == 1000
    p.feed("  calibration frame 500"); assert p.stage == 0 and abs(p.frac - 0.5) < 1e-9
    p.feed("  tracking frame 500"); assert p.stage == 1 and abs(p.frac - (0.08 + 0.45)) < 1e-9
    p.feed("  ball frame 250"); assert p.stage == 2 and abs(p.frac - 0.25) < 1e-9
    p.feed("teams: offline override changed 12 player rows")          # a late stage-1 line must not pull the bar back
    assert p.stage == 2 and abs(p.frac - 0.25) < 1e-9
    p.feed("clean: dropped 3 off-pitch")                                 # stage 1 'done' must not fill the ball stage
    assert p.stage == 2 and abs(p.frac - 0.25) < 1e-9
    p.feed("  ball frame 100"); assert abs(p.frac - 0.25) < 1e-9        # a lower count inside a stage doesn't either
    p.feed("  step: passes (3 s since the last step)"); assert p.stage == 3 and abs(p.frac - 4 / 7) < 1e-9
    p.feed("  step: export (1 s)"); assert p.stage == 4
    p.feed("upload: M1 -> Supabase + R2 (video)"); assert abs(p.frac - 0.9) < 1e-9
    assert abs(p.overall() - (0.9 + 0.1 * 0.9)) < 1e-9


def test_unknown_frame_count_does_not_crash_and_caps_at_full():
    p = _prog(); p.feed("  tracking frame 9000")                        # no 'N frames (' line seen yet
    assert p.stage == 1 and p.frac == 1.0


def test_push_writes_only_processing_rows_and_eta():
    p = _prog("M1_s1200"); p.t0 -= 100                                   # 100 s in
    p.feed("  step: export")                                             # stage change -> pushes now
    w = p.db.writes[-1]
    assert p.db.tables[-1] == "matches" and w["id"] == "M1" and w["status"] == "processing"
    prog = w["update"]["summary"]["progress"]
    assert prog["stage"] == 4 and prog["label"] == "Writing findings" and prog["scope"] == "5-minute test"
    assert prog["pct"] == round(p.overall() * 100) and prog["eta_s"] is not None and prog["eta_s"] >= 0
    p2 = _prog(); p2.push()                                              # nothing done yet -> no ETA guess
    assert p2.db.writes[-1]["update"]["summary"]["progress"]["eta_s"] is None


def test_finished_segment_and_error():
    p = _prog("M1_s60"); p.finished(True)
    prog = p.db.writes[-1]["update"]["summary"]["progress"]
    assert prog["pct"] == 100 and prog["segment_ready"] == "M1_s60" and prog["label"] == "5-minute test ready"
    q = _prog(); q.finished(False); assert q.db.writes[-1]["update"]["summary"]["progress"]["error"] is True
    r = _prog(); r.finished(True); assert r.db.writes == []              # full match done: the run itself writes the row


def test_no_credentials_and_db_failure_are_quiet(monkeypatch, capsys):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    p = Progress("M1"); assert p.db is None; p.feed("  step: export"); p.finished(False)   # no crash, nothing to write

    class _Bad:
        def table(self, name): raise RuntimeError("network down")
    q = Progress("M1"); q.db = _Bad(); q.push()
    assert "progress update failed" in capsys.readouterr().out


REAL = ["results/modal/SFKBP1109_full.txt", "results/modal/SFKBP1109_s1200.txt",
        "results/modal/p15u-vs-vallentuna-2026-10-03-6cce_full.txt"]


@pytest.mark.parametrize("path", REAL)
def test_real_logs_bar_only_moves_forward(path):
    full = os.path.join(os.path.dirname(__file__), "..", path)
    if not os.path.exists(full): pytest.skip("log not in this checkout")
    p = _prog("M_s1200" if "_s" in os.path.basename(path) else "M"); seen = []
    for line in open(full, errors="replace"):
        p.feed(line.rstrip("\n")); seen.append(p.overall())
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:]))
    assert p.stage == 4 and seen[-1] > 0.95                              # reached 'Writing findings', upload seen
