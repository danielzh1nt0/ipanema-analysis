"""18 Oct (N0 worker): tools/reviews_to_labels.py (the coach's reviews / corrections / lineups -> answer keys) and the
coach part of tools/export_qa.py, with a fake Supabase client - no secrets, no network."""
import json, os, sys, types
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import reviews_to_labels as R  # noqa: E402
import export_qa as Q  # noqa: E402

EVENTS = [{"id": "pass_521.52_101503", "t": 521.52, "type": "better_option", "team": "A"},
          {"id": "lost_15202", "t": 506.7, "type": "turnover_lost", "team": "B"},
          {"id": "restart_0.4", "t": 0.4, "type": "set_piece", "team": "A"}]


def test_id_type_and_time():
    assert R.id_type("lost_15202") == "turnover_lost" and R.id_type("hto_510.08") == "high_turnover"
    assert R.id_type("seq_200") == "sequence_end" and R.id_type("weird_1") == "weird"
    assert R.id_time("pass_521.52_101503") == 521.52 and R.id_time("restart_1.87") == 1.87
    assert R.id_time("lost_15202") is None  # a frame number, not seconds
    assert R.id_time("pass_x_1") is None


def test_build_match():
    ev = {e["id"]: e for e in EVENTS}
    reviews = [{"event_id": "pass_521.52_101503", "verdict": "retimed", "t_corrected": 522.0, "team_corrected": "B"},
               {"event_id": "lost_15202", "verdict": "deleted"},
               {"event_id": "restart_0.4", "verdict": "confirmed"},
               {"event_id": "hto_510.08", "verdict": "confirmed"},     # not in this export: type + time from the id
               {"event_id": "lost_99999", "verdict": "confirmed"}]     # not in the export, no time: no kick window
    corr = [{"field": "possession_pct", "value": [60, 40], "updated_at": "2026-10-18T10:00:00Z"},
            {"field": "possession_pct", "value": [55, 45], "updated_at": "2026-10-18T12:00:00Z"},  # latest wins
            {"field": "score", "value": [2, 1], "created_at": "2026-10-18T09:00:00Z"}]
    corr_rev = list(reversed(corr))
    lineups = [{"player_id": "p1", "shirt": 9, "started": True, "minute_on": 0}]
    ek, ck, kw = R.build_match("m", reviews, corr, lineups, ev)
    it = {i["event_id"]: i for i in ek["items"]}
    assert it["pass_521.52_101503"] == {"event_id": "pass_521.52_101503", "t": 522.0, "type": "better_option", "team": "B", "verdict": "retimed", "note": None}
    assert it["hto_510.08"]["type"] == "high_turnover" and it["hto_510.08"]["t"] == 510.08
    assert it["lost_99999"]["t"] is None and it["lost_99999"]["type"] == "turnover_lost"
    w = {x["type"]: x for x in kw["windows"]}
    assert set(w) == {"better_option", "set_piece", "high_turnover"}  # deleted + no-time ones left out
    assert w["set_piece"]["t0"] == 0.0 and w["set_piece"]["t1"] == 1.4  # never a negative start
    assert ck["possession_pct"] == [55, 45] and ck["score"] == [2, 1] and ck["lineup"][0]["shirt"] == 9
    assert R.build_match("m", reviews, corr_rev, lineups, ev)[1]["possession_pct"] == [55, 45]  # row order does not matter
    assert R.build_match("m", [], [], [], {}) == (None, None, None)


class _Q:
    def __init__(self, data): self.data = data
    def select(self, *_): return self
    def execute(self): return self


class _SB:
    def __init__(self, tables, storage_ok=True):
        self.tables = tables; self.storage_ok = storage_ok
        self.storage = types.SimpleNamespace(from_=lambda bucket: self)
    def table(self, name): return _Q(self.tables.get(name, []))
    def download(self, path):
        if not self.storage_ok: raise RuntimeError("not public")
        return json.dumps({"events": EVENTS}).encode()


def _fake_supabase(monkeypatch, sb):
    mod = types.ModuleType("supabase"); mod.create_client = lambda url, key: sb
    monkeypatch.setitem(sys.modules, "supabase", mod)
    monkeypatch.setenv("SUPABASE_URL", "https://x"); monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")


def test_main_writes_keys(tmp_path, monkeypatch):
    sb = _SB({"event_reviews": [{"match_id": "m1", "event_id": "lost_15202", "verdict": "confirmed"},
                                {"match_id": "m2", "event_id": "pass_521.52_101503", "verdict": "deleted"}],
              "match_corrections": [{"match_id": "m3", "field": "score", "value": [1, 0]}],
              "match_lineups": []})
    _fake_supabase(monkeypatch, sb)
    R.main([], root=str(tmp_path))  # results/labels does not exist yet: created
    ek = json.load(open(tmp_path / "reference/m1/events_key.json"))
    assert ek["items"][0]["t"] == 506.7 and ek["items"][0]["team"] == "B"  # from the export in storage
    assert json.load(open(tmp_path / "results/labels/m1_kick_windows.json"))["windows"][0]["t0"] == 505.7
    assert json.load(open(tmp_path / "results/labels/m2_kick_windows.json"))["windows"] == []
    assert json.load(open(tmp_path / "reference/m3/coach_key.json"))["score"] == [1, 0]
    assert not (tmp_path / "reference/m3/events_key.json").exists()
    s = json.load(open(tmp_path / "results/labels/reviews_to_labels.json"))
    assert s == {"m1": {"reviews": 1, "corrections": 0, "lineup_rows": 0}, "m2": {"reviews": 1, "corrections": 0, "lineup_rows": 0},
                 "m3": {"reviews": 0, "corrections": 1, "lineup_rows": 0}}
    R.main(["m1"], root=str(tmp_path / "only"))  # match filter
    assert os.listdir(tmp_path / "only/reference") == ["m1"]


def test_main_storage_unreadable_and_no_rows(tmp_path, monkeypatch):
    _fake_supabase(monkeypatch, _SB({"event_reviews": [{"match_id": "m1", "event_id": "pass_521.52_101503", "verdict": "confirmed"}]}, storage_ok=False))
    R.main([], root=str(tmp_path))
    it = json.load(open(tmp_path / "reference/m1/events_key.json"))["items"][0]
    assert it["t"] == 521.52 and it["type"] == "pass"  # from the id
    _fake_supabase(monkeypatch, _SB({}))
    R.main([], root=str(tmp_path / "empty"))  # no rows at all: no crash, empty summary
    assert json.load(open(tmp_path / "empty/results/labels/reviews_to_labels.json")) == {}


def test_main_without_secrets_exits_2(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False); monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    with pytest.raises(SystemExit) as e: R.main([])
    assert e.value.code == 2


def test_export_qa_coach_events_survive_a_rerun(tmp_path):
    """After a re-run the ids move (pass_<t>_<player id>, lost_<frame>): the grade must still find the event."""
    folder = tmp_path / "exp"; ref = tmp_path / "ref"; folder.mkdir(); ref.mkdir()
    rerun = [{"id": "pass_521.61_200007", "t": 521.61, "type": "better_option"},   # same pass, new player id
             {"id": "lost_15210", "t": 507.0, "type": "turnover_lost"},          # same loss, frame moved
             {"id": "won_15210", "t": 530.0, "type": "turnover_won"}]
    json.dump({"fps": 10, "frames": [{"t": 0.0}], "events": rerun}, open(folder / "match_data.json", "w"))
    json.dump({"possession_pct": [52, 48], "goals": [1, 1]}, open(folder / "summary.json", "w"))
    items = [{"event_id": "pass_521.52_101503", "t": 521.52, "verdict": "confirmed"},
             {"event_id": "lost_15202", "t": 506.7, "verdict": "deleted"},
             {"event_id": "restart_100.0", "t": 100.0, "verdict": "confirmed"},     # gone from the export
             {"event_id": "won_99", "t": None, "verdict": "confirmed"}]           # no time, id not present
    json.dump({"items": items}, open(ref / "events_key.json", "w"))
    json.dump({"possession_pct": [55, 45], "score": [2, 1], "lineup": [{}, {}]}, open(ref / "coach_key.json", "w"))
    out = Q.grade(str(folder), str(ref), log=lambda *_: None)
    assert out["coach_events"] == {"confirmed_still_present": 1, "deleted_still_present": 1, "n": 4}
    assert out["coach"] == {"possession_pct": {"coach": [55, 45], "ours": [52, 48]}, "score": {"coach": [2, 1], "ours": [1, 1]}, "lineup_rows": 2}
    # another kind of event at the same time is not a match
    assert not Q.coach_event_present({"event_id": "won_1", "t": 521.52}, [{"id": "pass_521.52_1", "t": 521.52}])
    assert not Q.coach_event_present({"event_id": "pass_1_1", "t": 10.0}, [{"id": "pass_12.0_1", "t": 12.0}])


def test_real_export_every_event_finds_itself():
    p = os.path.join(ROOT, "results/volume/runs/matches/SFKBP1109_s1200/match_data.json")
    if not os.path.exists(p): pytest.skip("repo export missing")
    evs = json.load(open(p))["events"]
    assert all(Q.coach_event_present({"event_id": e["id"], "t": e["t"]}, evs) for e in evs)
    assert all(R.id_type(e["id"]) == e["type"] or e["id"].startswith(("pass", "shot")) for e in evs)
