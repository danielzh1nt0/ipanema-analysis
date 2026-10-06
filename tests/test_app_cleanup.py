from tools import app_cleanup as AC

class Fake:
    def __init__(self):
        self.store = {"old1": ["stats.json", "match_data.json"], "SFKBP1109": ["stats.json"], "old2": []}
        self.rows_ = {"matches": {"old1", "SFKBP1109", "old2"}, "match_labels": {"old1", "SFKBP1109"}}; self.removed = []
    def list(self, prefix): return [{"name": n} for n in self.store.get(prefix, [])]
    def remove(self, paths): self.removed += paths; [self.store[p.split("/")[0]].remove(p.split("/", 1)[1]) for p in paths]
    def delete_rows(self, table, column, value): self.rows_[table].discard(value)

def test_plan_keeps_only_the_demo_list():
    rows = [{"id": "old1"}, {"id": "SFKBP1109"}, {"id": "old2"}, {"id": None}]
    assert AC.plan(rows, ["SFKBP1109"]) == ["old1", "old2"]

def test_apply_removes_files_and_rows_but_not_kept():
    f = Fake(); rep = AC.apply(f, AC.plan([{"id": i} for i in f.rows_["matches"]], ["SFKBP1109"]), log=lambda *a: None)
    assert sorted(f.removed) == ["old1/match_data.json", "old1/stats.json"]
    assert f.rows_["matches"] == {"SFKBP1109"} and f.rows_["match_labels"] == {"SFKBP1109"} and not rep["errors"]
    assert f.store["SFKBP1109"] == ["stats.json"]

def test_demo_keep_has_all_three_demo_matches():
    assert "p15u-vs-vallentuna-2026-10-03-6cce" in AC.DEMO_KEEP and len(AC.DEMO_KEEP) == 3

def test_check_keep_blocks_old_keep_list_that_would_delete_vallentuna():
    rows = [{"id": i} for i in AC.DEMO_KEEP + ["old1"]]
    old = ["SFKBP1109", "p15u-vs-aik-2026-09-21-bd09"]          # the keep list written in BACKLOG D5 before 6 Oct
    probs = AC.check_keep(old, rows, apply=True)
    assert any("vallentuna" in p for p in probs)
    assert "p15u-vs-vallentuna-2026-10-03-6cce" in AC.plan(rows, old)   # what it would have deleted
    assert AC.check_keep(AC.DEMO_KEEP, rows, apply=True) == [] and AC.plan(rows, AC.DEMO_KEEP) == ["old1"]

def test_check_keep_blocks_typo_in_keep():
    rows = [{"id": i} for i in AC.DEMO_KEEP]
    assert AC.check_keep(AC.DEMO_KEEP + ["SFKBP11O9"], rows, apply=True) == ["keep id SFKBP11O9 not found in the app"]
