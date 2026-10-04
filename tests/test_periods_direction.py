"""4 Oct: the first-period attack direction can be fixed by eye in periods/<match>.json ("attack_right_A")."""
import json, os
from ipanema.run import periods_direction

def test_reads_override(tmp_path):
    (tmp_path / "periods").mkdir(); json.dump({"periods_s": [[0, 10]], "attack_right_A": False}, open(tmp_path / "periods" / "m.json", "w"))
    assert periods_direction(str(tmp_path), "m") is False

def test_repo_files():
    assert periods_direction("/nonexistent", "SFKBP1109") is False          # SFK (dark) attacks left in the first half
    assert periods_direction("/nonexistent", "p15u-vs-aik-2026-09-21-bd09") is True
    assert periods_direction("/nonexistent", "no-such-match") is None
