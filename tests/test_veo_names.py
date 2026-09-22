"""Veo highlight file names -> shot/goal list (the real SFK-BP paste, links and all)."""
from scripts_veo_highlights import parse

PASTE = """[11 005944_-_Shot_on_goal.mp4](https://drive.google.com/open?id=1yne&usp=drive_copy)
[08 004339_-_Goal.mp4](https://drive.google.com/open?id=1sOJ&usp=drive_copy)
[01 000028_-_Goal.mp4](https://drive.google.com/open?id=17HT&usp=drive_copy)
[02 000028_-_Shot_on_goal.mp4](https://drive.google.com/open?id=1mzn&usp=drive_copy)
02 000028_-_Shot_on_goal.mp4
[29 013749_-_Shot_on_goal.mp4](x)"""

def test_parse_real_paste():
    rows = parse(PASTE)
    assert rows == [(28.0, "goal"), (28.0, "shot"), (2619.0, "goal"), (3584.0, "shot"), (5869.0, "shot")]   # sorted, duplicate dropped

def test_offset_and_duration():
    assert parse(PASTE, offset_s=593.0, duration_s=1807.0) == []                       # a 30-min clip from 9:53: none of these fall inside
    assert parse(PASTE, offset_s=593.0, duration_s=3000.0) == [(2026.0, "goal"), (2991.0, "shot")]   # a longer clip: the 43:39 goal and the 59:44 shot
