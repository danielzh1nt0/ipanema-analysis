"""N0-29d: the runs' with_ball label reads the spell state (IPANEMA_RUN_BALL=raw = old raw state)."""
import os, sys, json, subprocess, numpy as np
from ipanema import metrics as M

FPS = 25
AR = {"A": True, "B": False}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _sprint(start_s=2.0, run_s=3.0, n_s=8.0, v=7.0):
    """player 7 of team A: still, sprint from start_s for run_s, still; tiny wobble (exact lines = gap fills)"""
    n = int(n_s * FPS); a, b = int(start_s * FPS), int((start_s + run_s) * FPS); per = {}
    for k in range(n):
        x = 20 + v * (min(max(k, a), b) - a) / FPS
        per[k] = [[7, "A", np.array([x, 30.0 + 0.05 * np.sin(1.3 * k)]), None, None, False]]
    return per, n


def test_look_back_and_start_frame():
    per, n = _sprint()
    st = np.full(n, 2); st[:int(1.0 * FPS)] = 0                         # A had it until 1 s, loose at the start (2 s)
    assert M.runs(per, FPS, st, AR)[0]["with_ball"] is None              # first frame only: nobody
    assert M.runs(per, FPS, st, AR, look_s=3.0)[0]["with_ball"] is True  # 3-s look-back finds A
    st2 = np.full(n, 2); st2[int(1.5 * FPS):] = 1                        # B has it from 1.5 s
    assert M.runs(per, FPS, st2, AR)[0]["with_ball"] is False


def test_compute_reads_the_spell_state_by_default():
    per, n = _sprint()
    raw = np.full(n, 2)                                                  # raw state: loose (a flicker) at the run's start
    spell = np.zeros(n, int)                                             # spell state: A's spell
    mx = M.compute(raw, {}, {}, FPS, 105.0, 68.0, AR, [], [], [], [], [], per=per, cstate=spell)
    assert M.RUN_BALL == "spell" and mx["runs"][0]["with_ball"] is True
    assert M.run_ball_state(raw, None) is raw                            # no spell state given -> raw
    mx0 = M.compute(raw, {}, {}, FPS, 105.0, 68.0, AR, [], [], [], [], [], per=per)
    assert mx0["runs"][0]["with_ball"] is None


def test_old_switch():
    code = ("import numpy as np; from ipanema import metrics as M; r = np.full(5, 2); c = np.zeros(5, int); "
            "print(M.RUN_BALL, M.run_ball_state(r, c) is r)")
    env = dict(os.environ, PYTHONPATH=ROOT, IPANEMA_RUN_BALL="raw")
    assert subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT).stdout.strip() == "raw True"


def test_lab_labels_match_the_app():
    import n0_29d_runball as R
    st = np.array([0] * 10 + [2] * 10 + [1] * 10); cs = np.array([0] * 20 + [1] * 10)
    assert R.label("raw_start", st, cs, 12, 25, "A") is None
    assert R.label("spell_start", st, cs, 12, 25, "A") is True
    assert R.label("spell_back", st, st, 12, 25, "A", fps=FPS) is True
    assert R.label("spell_major", st, cs, 12, 29, "B") is True           # spell frames 12-29: A 8 (12-19), B 10 (20-29) -> B
    per, n = _sprint(); s = np.zeros(n, int); s[:int(2.5 * FPS)] = 2
    for look, kind in ((0.0, "spell_start"), (3.0, "spell_back")):
        r = M.runs(per, FPS, s, AR, look_s=look)[0]; a = int(round(r["t"] * FPS))
        assert r["with_ball"] == R.label(kind, s, s, a, a + 10, "A")


def test_truth_and_scores():
    import n0_29d_runball as R
    seqs = [{"team": "A", "start": 100, "end": 200}, {"team": "B", "start": 250, "end": 300}]
    assert R.truth_team_at(seqs, 150) == "A" and R.truth_team_at(seqs, 220) is None and R.truth_team_at(seqs, 260) == "B"
    rows = [{"team": "A", "truth": True, "label": {k: True for k in R.LABELS}}, {"team": "A", "truth": False, "label": {k: None for k in R.LABELS}},
            {"team": "B", "truth": None, "label": {k: True for k in R.LABELS}}]
    S = R.summarise(rows)
    assert S["truth_known_pct"] == 67 and S["raw_start"]["labelled_pct"] == 67 and S["raw_start"]["right_of_labelled_pct"] == 100 and S["raw_start"]["right_of_known_pct"] == 50
    assert R.truth_poss_share(seqs) == {"A": 101 / 152, "B": 51 / 152}


def test_dry_run():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "n0_29d_runball.py"), "--dry"], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=ROOT), timeout=300)
    assert r.returncode == 0, r.stderr
    S = json.loads(r.stdout)
    assert S["runs"] == 2 and S["spell_back"]["right_of_known_pct"] == 100
