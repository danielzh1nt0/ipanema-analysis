"""V2 (5 Oct): automatic upload checks flag parts of the app as 'needs review'."""
import json
from ipanema import uploadcheck as UC

L, W = 106.0, 64.0
PER = [{"index": 1, "t_start": 0.0, "t_end": 1000.0, "attack_right": {"A": True, "B": False}}]


def _p(team, x=50.0, y=30.0, gk=False, state="observed"):
    return {"team": team, "gk": gk, "state": state, "m": [x, y]}


def _frames(n=100, a=7, b=7, gk_a=1, ball=(50.0, 30.0), poss="A", t0=0.0, dt=0.1):
    out = []
    for i in range(n):
        pl = [_p("A", gk=(j < gk_a)) for j in range(a)] + [_p("B") for _ in range(b)]
        out.append({"t": round(t0 + i * dt, 3), "players": pl, "ball": {"m": list(ball), "state": "observed"}, "possession": poss, "cal_ok": True})
    return out


def _md(events=()):
    return {"pitch": {"length": L, "width": W}, "periods": PER, "events": list(events), "restarts": []}


def _stats(shots=()):
    return {"metrics": {"shots": list(shots)}}


def _by(rep, name):
    return next(r for r in rep["checks"] if r["name"] == name)


def test_clean_match_needs_nothing():
    rep = UC.run_checks(_md(), _stats([{"t": 10, "team": "A", "x_m": 95, "y_m": 30, "goal": False}]), _frames())
    assert rep["needs_review"] == []
    assert _by(rep, "players")["ok"] and _by(rep, "keepers")["ok"] and _by(rep, "ball")["ok"]
    assert _by(rep, "goals")["ok"] is None and _by(rep, "possession")["ok"] is None        # nothing to check -> not flagged


def test_lopsided_teams_and_keepers_flagged():
    rep = UC.run_checks(_md(), _stats(), _frames(a=3, b=11))                                # Vallentuna 4 Oct: 3 vs 11
    assert "players" in rep["needs_review"]
    rep = UC.run_checks(_md(), _stats(), _frames(gk_a=3))                                   # keeper rule painting the goalmouth
    assert rep["needs_review"] == ["keepers"]


def test_shots_without_spot_and_ball_off_pitch():
    rep = UC.run_checks(_md(), _stats([{"t": 10, "team": "A", "x_m": None, "y_m": None, "goal": False}]), _frames(ball=(-30.0, 30.0)))
    assert set(rep["needs_review"]) == {"shot_map", "ball_map"}


def _goal_frames(kicker):
    """goal at t=100; from 130 s the ball sits on the centre spot with `kicker` on it"""
    fr = _frames(n=500, t0=100.0, ball=(95.0, 30.0), poss=None)
    for f in fr:
        if f["t"] >= 130: f["ball"]["m"] = [L / 2, W / 2]; f["possession"] = kicker
    return fr


def test_goal_kickoff_by_the_other_team_is_ok():
    g = {"t": 100.0, "team": "A", "x_m": None, "goal": True, "located_by": "end only"}
    ev = [{"t": 100.0, "type": "goal", "team": "A", "payload": {}}]
    rep = UC.run_checks(_md(ev), _stats([g]), _goal_frames("B"))
    assert _by(rep, "goals")["ok"] is True and "B kicks off" in _by(rep, "goals")["detail"]


def test_goal_kickoff_by_the_scorer_is_flagged_unless_checked_by_eye():
    ev = [{"t": 100.0, "type": "goal", "team": "A", "payload": {}}]
    g = {"t": 100.0, "team": "A", "x_m": None, "goal": True, "located_by": "end only"}
    assert "goals" in UC.run_checks(_md(ev), _stats([g]), _goal_frames("A"))["needs_review"]
    g["located_by"] = "end only, team by eye"                                              # by eye wins over our possession reading
    assert "goals" not in UC.run_checks(_md(ev), _stats([g]), _goal_frames("A"))["needs_review"]


def test_goal_at_the_wrong_end_and_files_disagreeing():
    g = {"t": 100.0, "team": "A", "x_m": 2.0, "y_m": 30.0, "goal": True}                   # A attacks right but scores at the left end
    ev = [{"t": 100.0, "type": "goal", "team": "A", "payload": {}}]
    assert "goals" in UC.run_checks(_md(ev), _stats([g]), _frames())["needs_review"]
    g["x_m"] = 104.0; ev[0]["team"] = "B"                                                   # SFK-BP repo copy: stats A, events B
    rep = UC.run_checks(_md(ev), _stats([g]), _frames())
    assert "goals" in rep["needs_review"] and "match_data" in _by(rep, "goals")["detail"]


def test_possession_against_a_by_eye_key(tmp_path):
    ref = tmp_path / "reference" / "m1"; ref.mkdir(parents=True)
    (ref / "owner_key.json").write_text(json.dumps({"items": [{"t": i, "owner": "A"} for i in range(1, 13)] + [{"t": 13, "owner": "loose"}]}))
    key = UC.load_owner_key(str(tmp_path), "m1")
    assert len(key) == 12                                                                   # loose moments are not graded
    fr = _frames(n=200, dt=0.1)                                                             # 0..20 s, all 'A'
    assert _by(UC.run_checks(_md(), _stats(), fr, key), "possession")["ok"] is True
    for f in fr: f["possession"] = "B" if f["t"] < 8 else "A"
    rep = UC.run_checks(_md(), _stats(), fr, key)
    assert rep["needs_review"] == ["possession"] and _by(rep, "possession")["value"] < 0.7


def test_safe_run_never_breaks_an_upload():
    logs = []
    assert UC.safe_run({"pitch": None}, {}, [], "m", log=logs.append) is None and "failed" in logs[0]
    rep = UC.safe_run(_md(), _stats(), _frames(), "m", log=logs.append)
    assert rep["needs_review"] == [] and rep["match_id"] == "m"


def test_repo_keys_load():
    """the SFK-BP and Vallentuna by-eye keys in reference/ are readable (V2 grades possession on them)"""
    assert len(UC.load_owner_key(UC.REPO, "SFKBP1109")) >= 50
    assert len(UC.load_owner_key(UC.REPO, "p15u-vs-vallentuna-2026-10-03-6cce")) >= 15


def test_export_writes_the_review(tmp_path):
    """export.write runs the checks and puts them in stats.json (st['review']) and summary.json"""
    import numpy as np
    from ipanema import export as EX
    n, fps = 40, 10.0
    per = [[(i, "A" if i < 7 else "B", (50.0, 30.0), (100.0, 100.0), None, i in (0, 1)) for i in range(14)] for _ in range(n)]   # 2 keepers in A
    fr = [{"carrier": None, "pressure_m": None, "near_opps": None} for _ in range(n)]
    H = {k: np.eye(3) for k in range(n)}
    EX.write(str(tmp_path), "m1", str(tmp_path / "none.mp4"), {"fps": fps, "width": 1920, "height": 1080}, per, fr, {}, {}, [2] * n, H, L, W,
             {"A": True, "B": False}, 0.5, [], [], [], [], {}, [{"A": None, "B": None}] * n, {"metrics": {"shots": [], "high_turnovers": []}, "teams": []}, None,
             {"ball_grade": None, "ball_reliable": False}, log=lambda *a: None, copy_video=False, make_zip=False)
    st = json.load(open(tmp_path / "matches" / "m1" / "stats.json")); sm = json.load(open(tmp_path / "matches" / "m1" / "summary.json"))
    assert st["review"]["needs_review"] == ["keepers"] and sm["review"]["needs_review"] == ["keepers"]
