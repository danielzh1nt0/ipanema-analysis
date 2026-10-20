"""N0-20d (20 Oct): team votes on a tie were decided by Python's set order, which changes from run to run (salted string
hashes). Now: tie -> prefer (if given and tied), else the tied label read last. IPANEMA_VOTE_TIE=hash = old."""
import os, sys, random, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema.vote import majority, team_vote
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _old_team(v, k_share=0.6):
    ab = [x for x in v if x in ("A", "B")]
    return "K" if (v.count("K") >= k_share * len(v) or not ab) and "K" in v else (max(set(ab), key=ab.count) if ab else max(set(v), key=v.count))


def test_majority_basics():
    assert majority([]) is None
    assert majority(["A", "B", "A"]) == "A"
    assert majority(["A", "B"]) == "B" and majority(["B", "A"]) == "A"        # tie -> the one read last
    assert majority(["A", "B", "B", "A"]) == "A"
    assert majority(["A", "B"], prefer="A") == "A"                           # tie -> prefer
    assert majority(["A", "A", "B"], prefer="B") == "A"                      # prefer only breaks ties
    assert majority(["A", "B", "K", "K", "A", "B"], prefer="R") == "B"       # prefer not among the tied -> latest tied


def test_team_vote_same_as_old_rule_without_a_tie():
    rng = random.Random(20)
    for _ in range(5000):
        v = [rng.choice("AABBKR") for _ in range(rng.randint(1, 25))]
        ab = [x for x in v if x in "AB"]
        tie = ab and ab.count("A") == ab.count("B") and not ((v.count("K") >= 0.6 * len(v)) and "K" in v)
        if not tie and not (not ab and len(set(v)) > 1 and "K" not in v):
            assert team_vote(v) == _old_team(v), v
        else:
            assert team_vote(v) in ("A", "B", "K", "R")


def test_team_vote_old_cases_and_ties():
    assert team_vote(["B"] * 6 + ["K"] * 4) == "B" and team_vote(["K"] * 9 + ["A"]) == "K"
    assert team_vote(["A", "K", "B"]) == "B"                                # tie -> latest reading
    assert team_vote(["A", "K", "B"], prev="A") == "A"                       # prev only when given
    assert team_vote([]) is None


def test_env_hash_restores_old(monkeypatch):
    monkeypatch.setenv("IPANEMA_VOTE_TIE", "hash")
    v = ["A", "B", "B", "A"]
    assert majority(v) == max(set(v), key=v.count)


def _run(code, seed, env_extra=None):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=ROOT); env.pop("IPANEMA_VOTE_TIE", None); env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, check=True).stdout.strip()


def test_same_answer_in_every_run():
    """the bug itself: the old rule's tie answer changes with the process's hash seed, the new one does not"""
    code = ("from ipanema.vote import team_vote, majority; from ipanema import fullmatch as FM\n"
            "s=[(6*i,'A') for i in range(9)]+[(54+6*i,'B') for i in range(9)]+[(108,'A'),(114,'B')]\n"
            "print(team_vote(['A','B','A','B']), team_vote(['A','K','B','B','A']), majority(['B','A']), FM.team_segments(s, 25.0, 4))")
    new = {_run(code, s) for s in range(12)}
    old = {_run(code, s, {"IPANEMA_VOTE_TIE": "hash"}) for s in range(12)}
    assert len(new) == 1, new
    assert len(old) > 1, old                                                # 12 seeds: the old rule gives more than one answer


def test_app_code_uses_the_vote():
    for f, line in (("ipanema/tracking.py", "team = team_vote(v, K_SHARE)"), ("ipanema/tracking.py", "kit = majority(team_of[r[0]])"),
                    ("ipanema/analytics.py", "team_id = {t: majority(team_of[t]) for t in ids}"),
                    ("ipanema/fullmatch.py", "majority(labs_[max(0, i - 2):i + 3], prefer=labs_[i])")):
        assert line in open(os.path.join(ROOT, f)).read(), (f, line)
        assert "max(set(" not in open(os.path.join(ROOT, f)).read(), f


def test_replay_tool_new_is_one_of_the_old_answers():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import n0_20d_ties as T
    R = {str(k): [[1, "A", [0, 0], False], [2, "B", [5, 5], False]] for k in range(6)}
    Tm = {"0": ["A", "B"], "1": ["B", "B"], "2": [None, "A"], "3": ["A", None], "4": ["B", "A"], "5": [None, None]}
    c = T.replay(R, Tm)
    assert c["rows"] == 12 and c["differ_rows"] == 0 and c["tie_rows"] >= 2 and c["tracks"] == 2
