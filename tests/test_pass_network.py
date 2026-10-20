"""N0-20c (20 Oct): metrics.pass_network (in every export's stats.metrics, no test before)."""
import json, os, sys, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ipanema import metrics as M


def _p(t, a, b, team, done=True, fm=(10.0, 10.0), tm=(20.0, 10.0), kind="forward", prog=False):
    return {"t": t, "from": a, "to": b, "team": team, "completed": done, "from_m": list(fm), "to_m": list(tm), "kind": kind, "progressive": prog}


PLAYERS = [{"id": 1, "team": "A", "avg_x_m": 50.0, "avg_y_m": 30.0}, {"id": 2, "team": "A", "avg_x_m": 60.0, "avg_y_m": 30.0},
           {"id": 3, "team": "B", "avg_x_m": 80.0, "avg_y_m": 50.0}]


def test_edges_count_completed_only_per_team():
    ps = [_p(1, 1, 2, "A", prog=True), _p(2, 1, 2, "A", kind="sideways"), _p(3, 2, 1, "A", done=False), _p(4, 3, 4, "B")]
    net = M.pass_network(ps, PLAYERS, nodes="passes")
    e = net["A"]["edges"]
    assert len(e) == 1 and e[0] == {"from": 1, "to": 2, "count": 2, "forward": 1, "progressive": 1}
    assert [(x["from"], x["to"]) for x in net["B"]["edges"]] == [(3, 4)]


def test_node_is_mean_of_pass_and_reception_spots():
    ps = [_p(1, 1, 2, "A", fm=(10, 10), tm=(30, 20)), _p(2, 2, 1, "A", fm=(30, 30), tm=(20, 30))]
    nd = M.pass_network(ps, PLAYERS, nodes="passes")["A"]["nodes"]
    assert nd["1"] == {"x": 15.0, "y": 20.0, "n": 2}                          # passed at (10,10), received at (20,30)
    assert nd["2"] == {"x": 30.0, "y": 25.0, "n": 2}


def test_track_missing_from_players_still_gets_a_node():
    ps = [_p(1, 1, 99, "A", tm=(70, 40))]                                     # 99: too short for the players' list
    old = M.pass_network(ps, PLAYERS, nodes="avg")["A"]["nodes"]; new = M.pass_network(ps, PLAYERS, nodes="passes")["A"]["nodes"]
    assert "99" not in old and new["99"] == {"x": 70.0, "y": 40.0, "n": 1}


def test_track_listed_for_the_other_team_sits_where_it_passed_for_this_team():
    ps = [_p(1, 1, 3, "A", tm=(40, 12))]                                      # 3 is mostly B (ID slide); this A pass reached it at (40,12)
    old = M.pass_network(ps, PLAYERS, nodes="avg")["A"]["nodes"]; new = M.pass_network(ps, PLAYERS, nodes="passes")["A"]["nodes"]
    assert (old["3"]["x"], old["3"]["y"]) == (80.0, 50.0) and (new["3"]["x"], new["3"]["y"]) == (40.0, 12.0)


def test_pass_without_spots_or_team_does_not_crash():
    ps = [{**_p(1, 1, 2, "A"), "from_m": None, "to_m": None}, {**_p(2, 1, 2, None)}]
    nd = M.pass_network(ps, PLAYERS, nodes="passes")["A"]["nodes"]
    assert nd["1"] == {"x": 50.0, "y": 30.0, "n": 0}                          # falls back to the track average


def test_avg_mode_reproduces_the_repo_export():
    st = json.load(open(os.path.join(ROOT, "results/volume/runs/matches/SFKBP1109_s1200/stats.json")))
    assert M.pass_network(st["passes"], st["players"], nodes="avg") == st["metrics"]["pass_network"]
    new = M.pass_network(st["passes"], st["players"], nodes="passes")
    for tm in ("A", "B"):                                                    # every edge has both nodes now (old: 2 dangling in A)
        assert all(str(e["from"]) in new[tm]["nodes"] and str(e["to"]) in new[tm]["nodes"] for e in new[tm]["edges"])
        assert new[tm]["edges"] == st["metrics"]["pass_network"][tm]["edges"]


def test_default_and_env_switch():
    code = "import ipanema.metrics as M; print(M.NET_NODES)"
    run = lambda env: subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**os.environ, **env}, capture_output=True, text=True).stdout.strip()
    assert run({"IPANEMA_NET_NODES": "passes"}) == "passes" and run({"IPANEMA_NET_NODES": "avg"}) == "avg"
    env = {k: v for k, v in os.environ.items() if k != "IPANEMA_NET_NODES"}
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip() == "passes"
