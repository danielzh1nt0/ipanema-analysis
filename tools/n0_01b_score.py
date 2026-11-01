"""N0-01b: score the blind grades (results/qa/n0-01b/grades_blind.json) against the key (moments_key.json).
Team colours per match come from the reference players' graded colours (majority). For each pass: is the passer's
shirt the pass's team (stats.passes[].team) or the players-list team (stats.players[].team)?

    python tools/n0_01b_score.py  -> results/qa/n0-01b/score.json
"""
import json, collections


def colours(g, k):
    c = collections.defaultdict(collections.Counter)
    for i, v in k.items():
        one = v["one_is"]; two = "B" if one == "A" else "A"
        for team, x in ((one, g[i]["r1"]), (two, g[i]["r2"])):
            if x != "?": c[(v["match"], team)][x] += 1
    return {key: cnt.most_common(1)[0][0] for key, cnt in c.items()}


def score(g, k):
    col = colours(g, k); out = {"mismatch": collections.Counter(), "control": collections.Counter(), "rows": []}
    for i, v in sorted(k.items()):
        p = g[i]["P"]; grp = "mismatch" if v["player_team"] != v["pass_team"] else "control"
        if p in ("cant", "none"): res = p
        else:
            seen = next((t for t in "AB" if col.get((v["match"], t)) == p), None)
            res = ("both" if v["pass_team"] == v["player_team"] == seen else "pass team" if seen == v["pass_team"]
                   else "player-list team" if seen == v["player_team"] else "neither")
        out[grp][res] += 1; out["rows"].append({"id": i, "group": grp, "match": v["match"], "seen": p, "result": res})
    out["colours"] = {f"{m} {t}": c for (m, t), c in col.items()}
    return out


if __name__ == "__main__":
    g = json.load(open("results/qa/n0-01b/grades_blind.json")); k = json.load(open("results/qa/n0-01b/moments_key.json"))
    s = score(g, k); json.dump(s, open("results/qa/n0-01b/score.json", "w"), indent=1)
    print("colours", s["colours"]); print("mismatch", dict(s["mismatch"])); print("control", dict(s["control"]))
