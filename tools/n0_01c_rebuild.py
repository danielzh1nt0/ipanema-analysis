"""N0-01c (1 Nov): rebuild the pass network of the repo's app copies (stats.json metrics.pass_network) without passes from
a player to himself (metrics.NET_SELF). Checks first that the old rule reproduces the stored network exactly.

    PYTHONPATH=. python tools/n0_01c_rebuild.py [--dry]  -> results/qa/n0-01c/rebuild.json
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import metrics as MT


def rebuild(st):
    old_stored = st["metrics"]["pass_network"]
    # the copies were exported before nodes became pass spots (they have no 'n'): rebuild in the copy's own node mode
    mode = "passes" if any("n" in nd for v in old_stored.values() for nd in v["nodes"].values()) else "avg"
    MT.NET_SELF = True; old = json.loads(json.dumps(MT.pass_network(st["passes"], st["players"], nodes=mode)))
    MT.NET_SELF = False; new = json.loads(json.dumps(MT.pass_network(st["passes"], st["players"], nodes=mode)))
    ok = old == old_stored
    loops = {tm: sum(e["from"] == e["to"] for e in v["edges"]) for tm, v in old_stored.items()}
    if ok: st["metrics"]["pass_network"] = new
    return ok, loops


if __name__ == "__main__":
    res = {}
    for pth in sorted(glob.glob("results/volume/runs/matches/*/stats.json")):
        st = json.load(open(pth)); mid = pth.split("/")[-2]
        if not (st.get("metrics") or {}).get("pass_network"): res[mid] = "no network"; print(mid, "no network"); continue
        ok, loops = rebuild(st)
        res[mid] = {"old_rule_reproduces": ok, "self_loops_removed": loops, "edges_now": {tm: len(v["edges"]) for tm, v in st["metrics"]["pass_network"].items()}}
        print(mid, res[mid])
        if "--dry" not in sys.argv and ok: json.dump(st, open(pth, "w"))
    os.makedirs("results/qa/n0-01c", exist_ok=True)
    json.dump(res, open("results/qa/n0-01c/rebuild.json", "w"), indent=1)
