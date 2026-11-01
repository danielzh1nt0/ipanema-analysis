"""N0-01d (1 Nov): what the spot clamp (ipanema/export.clamp_all) changes on the repo's app match copies: spots moved per
kind, and the places check (tools/n0_01d_places.py) before / after. Copies are read only, nothing is written back.

    python tools/n0_01d_clamp.py [--out results/qa/n0-01d/clamp.json]
"""
import sys, os, json, glob, copy, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema.export import clamp_all
import n0_01d_places as P


def main(out):
    res = {}
    for pth in sorted(glob.glob("results/volume/runs/matches/*/stats.json")):
        mid = pth.split("/")[-2]; st = json.load(open(pth)); L, W = st["pitch"]["length"], st["pitch"]["width"]
        before = collections.Counter(x["check"] for x in P.check(st))
        new = copy.deepcopy(st); moved = clamp_all(new.get("restarts"), new.get("metrics"), L, W, new.get("attack_right") or {"A": True, "B": False}, on=True)
        after = collections.Counter(x["check"] for x in P.check(new))
        n = {k: len(st.get(k) or []) if k == "restarts" else len((st.get("metrics") or {}).get(k) or []) for k in moved}
        res[mid] = {"moved": moved, "of": n, "broken_before": dict(before), "broken_after": dict(after)}
        print(mid, "moved", moved, "of", n, "| broken", sum(before.values()), "->", sum(after.values()), dict(after))
    json.dump(res, open(out, "w"), indent=1); return res


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/qa/n0-01d/clamp.json")
