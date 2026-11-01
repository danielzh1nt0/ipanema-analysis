"""N0-01b (1 Nov): rebuild the per-player pass fields of the repo's app copies (stats.json players[]) from their own
passes list with the new rule (a pass counts on a player's card only when it is his team's pass; analytics.PASS_OWN_TEAM).
Checks first that the old rule reproduces the stored numbers exactly, so only the rule changes.

    python tools/n0_01b_rebuild.py [--dry]  -> results/qa/n0-01b/rebuild.json
"""
import sys, json, glob

Q = ("good", "risky_completed", "bad_lost", "execution_error")


def fields(passes, tid, tm, own):
    mp = [p for p in passes if p["from"] == tid and (not own or p["team"] == tm)]
    rc = [p for p in passes if p.get("to") == tid and p["completed"] and (not own or p["team"] == tm)]
    return {"passes": len(mp), "passes_completed": sum(bool(p["completed"]) for p in mp),
            "passes_forward": sum(p["kind"] == "forward" for p in mp), "passes_sideways": sum(p["kind"] == "sideways" for p in mp),
            "passes_backward": sum(p["kind"] == "backward" for p in mp), "progressive_passes": sum(bool(p["progressive"]) for p in mp),
            "passes_received": len(rc), "pass_quality": {q: sum(p.get("quality") == q for p in mp) for q in Q},
            "better_option_count": sum(bool(p.get("better_option")) for p in mp), "passes_risky": sum(p.get("quality") == "risky_completed" for p in mp)}


def rebuild(st):
    """-> (old_ok, changed players, passes moved off cards). Edits st in place."""
    P = st["passes"]; ok = True; changed = 0; moved = 0
    for pl in st["players"]:
        old = fields(P, pl["id"], pl["team"], False)
        for k, v in old.items():
            if k in ("better_option_count", "passes_risky") and pl.get(k) is None: continue   # withheld / older copy
            if pl.get(k) != v: ok = False
        new = fields(P, pl["id"], pl["team"], True)
        for k in ("better_option_count", "passes_risky"):
            if pl.get(k) is None: new[k] = None
        new = {k: v for k, v in new.items() if k in pl}                        # older copies lack some fields
        if any(pl.get(k) != v for k, v in new.items()): changed += 1
        moved += pl["passes"] - new["passes"]; pl.update(new)
    return ok, changed, moved


if __name__ == "__main__":
    res = {}
    for pth in sorted(glob.glob("results/volume/runs/matches/*/stats.json")):
        st = json.load(open(pth)); ok, ch, mv = rebuild(st); mid = pth.split("/")[-2]
        res[mid] = {"old_rule_reproduces": ok, "players_changed": ch, "passes_off_cards": mv, "passes": len(st["passes"])}
        print(mid, res[mid])
        if "--dry" not in sys.argv and ok: json.dump(st, open(pth, "w"))
    json.dump(res, open("results/qa/n0-01b/rebuild.json", "w"), indent=1)
