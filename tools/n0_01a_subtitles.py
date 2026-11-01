"""N0-01a (1 Nov): event subtitles on the repo's app match copies, old text vs the new plain-words text (ipanema/export.py
lost_subtitle / won_subtitle / pass_subtitle / sequence_subtitle), rebuilt from each event's own payload. Counts the
texts that show Python's None, 'nan' or a bare dash for a missing value, and writes before/after examples.

    python tools/n0_01a_subtitles.py [--out results/qa/n0-01a/subtitles.json]
"""
import sys, os, json, glob, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import export as EX

BAD = re.compile(r"None|nan\b|inf\b|— s|— within|— m|·\s*—\s*$")


def new_text(e):
    p = e.get("payload") or {}; ty = e["type"]
    if ty == "turnover_lost": return e["title"], EX.lost_subtitle(p)
    if ty == "turnover_won": return e["title"], EX.won_subtitle(p)
    if ty == "sequence_end": return e["title"], EX.sequence_subtitle(p)
    if ty in ("pass_risky", "pass_bad"): return e["title"], EX.pass_subtitle(p)
    if ty == "set_piece": return (f"{p['kind']} · {p['team']}" if p.get("team") else p.get("kind", e["title"])), e["subtitle"]
    return e["title"], e["subtitle"]


def main(out):
    res = {}
    for pth in sorted(glob.glob("results/volume/runs/matches/*/match_data.json")):
        mid = pth.split("/")[-2]; d = json.load(open(pth))
        r = {"events": len(d["events"]), "by_type": {}}; ex = []
        for e in d["events"]:
            old = f"{e['title']} | {e['subtitle']}"; t2, s2 = new_text(e); new = f"{t2} | {s2}"
            b = r["by_type"].setdefault(e["type"], {"n": 0, "old_bad": 0, "new_bad": 0, "changed": 0})
            b["n"] += 1; b["old_bad"] += bool(BAD.search(old)); b["new_bad"] += bool(BAD.search(new)); b["changed"] += old != new
            if old != new and sum(1 for x in ex if x["type"] == e["type"]) < 3: ex.append({"type": e["type"], "t": e["t"], "old": old, "new": new})
        r["old_bad"] = sum(b["old_bad"] for b in r["by_type"].values()); r["new_bad"] = sum(b["new_bad"] for b in r["by_type"].values())
        r["examples"] = ex; res[mid] = r
        print(mid, "events", r["events"], "old bad", r["old_bad"], "new bad", r["new_bad"])
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/qa/n0-01a/subtitles.json")
