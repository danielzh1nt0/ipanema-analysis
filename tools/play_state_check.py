"""N0-21d: possession in the app copies of the full matches, split by match time / outside it (frames files, every 3rd frame).
Run from the repo root: python tools/play_state_check.py"""
import json, glob, collections, sys
for m in ["SFKBP1109", "p15u-vs-aik-2026-09-21-bd09", "p15u-vs-vallentuna-2026-10-03-6cce"]:
    d = f"results/volume/runs/matches/{m}/"
    st = json.load(open(d + "stats.json")); per = [(p["t_start"], p["t_end"]) for p in st["periods"]]
    app = {t["team"]: t["possession_pct"] for t in st["teams"]}
    c = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(d + "frames_*.json")):
        for fr in json.load(open(f))["frames"]:
            inside = any(a <= fr["t"] <= b for a, b in per)
            empty = not fr["players"] and fr["ball"] is None
            w = "out of periods" if not inside else ("in, blanked" if empty else "in, play")
            c[w][fr["possession"] or "none"] += 1
    def pct(cc): a, b = cc["A"], cc["B"]; return f"A {round(100*a/max(1,a+b))} / B {round(100*b/max(1,a+b))}"
    allc = sum(c.values(), collections.Counter())
    print(m, "periods", per, "| app", app, "| all frames", pct(allc), "| play only", pct(c["in, play"]))
    for w, cc in c.items(): print("   ", w, dict(cc))
