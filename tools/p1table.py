"""P1 (29 Sep): before/after table of the free Kaggle tracking runs for the striped-kit fix.
before = results/kaggle/players_all (P2 run, before P1); after = players_p1b (Spånga, Reymersholm, SFK, fix v2) +
players_p1c (the other 4 grounds, fix v2). Per 60-s piece: players per frame (dark/light), median time tracked,
track ids, rows thrown out as 'neither team' (referee/staff), kit reading chosen. Free, local: python tools/p1table.py"""
import os, re, json, glob, sys
K = "results/kaggle"
def piece(d, tag):
    s = f"{K}/{d}/{tag}_summary.json"
    if not os.path.exists(s): return None
    r = list(json.load(open(s)).values())[0]; log = open(f"{K}/{d}/{tag}_log.txt").read() if os.path.exists(f"{K}/{d}/{tag}_log.txt") else ""
    m = re.search(r"removed (\d+) rows of non-team", log); rd = re.search(r"reading (\w+/\w+)", log)
    return {"dark": r["observed_per_frame"]["dark"], "light": r["observed_per_frame"]["light"], "dark_s": r["median_track_s"]["dark"],
            "light_s": r["median_track_s"]["light"], "tracks": r["tracks"], "neither_rows": int(m.group(1)) if m else None,
            "reading": rd.group(1) if rd else ("default" if "kits:" in log else "?")}
def table(before="players_all", after=("players_p1b", "players_p1c")):
    tags = sorted({os.path.basename(p)[:-len("_summary.json")] for d in after for p in glob.glob(f"{K}/{d}/*_summary.json")})
    out = {}
    for t in tags:
        a = next((piece(d, t) for d in after if piece(d, t)), None); b = piece(before, t)
        out[t] = {"before": b, "after": a}
    return out
if __name__ == "__main__":
    T = table(); json.dump(T, open("results/qa/p1/before_after.json", "w"), indent=1)
    f = lambda x, k: "-" if x is None or x.get(k) is None else x[k]
    print("| piece | players/frame dark+light before -> after | median s tracked (dark/light) before -> after | 'neither' rows thrown out before -> after | kit reading |")
    print("|---|---|---|---|---|")
    for t, v in T.items():
        b, a = v["before"], v["after"]
        print(f"| {t} | {f(b,'dark')}+{f(b,'light')} -> {f(a,'dark')}+{f(a,'light')} | {f(b,'dark_s')}/{f(b,'light_s')} -> {f(a,'dark_s')}/{f(a,'light_s')} | {f(b,'neither_rows')} -> {f(a,'neither_rows')} | {f(a,'reading')} |")
