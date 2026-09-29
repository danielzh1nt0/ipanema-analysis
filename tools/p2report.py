"""P2 (29 Sep): table of the Kaggle player-tracking pieces (kaggle/players_all.py, players_p1.py) per match:
players seen per frame by team, median time a player stays tracked, track ids per minute, frames with too many players.
python tools/p2report.py results/kaggle/players_all [results/kaggle/players_p1]  -> prints markdown + <dir>/p2_table.json"""
import os, sys, json, gzip, glob, re, numpy as np
def piece_stats(d, tag):
    s = json.load(open(f"{d}/{tag}_summary.json")); r = s.get("new, RF-DETR")
    if not r: return None
    out = {"dark": r["observed_per_frame"]["dark"], "light": r["observed_per_frame"]["light"], "dark_s": r["median_track_s"]["dark"],
           "light_s": r["median_track_s"]["light"], "tracks": r["tracks"], "minutes": r.get("minutes")}
    p = f"{d}/{tag}_rows_all.json.gz"
    if os.path.exists(p):
        A = json.load(gzip.open(p, "rt")); rows = A["rows"]["new, RF-DETR"]; fps = A["fps"]
        per = [(sum(1 for x in v if x[1] == "A" and not x[3]), sum(1 for x in v if x[1] != "A" and not x[3])) for v in rows.values()]
        tot = np.array([a + b for a, b in per])
        out.update(frames=len(per), too_many=round(float(((np.array([a for a, _ in per]) > 11) | (np.array([b for _, b in per]) > 11)).mean()), 3),
                   total_p10_p90=[int(np.percentile(tot, 10)), int(np.percentile(tot, 90))], seconds=round(len(per) / fps, 1))
        out["tracks_per_min"] = round(out["tracks"] / max(1e-6, len(per) / fps / 60), 0)
    log = f"{d}/{tag}_log.txt"
    if os.path.exists(log):
        m = re.search(r"reading (\S+)/(\S+)", open(log).read()); out["kit reading"] = f"{m.group(1)}/{m.group(2)}" if m else "default"
    return out
res = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/*_summary.json")):
        tag = os.path.basename(f)[:-len("_summary.json")]; m, s = tag.rsplit("_", 1)
        st = piece_stats(d, tag)
        if st: res.setdefault(os.path.basename(d), {}).setdefault(m, {})[int(s)] = st
    json.dump(res.get(os.path.basename(d), {}), open(f"{d}/p2_table.json", "w"), indent=1)
for run, ms in res.items():
    print(f"\n### {run}\n\n| match | start (s) | dark / light seen per frame | median s tracked dark / light | track ids per min | frames with >11 of one team | kit reading |\n|---|---|---|---|---|---|---|")
    for m, sp in ms.items():
        for s, r in sorted(sp.items()):
            print(f"| {m} | {s} | {r['dark']:.0f} / {r['light']:.0f} | {r['dark_s']} / {r['light_s']} | {r.get('tracks_per_min', '-')} | {100 * r.get('too_many', 0):.0f}% | {r.get('kit reading', '-')} |")
