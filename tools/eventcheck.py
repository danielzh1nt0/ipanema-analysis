"""3 Oct (Daniel: 'I saw a free kick when there was no free kick'): our exported events vs Veo's own minute-by-minute event
list for SFK-BP (reference/veo_events_SFKBP1109.txt; Veo minute m = video minute m, coverage minutes 1-31 and 44-51; our
first half is video 9:15-51:00). Per type: Veo count, our count, and how many of ours fall in a Veo minute with that type
(+-1 min, each Veo event used once) = 'confirmed'. Writes results/review/eventcheck_<date>.md.
    PYTHONPATH=. python tools/eventcheck.py [results/volume/runs/matches/SFKBP1109]"""
import sys, os, json, re, collections, time
D = sys.argv[1] if len(sys.argv) > 1 else "results/volume/runs/matches/SFKBP1109"
veo = []
for line in open("reference/veo_events_SFKBP1109.txt"):
    m = re.match(r"^(\d+)\s+(\S+)\s+(.+)$", line.strip())
    if m: veo.append((int(m.group(1)), m.group(2), m.group(3).strip().lower()))
COVER = set(range(1, 32)) | set(range(44, 52)); HALF = (555.0, 3060.0)
KIND = {"throw-in": "throw-in", "corner": "corner", "goal kick": "goal kick", "free kick": "free kick"}
st = json.load(open(f"{D}/stats.json")); md = json.load(open(f"{D}/match_data.json"))
ours = [(r["t"], r["kind"], r["team"]) for r in st["restarts"] if HALF[0] <= r["t"] <= HALF[1]]
ev = collections.Counter(e["type"] for e in md["events"] if HALF[0] <= e["t"] <= HALF[1])
lines = [f"# Events vs Veo, SFK-BP first half ({time.strftime('%Y-%m-%d')})", "", f"Export: {D}. Veo list covers video minutes 1-31 and 44-51 (29 of the 42 first-half minutes).", "",
         "| type | Veo (covered minutes) | ours (covered minutes) | ours confirmed by Veo (+-1 min) | ours in uncovered minutes |", "|---|---|---|---|---|"]
tot = {}
for kind in ("throw-in", "corner", "goal kick", "free kick"):
    v = [(m, t) for m, t, k in veo if k == kind and m in COVER and m >= 10]
    o = [(int(t // 60) + 1, team) for t, k, team in ours if k == kind]
    oc = [x for x in o if x[0] in COVER]; left = list(v); conf = 0
    for m, team in oc:
        j = next((i for i, (vm, vt) in enumerate(left) if abs(vm - m) <= 1), None)
        if j is not None: left.pop(j); conf += 1
    lines.append(f"| {kind} | {len(v)} | {len(oc)} | {conf} ({100 * conf // max(1, len(oc))}%) | {len(o) - len(oc)} |"); tot[kind] = (len(v), len(oc), conf)
veo_total = sum(x[0] for x in tot.values()); our_total = sum(x[1] for x in tot.values()); conf_total = sum(x[2] for x in tot.values())
lines += ["", f"All restarts: Veo {veo_total}, ours {our_total} in the covered minutes, {conf_total} of ours confirmed ({100 * conf_total // max(1, our_total)}%), {veo_total - conf_total} of Veo's missed.", "",
          "Veo events we never produce: foul, offside (Veo lists them).", "", "## Our event types in the first half (what the match section can list)", ""]
for t, c in ev.most_common(): lines.append(f"- {t}: {c}")
goals = [e for e in md["events"] if e["type"] == "goal" and HALF[0] <= e["t"] <= HALF[1]]; vg = [(m, t) for m, t, k in veo if k == "goal"]
lines += ["", f"Goals: ours {[(round(e['t'] / 60, 1), e['team']) for e in goals]} (from Veo's highlight list); Veo's event list goals {vg} (list has gaps)."]
os.makedirs("results/review", exist_ok=True); out = f"results/review/eventcheck_{time.strftime('%Y-%m-%d')}.md"; open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
