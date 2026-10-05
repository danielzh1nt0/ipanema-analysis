"""5 Oct: one check of every stat the app shows for the three demo matches, against what we know independently
(Veo's goal/shot clips, by-eye keys) plus plausibility rules. Reads results/free/app_files/<id> (what the app shows,
pulled free by tools/app_pull.py) when present, else results/volume/runs/matches/<id>.
    PYTHONPATH=. python tools/demo_audit.py  -> prints a table, writes results/qa/demo_audit.md"""
import os, sys, json, glob, bisect, math, collections, statistics as stt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MS = {"SFKBP1109": "SFK-BP (1st half)", "p15u-vs-aik-2026-09-21-bd09": "SFK-AIK (1st half)", "p15u-vs-vallentuna-2026-10-03-6cce": "SFK-Vallentuna (full)"}
EXPECT_SCORE = {"SFKBP1109": (1, 0), "p15u-vs-aik-2026-09-21-bd09": (0, 2), "p15u-vs-vallentuna-2026-10-03-6cce": (3, 2)}   # by eye (kick-offs after each goal)
rows = []

def src(m):
    a = f"results/free/app_files/{m}"
    return a if os.path.exists(f"{a}/stats.json") else f"results/volume/runs/matches/{m}"

def frames_of(m):
    fs = []
    for d in (f"results/free/app_files/{m}", f"results/volume/runs/matches/{m}"):
        got = sorted(glob.glob(f"{d}/frames_*.json"))
        if got:
            for fn in got: fs += json.load(open(fn))["frames"]
            return fs, d
    return fs, None

def veo(m):
    p = f"reference/veo_highlights_{m}.txt"; out = []
    if not os.path.exists(p): return out
    for line in open(p):
        line = line.split("#")[0].split()
        if len(line) >= 2: out.append((float(line[0]), line[1], line[2] if len(line) > 2 and line[2] in ("A", "B") else None))
    return out

def add(m, stat, value, verdict, why):
    rows.append((MS[m], stat, value, verdict, why))

for m in MS:
    D = src(m); st = json.load(open(f"{D}/stats.json")); md = json.load(open(f"{D}/match_data.json"))
    per = md.get("periods") or st.get("periods") or []
    inwin = lambda t: any(p["t_start"] - 5 <= t <= p["t_end"] + 5 for p in per)
    T = {t["team"]: t for t in st["teams"]}; A, B = T["A"], T["B"]
    # score
    goals = [s for s in st["metrics"]["shots"] if s.get("goal") and inwin(s["t"])]
    sc = (sum(g["team"] == "A" for g in goals), sum(g["team"] == "B" for g in goals))
    add(m, "Score", f"{sc[0]}-{sc[1]}", "OK" if sc == EXPECT_SCORE[m] else "WRONG", f"by eye {EXPECT_SCORE[m][0]}-{EXPECT_SCORE[m][1]}")
    # shots vs Veo
    shots = [s for s in st["metrics"]["shots"] if inwin(s["t"])]
    vs = sorted({round(t) for t, k, _ in veo(m) if inwin(t) and k in ("shot", "goal")})
    vs = [t for i, t in enumerate(vs) if not i or t - vs[i - 1] > 3]
    add(m, "Shots SFK / opp", f"{sum(s['team']=='A' for s in shots)} / {sum(s['team']=='B' for s in shots)}", "OK" if len(shots) == len(vs) else "CHECK", f"{len(vs)} Veo shot clips in the playing time")
    pos = [s for s in shots if s.get("x_m") is not None]
    add(m, "Shot map spots", f"{len(pos)} of {len(shots)}", "OK" if len(pos) == len(shots) else "PARTLY", "spots only where clicked by eye; others listed without a spot")
    # possession
    add(m, "Possession SFK", f"{A['possession_pct']}%", "", "")
    fs, fd = frames_of(m)
    if fs:
        ts = [f["t"] for f in fs]; n = sum(1 for f in fs if f["players"])
        cnt = collections.Counter(p["team"] for f in fs for p in f["players"])
        pa, pb = cnt["A"] / max(1, n), cnt["B"] / max(1, n)
        add(m, "Players per picture SFK / opp", f"{pa:.1f} / {pb:.1f}", "OK" if 0.6 <= pa / max(pb, 0.1) <= 1.6 else "WRONG", "both teams should be about equal on a follow-cam")
        gk = max((collections.Counter(p["team"] for p in f["players"] if p.get("gk")).most_common(1) or [(None, 0)])[0][1] for f in fs)
        add(m, "GK labels per team per picture (max)", gk, "OK" if gk <= 1 else "WRONG", "at most one keeper per team")
        bal = sum(1 for f in fs if f.get("ball")) / max(1, len(fs))
        add(m, "Pictures with a ball position", f"{100*bal:.0f}%", "OK" if bal > 0.6 else "CHECK", "")
        # possession vs by-eye owners
        key = []
        if os.path.exists(f"reference/{m}/owner_key.json"): key = [(i["t"], i["owner"], None) for i in json.load(open(f"reference/{m}/owner_key.json"))["items"] if i["owner"] in ("A", "B")]
        if os.path.exists(f"reference/{m}/ball_key_graded.json"): key = [(i["t"], i["owner"], i.get("ball_px")) for i in json.load(open(f"reference/{m}/ball_key_graded.json"))["items"] if i.get("owner") in ("A", "B")]
        if key:
            r = collections.Counter()
            for t, o, bp in key:
                i = min(bisect.bisect_left(ts, t), len(ts) - 1); f = fs[i]
                if abs(f["t"] - t) > 0.3: r["no picture"] += 1; continue
                pv = f.get("possession") or (f.get("ball") or {}).get("team")
                if pv in ("A", "B"): r["right" if pv == o else "wrong"] += 1
                else: r["no owner shown"] += 1
            add(m, "Who has the ball (by-eye moments)", f"{r['right']} right / {r['wrong']} wrong / {r['no owner shown']} none", "OK" if r["right"] >= 0.75 * max(1, r["right"] + r["wrong"]) else "WRONG", f"{len(key)} moments checked by eye")
    else: add(m, "Players per picture", "-", "NOT CHECKED", "frame files not here")
    # passes etc: plausibility for U15 (per 45 min about 150-350 per team, completion 60-85%)
    mins = sum(p["t_end"] - p["t_start"] for p in per) / 60
    for t, nm in ((A, "SFK"), (B, "opp")):
        pp45 = t["passes"] / max(1, mins) * 45
        add(m, f"Passes {nm}", f"{t['passes']} ({t['pass_completion_pct']}% completed)", "OK" if 80 <= pp45 <= 400 and 50 <= t["pass_completion_pct"] <= 92 else "CHECK", f"{pp45:.0f} per 45 min")
    add(m, "Losses / recoveries SFK", f"{A['losses']} / {A['recoveries']}", "OK" if A["losses"] and abs(A["losses"] - B["recoveries"]) <= 3 else "CHECK", "SFK losses should equal opponent recoveries")
    add(m, "Regained within 5 s SFK / opp", f"{A['regained_within_5s_pct']}% / {B['regained_within_5s_pct']}%", "", "")
    add(m, "Pressures applied SFK / opp", f"{A['pressures_applied']} / {B['pressures_applied']}", "", "")
    add(m, "Defensive line height SFK / opp", f"{A['def_line_height_median_m']} / {B['def_line_height_median_m']} m", "OK" if 15 <= A["def_line_height_median_m"] <= 65 else "CHECK", "metres from own goal")
    add(m, "Distance (in camera view) SFK", f"{A['distance_m_total_visible']/1000:.1f} km", "", "only players the camera sees")
    rk = collections.Counter(r["kind"] for r in st.get("restarts", []) if inwin(r["t"]))
    add(m, "Restarts", dict(rk), "", "")

w = max(len(r[1]) for r in rows); out = ["| Match | Stat | Value | Verdict | Why |", "|---|---|---|---|---|"]
for r in rows: out.append("| " + " | ".join(str(x) for x in r) + " |")
open("results/qa/demo_audit.md", "w").write("# Demo audit (5 Oct)\n\nSources: " + ", ".join(f"{m}: {src(m)}" for m in MS) + "\n\n" + "\n".join(out) + "\n")
for r in rows: print(f"{r[0]:22s} {r[1]:{w}s} {str(r[2]):40s} {r[3]:12s} {r[4]}")
