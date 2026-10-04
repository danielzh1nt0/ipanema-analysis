"""4 Oct: check an exported match (results/volume/runs/matches/<id>) against the app rules in results/app/ui_contract_2026-10-02.md
and the Lovable prompts (events tiers, pressed, Veo window, periods). Prints PASS/FAIL lines.   PYTHONPATH=. python tools/contract_check.py <id>"""
import sys, json, glob, collections
D = f"results/volume/runs/matches/{sys.argv[1]}"
md = json.load(open(f"{D}/match_data.json")); st = json.load(open(f"{D}/stats.json")); ok = True
def chk(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (f": {detail}" if detail else ""))
per = md.get("periods") or []; chk("periods present", per, per)
lo, hi = (min(p["t_start"] for p in per) - 5, max(p["t_end"] for p in per) + 5) if per else (0, 1e9)
ev = md["events"]; tiers = collections.Counter(e.get("tier") for e in ev); chk("every event has a tier", None not in tiers, dict(tiers))
chk("tiers only verified/beta/hidden", set(tiers) <= {"verified", "beta", "hidden"}, set(tiers))
chk("verified = goal/shot only", {e["type"] for e in ev if e.get("tier") == "verified"} <= {"goal", "shot"}, {e["type"] for e in ev if e.get("tier") == "verified"})
chk("beta = turnovers only", {e["type"] for e in ev if e.get("tier") == "beta"} <= {"turnover_lost", "turnover_won"}, {e["type"] for e in ev if e.get("tier") == "beta"})
chk("verified flag matches tier", all((e.get("verified") is True) == (e.get("tier") == "verified") for e in ev))
gs = [e for e in ev if e["type"] in ("goal", "shot")]; out = [e["t"] for e in gs if not (lo <= e["t"] <= hi)]
chk("goals/shots inside the periods", not out, out); chk("shots list inside the periods", all(lo <= s["t"] <= hi for s in st["metrics"]["shots"]), [s["t"] for s in st["metrics"]["shots"] if not (lo <= s["t"] <= hi)])
chk("goals carry a team", all(e.get("team") in ("A", "B") for e in gs), [e["t"] for e in gs if e.get("team") not in ("A", "B")])
chk("goal count: events == metrics", sum(1 for e in ev if e["type"] == "goal") == sum(1 for s in st["metrics"]["shots"] if s["goal"]))
sp = [e for e in ev if e["type"] == "set_piece"]; kinds = collections.Counter(e["payload"].get("kind") for e in sp); chk("no 'free kick' set pieces (stoppage instead)", "free kick" not in kinds, dict(kinds))
tl = [e for e in ev if e["type"] == "turnover_lost"]; keys = set().union(*(e["payload"].keys() for e in tl)) if tl else set()
chk("turnover_lost payload has the new fields", {"pressed_within_2s", "near_at_2s", "time_to_press", "t_won"} <= keys, sorted(keys))
tw = [e for e in ev if e["type"] == "turnover_won"]; kw = set().union(*(e["payload"].keys() for e in tw)) if tw else set()
chk("turnover_won payload has forward_within_3s / lost_back_5s", {"forward_within_3s", "lost_back_5s"} <= kw, sorted(kw))
chunks = sorted(glob.glob(f"{D}/frames_*.json")); chk("a frame chunk fetched", chunks, len(chunks))
if chunks:
    fr = json.load(open(chunks[0]))["frames"]; f0 = next((f for f in fr if f.get("carrier")), fr[0])
    chk("frames carry 'pressed'", "pressed" in f0, sorted(f0)[:14])
    p2 = [f for f in fr if f.get("pressure_m") is not None]; chk("pressed == pressure_m <= 2", all(f["pressed"] == (f["pressure_m"] <= 2.0) for f in p2) if "pressed" in f0 else False, f"{len(p2)} frames with a carrier")
    chk("player state observed/filled", all(p.get("state") in ("observed", "filled") for f in fr for p in f["players"]), collections.Counter(p.get("state") for f in fr for p in f["players"]))
    chk("ball state observed/bridged/null", all((f["ball"] is None) or f["ball"].get("state") in ("observed", "bridged") for f in fr))
tm = {t["team"]: t for t in st["teams"]}; chk("team rows A and B", set(tm) == {"A", "B"}, list(tm))
chk("possession adds to 100", abs(tm["A"]["possession_pct"] + tm["B"]["possession_pct"] - 100) <= 1, (tm["A"]["possession_pct"], tm["B"]["possession_pct"]))
chk("losses(A) == recoveries(B) and vice versa", tm["A"]["losses"] == tm["B"]["recoveries"] and tm["B"]["losses"] == tm["A"]["recoveries"], {k: (v["losses"], v["recoveries"]) for k, v in tm.items()})
chk("better_option_count on teams", all("better_option_count" in v for v in tm.values()))
print("ALL PASS" if ok else "SOME FAILED")
