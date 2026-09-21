"""Append one line per finished run to results/scoreboard.md (called by the workflow after each run)."""
import json, os, re, sys, datetime
path, sha, match = sys.argv[1], sys.argv[2][:7], sys.argv[3]
txt = open(path).read()
i = txt.rfind("SUMMARY {"); summ = json.loads(txt[i + len("SUMMARY "):]) if i >= 0 else {}
bc = summ.get("ball_check") or {}; g = summ.get("ball_grade") or {}
ver = re.search(r"\(ipanema ([0-9.]+)", txt); cands = re.search(r"ball: ([A-Za-z +-]+candidates[^\n]*)", txt)
held = re.findall(r"kept best seed, held-out (\d+)", txt)
row = (f"| {datetime.datetime.utcnow():%Y-%m-%d %H:%M} | {sha} | {match} | {ver.group(1) if ver else '?'} | "
       f"{bc.get('correct', '—')}/{bc.get('total', '—')} | {bc.get('ceiling', '—')} | {held[-1] if held else '—'} | "
       f"{'OK' if g.get('possession_ok') else 'withheld'} | {'OK' if g.get('events_ok') else 'withheld'} | "
       f"{summ.get('players_per_frame_median', '—')} | {(cands.group(1) if cands else '').strip()[:60]} |")
sb = "results/scoreboard.md"
if not os.path.exists(sb):
    open(sb, "w").write("# Scoreboard — one line per run\n\nBall check = correct picks on held-out labelled frames; ceiling = frames where the ball was among the candidates.\n\n"
                        "| when (UTC) | commit | match | version | ball check | ceiling | held-out (train) | possession | events | players/frame | candidates |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
open(sb, "a").write(row + "\n"); print(row)
