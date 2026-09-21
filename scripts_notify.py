"""Write the title and body of the end-of-run issue (GitHub emails the repo owner when it opens)."""
import glob, os, re, subprocess
msg = os.environ.get("MSG", ""); url = os.environ.get("RUN_URL", "")
live = sorted(glob.glob("results/live/*.log"), key=os.path.getmtime)
status_line = open(live[-1]).readline().strip("# \n") if live else "no live log"
changed = subprocess.run("git diff --name-only HEAD~1 HEAD -- results/scoreboard.md", shell=True, capture_output=True, text=True).stdout.strip()
row = open("results/scoreboard.md").read().strip().splitlines()[-1] if changed and os.path.exists("results/scoreboard.md") else ""
m = re.search(r"\| (\d+/\d+) \| (\d+) \|", row)
what = "stopped automatically" if "STOPPED AUTOMATICALLY" in status_line else ("failed" if "failed" in status_line else "finished")
tag = re.search(r"\[(run|full|check|prepare|export|inventory)[:\]]([A-Za-z0-9_-]*)", msg)
name = (tag.group(2) or "run") if tag else "run"
title = f"Ipanema {what}: {name}" + (f" · ball {m.group(1)} (ceiling {m.group(2)})" if m else "")
body = f"**{status_line}**\n\n" + (f"Scoreboard line:\n\n{row}\n\n" if row else "") + f"Commit: {msg[:200]}\n\nRun: {url}\n"
open("/tmp/issue_title", "w").write(title[:200]); open("/tmp/issue_body.md", "w").write(body)
print(title)
