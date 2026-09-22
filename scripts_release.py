"""Append an entry to RELEASES.md whenever the pipeline version changes: what changed, and the numbers measured then.

Run by the workflow after every push: it compares ipanema/__init__.py's version with the last entry in RELEASES.md.
"""
import re, os, subprocess, datetime, sys

def version():
    return re.search(r'__version__ = "([^"]+)"', open("ipanema/__init__.py").read()).group(1)

def last_logged():
    if not os.path.exists("RELEASES.md"): return None
    m = re.findall(r"^## v([0-9.]+)", open("RELEASES.md").read(), re.M)
    return m[0] if m else None

def scoreboard_tail(n=2):
    if not os.path.exists("results/scoreboard.md"): return []
    rows = [l for l in open("results/scoreboard.md").read().splitlines() if l.startswith("| 20")]
    return rows[-n:]

def entry(v, msg, rows):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = [f"## v{v} — {now}", "", msg.strip() or "(no message)", ""]
    if rows: out += ["Measured around this version:", "", "| when | commit | match | version | ball check | ceiling | held-out | possession | events | players/frame | notes |", "|---|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    return "\n".join(out) + "\n"

def main():
    v = version()
    if v == last_logged(): print(f"v{v} already in RELEASES.md"); return
    msg = subprocess.run(["git", "log", "-1", "--pretty=%s%n%n%b"], capture_output=True, text=True).stdout
    head = "# Releases\n\nOne entry per pipeline version: what changed and what was measured at the time.\n\n"
    body = open("RELEASES.md").read().split("\n", 3)[-1] if os.path.exists("RELEASES.md") else ""
    open("RELEASES.md", "w").write(head + entry(v, msg, scoreboard_tail()) + body)
    print(f"RELEASES.md: added v{v}")

if __name__ == "__main__": main()
