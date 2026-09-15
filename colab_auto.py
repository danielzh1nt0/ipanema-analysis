"""AUTO mode: watch GitHub, re-run + score on every new commit, push the scoreboard back to the repo (results/).
Run after Cell 0. Needs Colab secret GITHUB_TOKEN (fine-grained, Contents: read+write on this repo). Stop with the stop button."""
import os, sys, json, time, subprocess, datetime
from google.colab import userdata
ROOT = os.environ.get("IPANEMA_ROOT", "/content/drive/MyDrive/match_analysis/match_analysis")
CODE = "/content/ipanema-analysis"; REPO = "danielzh1nt0/ipanema-analysis"; TOKEN = userdata.get("GITHUB_TOKEN")
AUTH = f"https://x-access-token:{TOKEN}@github.com/{REPO}.git"
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)
sh(f"cd {CODE} && git remote set-url origin {AUTH} && git config user.email colab@ipanema && git config user.name 'ipanema colab'")
def remote_sha(): return sh("git ls-remote https://github.com/" + REPO + ".git refs/heads/main").stdout.split()[0]
def local_sha(): return sh(f"cd {CODE} && git rev-parse HEAD").stdout.strip()
def reload():
    for m in [m for m in sys.modules if m == "ipanema" or m.startswith("ipanema.")]: del sys.modules[m]
    import ipanema; return ipanema
def run_all():
    ipanema = reload()
    from ipanema.run import run
    from ipanema.config import Settings
    from ipanema import check
    S = Settings(root=ROOT); logs = []
    def log(s=""): print(s); logs.append(str(s))
    for clip in sorted(f for f in os.listdir(f"{ROOT}/videos") if f.lower().endswith((".mp4", ".mov", ".mkv"))):
        try: run(f"{ROOT}/videos/{clip}", settings=S, log=log)
        except Exception as e: log(f"ERROR {clip}: {e!r}")
    board = check.check_all(ROOT, log=log)
    return ipanema.__version__, board, logs
def publish(sha, version, board, logs):
    os.makedirs(f"{CODE}/results", exist_ok=True); ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    json.dump({"sha": sha, "version": version, "time": ts, "scoreboard": board}, open(f"{CODE}/results/scoreboard_latest.json", "w"), indent=1, default=str)
    json.dump({"sha": sha, "version": version, "time": ts, "scoreboard": board}, open(f"{CODE}/results/scoreboard_{ts}_{sha[:7]}.json", "w"), indent=1, default=str)
    open(f"{CODE}/results/log_latest.txt", "w").write("\n".join(logs[-400:]))
    from ipanema import check as _chk
    for clip in sorted(os.listdir(f"{ROOT}/runs/matches")):
        try: open(f"{CODE}/results/turnovers_{clip}.txt", "w").write(_chk.turnover_table(ROOT, clip))
        except Exception as e: print("table failed", clip, repr(e))
    sh(f"cd {CODE} && git add results && git commit -qm 'results for {sha[:7]} (v{version})'")
    for attempt in range(3):
        r = sh(f"cd {CODE} && git push -q origin HEAD:main")
        if r.returncode == 0: print("published"); return
        sh(f"cd {CODE} && git pull -q --rebase -X theirs origin main")      # the branch moved (a new commit landed mid-run): rebase our results on top and retry
    print(f"publish failed: {r.stderr[-300:]}")
seen = None
print("AUTO: watching", REPO, "— leave this cell running")
while True:
    try:
        sha = remote_sha()
        if sha != seen:
            print(f"\n=== new commit {sha[:7]} at {datetime.datetime.now():%H:%M} — pulling and running ===")
            sh(f"cd {CODE} && git fetch -q origin && git reset -q --hard origin/main")
            version, board, logs = run_all(); publish(sha, version, board, logs); seen = sha      # only the commit we actually ran
            print("=== done; waiting for the next commit ===")
        time.sleep(60)
    except KeyboardInterrupt: break
    except Exception as e: print("AUTO error:", repr(e)); time.sleep(120)
