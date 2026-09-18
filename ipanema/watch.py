"""One watcher iteration. Reloaded from GitHub every loop, so fixes here never need a Colab restart."""
import os, sys, json, subprocess, datetime

def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)

def step(state, ROOT, CODE, REPO):
    sha = sh(f"git ls-remote https://github.com/{REPO}.git refs/heads/main").stdout.split()[0]
    if sha == state.get("seen"): return
    sh(f"cd {CODE} && git fetch -q origin && git reset -q --hard origin/main")
    # run when the newest *code* commit differs from the one the latest results were produced for
    code_sha = sh(f"cd {CODE} && git log --format=%H --invert-grep --grep='^results for' -1").stdout.strip()
    last_run = sh(f"cd {CODE} && git log --format=%s --grep='^results for' -1").stdout.strip()
    if last_run and code_sha[:7] in last_run and state.get("first_done"): state["seen"] = sha; return
    msg = sh(f"cd {CODE} && git log -1 --format=%s {code_sha}").stdout.strip(); state["first_done"] = True; sha = code_sha
    print(f"\n=== new commit {sha[:7]} at {datetime.datetime.now():%H:%M} — {msg} ===")
    for m in [m for m in sys.modules if m == "ipanema" or m.startswith("ipanema.")]: del sys.modules[m]
    import ipanema
    from ipanema.run import run
    from ipanema.config import Settings
    from ipanema import check
    S = Settings(root=ROOT); logs = []
    def log(s=""): print(s); logs.append(str(s))
    try:
        from ipanema.segments import prepare_segments; prepare_segments(ROOT, log=log)
    except Exception as e: log(f"segments: {e!r}")
    SKIP = {"SFKBP1109_seg1.mp4", "SFKBP1109_s3082.mp4", "08fd33_0.mp4", "0bfacc_0.mp4"}        # superseded segments (half-time / wrong calibration)
    for clip in sorted(f for f in os.listdir(f"{ROOT}/videos") if f.lower().endswith((".mp4", ".mov", ".mkv")) and f not in SKIP):
        try: run(f"{ROOT}/videos/{clip}", settings=S, log=log)
        except Exception as e: log(f"ERROR {clip}: {e!r}")
    board = check.check_all(ROOT, log=log)
    # publish
    os.makedirs(f"{CODE}/results", exist_ok=True); ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    payload = {"sha": sha, "version": ipanema.__version__, "time": ts, "scoreboard": board}
    json.dump(payload, open(f"{CODE}/results/scoreboard_latest.json", "w"), indent=1, default=str)
    json.dump(payload, open(f"{CODE}/results/scoreboard_{ts}_{sha[:7]}.json", "w"), indent=1, default=str)
    open(f"{CODE}/results/log_latest.txt", "w").write("\n".join(logs[-600:]))
    sh(f"cd {CODE} && git add results && git commit -qm 'results for {sha[:7]} (v{ipanema.__version__})'")
    for attempt in range(3):
        r = sh(f"cd {CODE} && git push -q origin HEAD:main")
        if r.returncode == 0: break
        sh(f"cd {CODE} && git pull -q --rebase -X theirs origin main")
    print("published" if r.returncode == 0 else f"publish failed: {r.stderr[-300:]}")
    state["seen"] = sh(f"cd {CODE} && git rev-parse HEAD").stdout.strip()
    print("=== done; waiting for the next commit ===")
