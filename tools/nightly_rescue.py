"""N0-27c: build tonight's nightly card from a nightly run whose 'scorecard' job never started.

27 Oct: the backup nightly (run 55) ran all 6 checks fine by 04:42 UTC, but its last job ('scorecard', which downloads
the checks' artifacts and writes results/nightly/<date>.md) sat 'queued' on GitHub for 16+ hours, so there was no card
and the next scheduled nightly waited behind it (concurrency: nightly). This sandbox can't download artifacts, but the
free runner can. Run it on the free runner (triggers/free.txt first line = tools/nightly_rescue.py): it finds the newest
nightly run whose checks finished but whose card job never ran, downloads that run's artifacts with the job's token and
does exactly what the scorecard job does (copy quick.json + tracktest folders, run tools/nightly_scorecard.py) under the
date the run started. Nothing on Modal.
"""
import io, json, os, shutil, subprocess, sys, urllib.error, urllib.request, zipfile

API = "https://api.github.com"
CHECK_JOBS = ("quick", "players")          # job names (players jobs are 'players (<match>, <s>)')


def _get(url, token, raw=False, opener=urllib.request.urlopen):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                               "X-GitHub-Api-Version": "2022-11-28"})
    with opener(req) as r:
        data = r.read()
    return data if raw else json.loads(data)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None


def download(url, token, open_api=None, open_blob=urllib.request.urlopen):
    """artifact zip: the API answers 302 to a signed storage URL that must be fetched WITHOUT our token"""
    open_api = open_api or urllib.request.build_opener(_NoRedirect).open
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    try:
        with open_api(req) as r: return r.read()               # (some hosts answer 200 directly)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308): raise
        loc = e.headers["Location"]
    with open_blob(loc) as r: return r.read()


def is_check(name):
    return name == "quick" or name.startswith("players")


def needs_rescue(jobs):
    """a run needs a card when every check job finished and the scorecard job did not (queued / missing / failed)."""
    checks = [j for j in jobs if is_check(j["name"])]
    card = [j for j in jobs if j["name"] == "scorecard"]
    if not checks or any(j["status"] != "completed" for j in checks): return False
    return not card or card[0]["status"] != "completed" or card[0].get("conclusion") != "success"


def pick_run(runs, jobs_of):
    """newest run (runs come newest first) that needs a card; jobs_of(run_id) -> list of jobs"""
    for r in runs:
        if needs_rescue(jobs_of(r["id"])): return r
    return None


def extract(zip_bytes, dest):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for n in z.namelist():                                  # no paths outside dest
            if n.startswith("/") or ".." in n.split("/"): raise ValueError(f"bad path in artifact: {n}")
        z.extractall(dest)


def place(art, root="."):
    """the scorecard job's copy step: quick.json -> results/nightly, every players-* artifact's folders -> results/qa"""
    os.makedirs(f"{root}/results/nightly", exist_ok=True); os.makedirs(f"{root}/results/qa", exist_ok=True)
    q = f"{art}/quick/quick.json"
    if os.path.exists(q): shutil.copy(q, f"{root}/results/nightly/quick.json")
    for d in sorted(os.listdir(art)):
        if not d.startswith("players-"): continue
        for x in os.listdir(f"{art}/{d}"):
            s, t = f"{art}/{d}/{x}", f"{root}/results/qa/{x}"
            if os.path.isdir(s): shutil.copytree(s, t, dirs_exist_ok=True)
            else: shutil.copy(s, t)


def main(argv=sys.argv[1:]):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "danielzh1nt0/ipanema-analysis")
    if not token: sys.exit("GITHUB_TOKEN missing")
    runs = _get(f"{API}/repos/{repo}/actions/workflows/nightly.yml/runs?per_page=10", token)["workflow_runs"]
    jobs_of = lambda rid: _get(f"{API}/repos/{repo}/actions/runs/{rid}/jobs?per_page=50", token)["jobs"]
    if argv: run = next((r for r in runs if str(r["id"]) == argv[0]), None)
    else: run = pick_run(runs, jobs_of)
    if not run: print("nothing to rescue: every recent nightly run has its card (or its checks are still running)"); return 0
    day = run["run_started_at"][:10]
    if os.path.exists(f"results/nightly/{day}.md"): print(f"card {day} already there - nothing to do"); return 0
    print(f"rescuing run {run['id']} (#{run['run_number']}, started {run['run_started_at']}) -> card {day}")
    art = os.environ.get("NIGHTLY_ART", "/tmp/art_rescue")
    shutil.rmtree(art, ignore_errors=True)
    arts = _get(f"{API}/repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=50", token)["artifacts"]
    for a in arts:
        if a.get("expired"): continue
        extract(download(a["archive_download_url"], token), f"{art}/{a['name']}")
        print("  got", a["name"])
    place(art)
    env = dict(os.environ, NIGHTLY_ART=art, NIGHTLY_DATE=day, PYTHONPATH=".")
    r = subprocess.run([sys.executable, "tools/nightly_scorecard.py"], env=env)
    if r.returncode == 0:
        with open(f"results/nightly/{day}.md", "a") as f:
            f.write(f"\n(Built by tools/nightly_rescue.py on the free runner from nightly run {run['id']}: its scorecard job never started.)\n")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
