"""24 Oct 2026: every match the app has marked "processing" (an upload with no analysis yet) is sent through the free Kaggle
pipeline (.github/workflows/process-match.yml) - one repository_dispatch per match. Runs in the nightly job.
Needs SUPABASE_URL + SUPABASE_SERVICE_KEY (to read the matches table) and GITHUB_TOKEN (to start the workflow).
    python tools/process_uploads.py [--dry]"""
import os, sys, json, urllib.request

def main():
    dry = "--dry" in sys.argv
    url, key, gh = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY"), os.environ.get("GITHUB_TOKEN")
    if not (url and key): print("process_uploads: SUPABASE_URL / SUPABASE_SERVICE_KEY not set - cannot see the uploads"); sys.exit(2)
    try:
        from supabase import create_client
    except ImportError:
        import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "supabase"], capture_output=True)
        from supabase import create_client
    sb = create_client(url, key)
    rows = sb.table("matches").select("id, status, files, updated_at").eq("status", "processing").execute().data or []
    started = json.load(open("results/free/process_uploads.json")) if os.path.exists("results/free/process_uploads.json") else {}
    repo = os.environ.get("GITHUB_REPOSITORY", "danielzh1nt0/ipanema-analysis"); out = []
    for r in rows:
        m = r["id"]; video = (r.get("files") or {}).get("video")
        if not video: print(f"{m}: no video url yet"); continue
        if m in started: print(f"{m}: already sent {started[m]}"); continue
        print(f"{m}: processing -> free pipeline" + (" (dry)" if dry else ""))
        if dry or not gh: out.append(m); continue
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}/dispatches", data=json.dumps({"event_type": "process-match", "client_payload": {"match_id": m}}).encode(),
                                     headers={"Authorization": f"Bearer {gh}", "Accept": "application/vnd.github+json", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp: print("  dispatched", resp.status)
        started[m] = r.get("updated_at") or "sent"; out.append(m)
    os.makedirs("results/free", exist_ok=True); json.dump(started, open("results/free/process_uploads.json", "w"), indent=1)
    print(json.dumps({"processing": [r["id"] for r in rows], "sent": out}))

if __name__ == "__main__": main()
