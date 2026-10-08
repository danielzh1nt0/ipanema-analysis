"""5 Oct / 9 Oct: download what the app actually shows (Supabase bucket 'matches', private: signed with the service key)
for the three demo matches, for a free audit of every stat. Free runner (this sandbox cannot reach supabase.co).
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python tools/app_pull.py -> results/free/app_files/<id>/"""
import os, urllib.request, json
URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SERVICE_KEY"]
MATCHES = os.environ.get("PULL", "p15u-vs-vallentuna-2026-10-03-6cce").split(",")
for m in MATCHES:
    d = f"results/free/app_files/{m}"; os.makedirs(d, exist_ok=True); got = []
    names = ["stats.json", "match_data.json"] + [f"frames_{i:03d}.json" for i in range(40)]
    for n in names:
        req = urllib.request.Request(f"{URL}/storage/v1/object/matches/{m}/{n}", headers={"Authorization": f"Bearer {KEY}", "apikey": KEY})
        try:
            with urllib.request.urlopen(req, timeout=120) as r: open(f"{d}/{n}", "wb").write(r.read()); got.append(n)
        except Exception as e:
            if n.startswith("frames_") and got and got[-1].startswith("frames_"): break
            print(m, n, repr(e)[:120])
    print(m, len(got), "files", flush=True)
