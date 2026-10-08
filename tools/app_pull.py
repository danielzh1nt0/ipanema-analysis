"""5 Oct: download what the app actually shows (Supabase public bucket 'matches') for the three demo matches, for a free
audit of every stat. Free runner (this sandbox cannot reach supabase.co).  python tools/app_pull.py -> results/free/app_files/<id>/"""
import os, urllib.request, json
SUPA = "https://savbsnvusqbogdzvkjaf.supabase.co/storage/v1/object/public/matches"
for m in ("SFKBP1109", "p15u-vs-aik-2026-09-21-bd09", "p15u-vs-vallentuna-2026-10-03-6cce"):
    d = f"results/free/app_files/{m}"; os.makedirs(d, exist_ok=True); got = []
    names = ["stats.json", "match_data.json"] + [f"frames_{i:03d}.json" for i in range(40)]
    for n in names:
        try:
            with urllib.request.urlopen(f"{SUPA}/{m}/{n}", timeout=120) as r: open(f"{d}/{n}", "wb").write(r.read()); got.append(n)
        except Exception as e:
            if n.startswith("frames_") and got and got[-1].startswith("frames_"): break
            print(m, n, repr(e)[:120])
    print(m, len(got), "files", flush=True)
