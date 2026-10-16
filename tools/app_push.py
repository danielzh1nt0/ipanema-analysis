"""Push an exported match (made on Kaggle, kaggle/full_match.py STAGE=join) into the app's storage (Supabase) - no Modal.
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python tools/app_push.py <export dir with library.json> [match_id]
Same files and matches row as ipanema.upload.upload_match, but the video is not uploaded: a full match's video is already
on R2 and library.json carries its public URL (video_url). Prints what it did; exits 2 when the secrets are missing."""
import os, sys, json, mimetypes

def push(export_dir, match_id=None, log=print):
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key): log("app_push: SUPABASE_URL / SUPABASE_SERVICE_KEY not set - nothing pushed"); return None
    lib = json.load(open(f"{export_dir}/library.json"))["matches"]
    m = next((x for x in lib if x["id"] == match_id), None) if match_id else (lib[-1] if lib else None)
    if m is None: log(f"app_push: {match_id} not in library.json"); return None
    match_id = m["id"]; folder = f"{export_dir}/{match_id}"
    try:
        from supabase import create_client
    except ImportError:
        import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "supabase"], capture_output=True)
        from supabase import create_client
    sb = create_client(url, key); files = {}
    for k, rel in m["files"].items():
        local = f"{folder}/{os.path.basename(rel)}"
        if k == "video":
            if m.get("video_url"): files["video"] = m["video_url"]
            continue
        if not os.path.exists(local): log(f"  missing {local}"); continue
        dest = f"{match_id}/{os.path.basename(local)}"
        with open(local, "rb") as f: sb.storage.from_("matches").upload(dest, f.read(), {"content-type": mimetypes.guess_type(local)[0] or "application/octet-stream", "upsert": "true"})
        files[k] = dest; log(f"  {dest} ({os.path.getsize(local) / 1e6:.1f} MB)")
    if os.path.exists(f"{folder}/thumb.jpg"):
        with open(f"{folder}/thumb.jpg", "rb") as f: sb.storage.from_("matches").upload(f"{match_id}/thumb.jpg", f.read(), {"content-type": "image/jpeg", "upsert": "true"}); files["thumb"] = f"{match_id}/thumb.jpg"
    md = json.load(open(f"{folder}/match_data.json"))
    row = {"id": match_id, "status": "ready", "duration_s": m["duration_s"], "fps": md.get("fps"), "width": md.get("width"), "height": md.get("height"), "schema_version": md.get("schema_version"),
           "contract": md.get("contract"), "summary": m.get("summary"), "attack_right": m.get("attack_right"), "attack_right_confidence": m.get("attack_right_confidence"), "files": files}
    sb.table("matches").upsert(row).execute(); log(f"app_push: {match_id} -> Supabase ({', '.join(files)})"); return files

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    r = push(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    if r is None: sys.exit(2)
