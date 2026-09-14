"""Upload a processed match to Supabase (json, stats, kits, thumb + matches row) and Cloudflare R2 (video). Runs at the end of run() when credentials are available."""
import os, json, mimetypes

def _creds():
    keys = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET", "R2_PUBLIC_URL")
    c = {}
    try:
        from google.colab import userdata
        for k in keys:
            try: c[k] = userdata.get(k)
            except Exception: c[k] = os.environ.get(k)
    except Exception:
        c = {k: os.environ.get(k) for k in keys}
    return c if all(c.get(k) for k in keys) else None

def upload_match(root, match_id, log=print):
    c = _creds()
    if not c: log("upload: no Supabase/R2 credentials found — skipped"); return None
    try:
        from supabase import create_client
        import boto3
    except ImportError:
        import subprocess, sys; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "supabase", "boto3"], capture_output=True)
        from supabase import create_client
        import boto3
    sb = create_client(c["SUPABASE_URL"], c["SUPABASE_SERVICE_KEY"])
    r2 = boto3.client("s3", endpoint_url=f"https://{c['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", aws_access_key_id=c["R2_ACCESS_KEY"], aws_secret_access_key=c["R2_SECRET_KEY"], region_name="auto")
    bucket, pub = c["R2_BUCKET"], c["R2_PUBLIC_URL"].rstrip("/")
    lib = json.load(open(f"{root}/runs/library.json"))["matches"]; m = next((x for x in lib if x["id"] == match_id), None)
    if m is None: log("upload: match not in library.json"); return None
    folder = f"{root}/runs/matches/{match_id}"; files = {}
    for key, rel in m["files"].items():
        local = f"{root}/runs/{rel}"
        if not os.path.exists(local): continue
        dest = f"{match_id}/{os.path.basename(local)}"
        if key == "video":
            try: r2.head_object(Bucket=bucket, Key=dest)                     # already there: skip the 200 MB
            except Exception: r2.upload_file(local, bucket, dest, ExtraArgs={"ContentType": "video/mp4"})
            files[key] = f"{pub}/{dest}"
        else:
            with open(local, "rb") as f: sb.storage.from_("matches").upload(dest, f.read(), {"content-type": mimetypes.guess_type(local)[0] or "application/octet-stream", "upsert": "true"})
            files[key] = dest
    if os.path.exists(f"{folder}/thumb.jpg"):
        with open(f"{folder}/thumb.jpg", "rb") as f: sb.storage.from_("matches").upload(f"{match_id}/thumb.jpg", f.read(), {"content-type": "image/jpeg", "upsert": "true"}); files["thumb"] = f"{match_id}/thumb.jpg"
    md = json.load(open(f"{folder}/match_data.json"))
    row = {"id": match_id, "status": "ready", "duration_s": m["duration_s"], "fps": md.get("fps"), "width": md.get("width"), "height": md.get("height"), "schema_version": md.get("schema_version"),
           "contract": md.get("contract"), "summary": m.get("summary"), "attack_right": m.get("attack_right"), "attack_right_confidence": m.get("attack_right_confidence"), "files": files}
    sb.table("matches").upsert(row).execute()
    log(f"upload: {match_id} -> Supabase + R2 ({', '.join(files)})"); return files
