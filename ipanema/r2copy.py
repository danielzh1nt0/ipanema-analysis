"""Copy files from the Modal volume to R2 (29 Sep, Daniel's OK 10:22: file copies on CPU only, no GPU, no training).
Used by modal_app.volume_to_r2. Plain module so it can be tested offline with a fake root and a fake S3 client.

Keys: a volume path keeps its relative path as the R2 key (models/..., labels/...), except a match video
videos/<m>/full.mp4 or videos/<m>.mp4, which goes to <m>/video.mp4 like copy_to_r2.
Matches Daniel did NOT approve for this copy (Reymersholm, Solberga) are never copied, also not inside a dataset zip."""
import os, glob, zipfile

BLOCKED = ("reymersholm", "solberga")

# what K1/T0 + the Kaggle ball-finder training need (29 Sep)
APPROVED = ("p15u-vs-vasalund-2026-09-20", "p09-norrviken-vs-solheim-2026-08-30", "p15u-vs-spanga-2026-09-25", "p15u-vs-djursholm-2026-09-26")
DEFAULT_PATTERNS = [f"labels/{m}_trainset*.json" for m in APPROVED] + [
    "labels/SFKBP1109_*.json",                    # SFK-BP clicks / auto labels kept on the volume
    "labels/*_ball_ds_*.zip",                     # cached crop datasets built by ball_dataset
    "models/wasb_finetuned.pth",                  # WASB fine-tuned on SFK-BP
    "models/ball/clicks_latest.pt", "models/ball/best.json",   # current click finder (70/108) + its record
    "models/ball_finetuned.pt",                   # the 'old' finder ball_round grades against
    "videos/SFKBP1109/full.mp4", "videos/SFKBP1109.mp4",       # full SFK-BP match: the 108 exam frames come from it
]

def key_for(rel):
    p = rel.replace("\\", "/").split("/")
    if p[0] == "videos" and len(p) == 3 and p[2] == "full.mp4": return f"{p[1]}/video.mp4"
    if p[0] == "videos" and len(p) == 2 and p[1].endswith(".mp4"): return f"{p[1][:-4]}/video.mp4"
    return "/".join(p)

def blocked(path):
    """why this file must not be copied (None = fine)"""
    low = os.path.basename(path).lower()
    if any(b in low for b in BLOCKED): return "match not approved"
    if path.endswith(".zip"):
        try:
            names = zipfile.ZipFile(path).namelist()
        except Exception as e: return f"unreadable zip: {e!r}"[:200]
        bad = sorted({b for n in names for b in BLOCKED if b in n.lower()})
        if bad: return f"zip holds crops of {','.join(bad)}"
    return None

def plan(root, patterns):
    """resolve the patterns on the volume -> [{rel, key, bytes} | {pattern, missing} | {rel, skipped}]"""
    out, seen = [], set()
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, pat)))
        hits = [h for h in hits if os.path.isfile(h)]
        if not hits: out.append({"pattern": pat, "missing": True}); continue
        for h in hits:
            rel = os.path.relpath(h, root).replace("\\", "/")
            if rel in seen: continue
            seen.add(rel); why = blocked(h)
            if why: out.append({"rel": rel, "skipped": why}); continue
            out.append({"rel": rel, "key": key_for(rel), "bytes": os.path.getsize(h)})
    return out

def listing(root, dirs=("models", "models/ball", "labels")):
    """names + sizes in a few volume folders, for the record"""
    out = {}
    for d in dirs:
        p = os.path.join(root, d)
        out[d] = {f: os.path.getsize(os.path.join(p, f)) for f in sorted(os.listdir(p)) if os.path.isfile(os.path.join(p, f))} if os.path.isdir(p) else None
    return out

def remote_size(client, bucket, key):
    try: return client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except Exception: return None

def copy(root, patterns, client, bucket, public_url="", dry=False, log=print):
    """upload every planned file whose R2 copy is missing or a different size; verify the size after upload"""
    import time
    rows = plan(root, patterns)
    for r in rows:
        if "key" not in r: continue
        src = os.path.join(root, r["rel"]); have = remote_size(client, bucket, r["key"])
        if have == r["bytes"]: r["status"] = "already on R2"
        elif dry: r["status"] = "would upload"
        else:
            t0 = time.time(); ct = "video/mp4" if r["key"].endswith(".mp4") else ("application/json" if r["key"].endswith(".json") else "application/octet-stream")
            client.upload_file(src, bucket, r["key"], ExtraArgs={"ContentType": ct})
            ok = remote_size(client, bucket, r["key"]) == r["bytes"]
            r["status"] = "uploaded" if ok else "UPLOAD SIZE MISMATCH"; r["seconds"] = round(time.time() - t0, 1)
        if public_url: r["url"] = f"{public_url.rstrip('/')}/{r['key']}"
        log(f"{r['rel']} -> {r['key']} ({r['bytes'] / 1e6:.1f} MB): {r['status']}")
    return rows
