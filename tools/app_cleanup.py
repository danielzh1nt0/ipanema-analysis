"""D5 (3 Oct): remove every match from the app except the demo list. Works on the app's database (Supabase 'matches' rows,
'match_labels' rows, storage bucket 'matches/<id>/*'); R2 videos are left (no cost to keep, avoids deleting footage).
Needs SUPABASE_URL + SUPABASE_SERVICE_KEY in the environment (GitHub secrets on the free runner, or run locally).
    python tools/app_cleanup.py --keep SFKBP1109,p15u-vs-aik-2026-09-21-bd09            (dry run: lists what would go)
    python tools/app_cleanup.py --keep ... --apply                                      (deletes)
The logic is in plan()/apply() with a tiny client interface so it is unit-tested with a fake (tests/test_app_cleanup.py)."""
import os, sys, json, argparse

def plan(rows, keep):
    """rows: [{id, ...}] -> ids to delete (everything not in keep), sorted"""
    keep = set(keep); return sorted(r["id"] for r in rows if r.get("id") and r["id"] not in keep)

def storage_paths(list_fn, mid):
    """every object under matches/<mid>/ (list_fn(prefix) -> [{name}])"""
    return [f"{mid}/{o['name']}" for o in (list_fn(mid) or []) if o.get("name")]

def apply(client, ids, log=print):
    """client: .list(prefix) -> [{name}], .remove(paths), .delete_rows(table, column, value). Returns a report."""
    rep = {"deleted": [], "errors": []}
    for mid in ids:
        try:
            paths = storage_paths(client.list, mid)
            if paths: client.remove(paths)
            for table in ("match_labels", "matches"): client.delete_rows(table, "match_id" if table == "match_labels" else "id", mid)
            rep["deleted"].append({"id": mid, "files": len(paths)}); log(f"deleted {mid}: {len(paths)} files + rows")
        except Exception as e:
            rep["errors"].append({"id": mid, "error": repr(e)[:300]}); log(f"FAILED {mid}: {e!r}")
    return rep

class Supabase:
    def __init__(self):
        from supabase import create_client
        self.db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]); self.bucket = self.db.storage.from_("matches")
    def rows(self): return self.db.table("matches").select("id,status,created_at,duration_s").execute().data or []
    def list(self, prefix): return self.bucket.list(prefix)
    def remove(self, paths): return self.bucket.remove(paths)
    def delete_rows(self, table, column, value): return self.db.table(table).delete().eq(column, value).execute()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--keep", required=True, help="comma-separated match ids to keep"); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    keep = [k.strip() for k in a.keep.split(",") if k.strip()]
    if not all(os.environ.get(k) for k in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")): sys.exit("SUPABASE_URL and SUPABASE_SERVICE_KEY are not set")
    c = Supabase(); rows = c.rows(); ids = plan(rows, keep)
    print(f"{len(rows)} matches in the app; keeping {len(keep)}; {'deleting' if a.apply else 'would delete'} {len(ids)}: {ids}")
    missing = [k for k in keep if k not in {r['id'] for r in rows}]
    if missing: print("WARNING: keep ids not found in the app:", missing)
    if a.apply:
        rep = apply(c, ids); os.makedirs("results/app", exist_ok=True); json.dump(rep, open("results/app/cleanup_report.json", "w"), indent=1); print(json.dumps(rep, indent=1))
