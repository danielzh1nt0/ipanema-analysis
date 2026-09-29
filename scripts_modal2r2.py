"""CI step [modal2r2] (or [modal2r2:dry]): copy the K1/T0 ball files from the Modal volume to R2 (CPU only, Daniel's OK
29 Sep 10:22) -> results/datasets/modal_to_r2.json (what was copied: keys, sizes) + modal_volume_listing.json"""
import os, re, json, modal
dry = "[modal2r2:dry]" in os.environ.get("MSG", "")
out = modal.Function.from_name("ipanema", "volume_to_r2").remote(None, dry)
os.makedirs("results/datasets", exist_ok=True)
json.dump(out.get("listing"), open("results/datasets/modal_volume_listing.json", "w"), indent=1)
rows = out.get("files", [])
rep = {"dry": dry, "copied_or_present": [r for r in rows if "key" in r], "skipped": [r for r in rows if "skipped" in r],
       "missing_patterns": [r["pattern"] for r in rows if r.get("missing")],
       "total_GB": round(sum(r["bytes"] for r in rows if "key" in r) / 1e9, 2)}
json.dump(rep, open("results/datasets/modal_to_r2.json", "w"), indent=1); print(json.dumps(rep, indent=1)[:6000])
