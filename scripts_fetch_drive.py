"""CI step: copy the training matches that have a Drive id to the Modal volume (in parallel); report into results/datasets/"""
import os, json, traceback, modal
os.makedirs("results/datasets", exist_ok=True); man = json.load(open("reference/training_matches.json"))["matches"]
todo = [m for m in man if m.get("drive_id")]; out = []
try:
    f = modal.Function.from_name("ipanema", "fetch_drive")
    out = list(f.starmap([(m["id"], m["drive_id"], m.get("bytes", 0)) for m in todo]))
except Exception: out.append({"error": traceback.format_exc()[-3000:]})
json.dump(out, open("results/datasets/fetch_drive.json", "w"), indent=1); print(json.dumps(out, indent=1))
