"""V2: run the upload checks on an exported match.   PYTHONPATH=. python tools/upload_check.py <match_id> [--out results/qa/v2]"""
import sys, os, json
from ipanema import uploadcheck as UC

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; out = "results/qa/v2" if "--out" in sys.argv else None
    for mid in args:
        rep = UC.check_export(f"results/volume/runs/matches/{mid}", ".", mid); print(UC.text(rep))
        if out:
            os.makedirs(out, exist_ok=True)
            with open(f"{out}/{mid}.json", "w") as fh: json.dump(rep, fh, indent=1)
