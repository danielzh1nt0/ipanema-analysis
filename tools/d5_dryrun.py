"""D5 dry run on the free runner (triggers/free.txt first line = tools/d5_dryrun.py): lists what tools/app_cleanup.py
WOULD delete from the app with the default keep list (the 3 demo matches). Never deletes. Writes results/free/d5/plan.md.
Needs SUPABASE_URL + SUPABASE_SERVICE_KEY as GitHub secrets (free.yml passes them); without them it writes that they are missing."""
import os, sys, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import app_cleanup as AC

def main(out="results/free/d5"):
    os.makedirs(out, exist_ok=True); md = [f"# D5 dry run (nothing deleted)", "", f"keep: {', '.join(AC.DEMO_KEEP)}", ""]
    have = {k: bool(os.environ.get(k)) for k in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")}
    if not all(have.values()):
        md.append(f"NOT RUN: secrets missing on this runner: {[k for k, v in have.items() if not v]}")
    else:
        try:
            import supabase  # noqa
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "supabase"], check=True)
        c = AC.Supabase(); rows = c.rows(); ids = AC.plan(rows, AC.DEMO_KEEP); probs = AC.check_keep(AC.DEMO_KEEP, rows, True)
        md += [f"{len(rows)} matches in the app; would delete {len(ids)}:", ""] + [f"- {i}" for i in ids] + [""]
        md += [f"problems: {probs or 'none'}"]
        json.dump({"rows": rows, "would_delete": ids, "problems": probs}, open(f"{out}/plan.json", "w"), indent=1, default=str)
    open(f"{out}/plan.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    main()
