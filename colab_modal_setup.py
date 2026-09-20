# MODAL SETUP (run once): token, storage secret, and copy models/labels/calibration/caches from Drive to the Modal volume.
import subprocess, sys, os
from google.colab import userdata
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal"], capture_output=True)
os.environ["MODAL_TOKEN_ID"] = userdata.get("MODAL_TOKEN_ID"); os.environ["MODAL_TOKEN_SECRET"] = userdata.get("MODAL_TOKEN_SECRET")
env = " ".join(f"{k}={userdata.get(k)}" for k in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET", "R2_PUBLIC_URL", "GITHUB_TOKEN"))
print(subprocess.run(f"modal secret create ipanema-storage {env} --force", shell=True, capture_output=True, text=True).stdout[-200:])
subprocess.run("modal volume create ipanema-data", shell=True, capture_output=True)
ROOT = "/content/drive/MyDrive/match_analysis/match_analysis"
for d in ("models", "reference", "cache"):
    if os.path.isdir(f"{ROOT}/{d}"):
        r = subprocess.run(f"modal volume put ipanema-data {ROOT}/{d} /match_analysis/{d} --force", shell=True, capture_output=True, text=True); print(d, "->", "ok" if r.returncode == 0 else (r.stderr or r.stdout)[-300:])
print("modal ready")
