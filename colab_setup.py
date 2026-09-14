"""Colab bootstrap. In Cell 0:  !git clone (see README) ; exec(open('/content/ipanema-analysis/colab_setup.py').read())"""
import os, subprocess, sys
ROOT = os.environ.get("IPANEMA_ROOT", "/content/drive/MyDrive/match_analysis/match_analysis")
if not os.path.exists("/content/sports/examples/soccer/data/football-player-detection.pt"):
    print("installing environment (2-4 min)...")
    subprocess.run("cd /content && git clone -q https://github.com/roboflow/sports.git", shell=True)
    subprocess.run(f"{sys.executable} -m pip install -q -r /content/ipanema-analysis/requirements.txt", shell=True)
    subprocess.run("cd /content/sports/examples/soccer && bash setup.sh", shell=True)
for d in ("videos", "runs", "cache", "reference", "models"): os.makedirs(f"{ROOT}/{d}", exist_ok=True)
sys.path.insert(0, "/content/ipanema-analysis")
for m in [m for m in sys.modules if m == "ipanema" or m.startswith("ipanema.")]: del sys.modules[m]
import ipanema; from ipanema.config import Settings
S = Settings(root=ROOT)
print("ipanema", ipanema.__version__, "ready · videos:", sorted(os.listdir(f"{ROOT}/videos")))
