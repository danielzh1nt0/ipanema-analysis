"""AUTO mode: keep this cell running. Every loop pulls the latest watcher code from GitHub and runs one step."""
import os, sys, time, subprocess, importlib
from google.colab import userdata
ROOT = os.environ.get("IPANEMA_ROOT", "/content/drive/MyDrive/match_analysis/match_analysis")
CODE = "/content/ipanema-analysis"; REPO = "danielzh1nt0/ipanema-analysis"; TOKEN = userdata.get("GITHUB_TOKEN")
subprocess.run(f"cd {CODE} && git remote set-url origin https://x-access-token:{TOKEN}@github.com/{REPO}.git && git config user.email colab@ipanema && git config user.name 'ipanema colab'", shell=True)
sys.path.insert(0, CODE); state = {}
print("AUTO: watching", REPO, "— leave this cell running")
while True:
    try:
        if "ipanema.watch" in sys.modules: del sys.modules["ipanema.watch"]
        import ipanema.watch as W; W.step(state, ROOT, CODE, REPO)
        time.sleep(60)
    except KeyboardInterrupt: break
    except Exception as e: print("AUTO error:", repr(e)); time.sleep(120)
