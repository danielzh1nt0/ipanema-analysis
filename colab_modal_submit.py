# SUBMIT A MATCH TO MODAL. Deploys the app (idempotent), spawns the job, and confirms it started. You can close the laptop after.
import subprocess, sys, os, time
from google.colab import userdata
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal"], capture_output=True)
os.environ["MODAL_TOKEN_ID"] = userdata.get("MODAL_TOKEN_ID"); os.environ["MODAL_TOKEN_SECRET"] = userdata.get("MODAL_TOKEN_SECRET")
subprocess.run("rm -rf /content/ipanema-analysis && git clone -q https://github.com/danielzh1nt0/ipanema-analysis.git /content/ipanema-analysis", shell=True)
r = subprocess.run("cd /content/ipanema-analysis && modal deploy modal_app.py", shell=True, capture_output=True, text=True); print((r.stdout or r.stderr)[-300:])
import modal
MATCH = os.environ.get("IPANEMA_MATCH", "SFKBP1109_s1200"); VIDEO_URL = os.environ.get("IPANEMA_VIDEO_URL", f"{userdata.get('R2_PUBLIC_URL')}/{MATCH}/video.mp4")
f = modal.Function.from_name("ipanema", "run_match")
call = f.spawn(MATCH, VIDEO_URL)
print("submitted", MATCH, "call id", call.object_id)
for _ in range(12):
    time.sleep(10)
    try: st = call.get(timeout=0.1); print("finished already:", st.get("summary", {}).get("ball_check")); break
    except modal.exception.TimeoutError: print("running...")
    except Exception as e: print("status:", type(e).__name__, str(e)[:200]); break
print("you can close this now — results publish to the repo and the app when done")
