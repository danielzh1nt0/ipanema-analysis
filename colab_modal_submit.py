# SUBMIT A MATCH TO MODAL (then you can close the laptop). Usage: set MATCH and VIDEO_URL below.
import subprocess, sys, os
from google.colab import userdata
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal"], capture_output=True)
os.environ["MODAL_TOKEN_ID"] = userdata.get("MODAL_TOKEN_ID"); os.environ["MODAL_TOKEN_SECRET"] = userdata.get("MODAL_TOKEN_SECRET")
subprocess.run("rm -rf /content/ipanema-analysis && git clone -q https://github.com/danielzh1nt0/ipanema-analysis.git /content/ipanema-analysis", shell=True)
MATCH = os.environ.get("IPANEMA_MATCH", "SFKBP1109_s1200"); VIDEO_URL = os.environ.get("IPANEMA_VIDEO_URL", f"{userdata.get('R2_PUBLIC_URL')}/{MATCH}/{MATCH}.mp4")
# detach: the job keeps running on Modal after this cell (and the laptop) stops
print(subprocess.run(f"cd /content/ipanema-analysis && modal run --detach modal_app.py --match-id {MATCH} --video-url {VIDEO_URL}", shell=True, capture_output=True, text=True).stdout[-800:])
