"""K1b on the free GitHub runner (29 Sep; Kaggle's 2 GPU sessions were busy): follow Daniel's checked balls by picture
matching, crops at the WASB peaks stored by the Kaggle K1 run. CPU only, no torch. -> results/free/k1b/"""
import os, sys, subprocess
env = dict(os.environ, K1_HOW="template", K1_NEG="4", K1_WINDOW="12", K1_TMP=os.environ.get("K1_TMP", "/tmp"),
           K1_PEAKS=os.environ.get("K1_PEAKS", "results/kaggle/ballcrops_k1/k1_peaks.json.gz"))
sys.exit(subprocess.run([sys.executable, "tools/k1_crops.py", sys.argv[1] if len(sys.argv) > 1 else "results/free/k1b"], env=env).returncode)
