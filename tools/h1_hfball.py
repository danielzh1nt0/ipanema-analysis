"""H1 (free runner, no GPU): download martinjolif/football-ball-detection (Hugging Face, CC BY 4.0) and look at it:
layout, splits, image sizes, ball size next to our Veo ball, and picture sheets (48 random images, ball boxed, zoomed
ball crop at Veo scale) + a strip of our own Veo exam balls at the same zoom. Nothing is trained here.
Local dry run: H1_LOCAL_DIR=<folder with a fake dataset> python tools/h1_hfball.py
-> results/free/h1/ (stats.json, sheet_*.jpg, veo_reference.jpg, files.txt)"""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import hfball as H

REPO = os.environ.get("H1_REPO", "martinjolif/football-ball-detection")
OUT = os.environ.get("H1_OUT", "results/free/h1")
root = os.environ.get("H1_LOCAL_DIR")
if not root:
    try: import pyarrow  # noqa: F401  parquet layout needs it; the free runner does not install it
    except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"], check=False)
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=REPO, repo_type="dataset", local_dir="/tmp/h1_data", token=os.environ.get("HF_TOKEN") or None)
os.makedirs(OUT, exist_ok=True)
files = []
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        if "/.cache/" not in p and "/.git/" not in p: files.append((os.path.relpath(p, root), os.path.getsize(p)))
exts = {}
for f, s in files: e = os.path.splitext(f)[1].lower(); exts[e] = exts.get(e, 0) + 1
with open(f"{OUT}/files.txt", "w") as fh:
    fh.write(f"{len(files)} files; by type: {json.dumps(exts)}\n")
    for f, s in sorted(files)[:200]: fh.write(f"{s:>12}  {f}\n")
for readme in ("README.md", "data.yaml", "README.dataset.txt", "README.roboflow.txt"):
    p = os.path.join(root, readme)
    if os.path.exists(p): open(f"{OUT}/{readme.replace('/', '_')}", "w").write(open(p, errors="replace").read()[:20000])
rf = "results/kaggle/ballfinder_rfdetr/result.json"
exam_rows = json.load(open(rf)).get("exam_rows") if os.path.exists(rf) else None
st = H.run(root, OUT, exam_dir="results/ballclicks/3", exam_rows=exam_rows)
print(json.dumps({k: v for k, v in st.items() if k != "sheets"}, indent=1)); print("sheets", len(st["sheets"]))
