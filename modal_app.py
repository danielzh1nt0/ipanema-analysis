"""Ipanema on Modal: one match per call on an NVIDIA L4. Video from R2, code from GitHub, models/labels/caches on a Volume,
results to Supabase + R2 like the Colab pipeline. Submit with:  modal run modal_app.py --match-id SFKBP1109_s1200"""
import modal, os

APP = "ipanema"; REPO = "https://github.com/danielzh1nt0/ipanema-analysis.git"; ROOT = "/data/match_analysis"
vol = modal.Volume.from_name("ipanema-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "git", "wget", "libgl1", "libglib2.0-0")
         .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
         .pip_install("ultralytics==8.3.40", "supervision==0.25.1", "opencv-python-headless", "numpy<2", "pandas", "scipy", "scikit-learn", "umap-learn", "transformers==4.46.3", "timm==1.0.11", "huggingface_hub<1.0", "pillow", "tqdm", "boto3", "supabase", "requests")
         .pip_install("gdown", "pyyaml", "omegaconf")
         .run_commands("git clone -q --depth 1 https://github.com/nttcom/WASB-SBDT.git /content/WASB-SBDT")
         .run_commands("git clone -q https://github.com/roboflow/sports.git /content/sports && pip install -q -e /content/sports",
                       "cd /content/sports/examples/soccer && bash setup.sh"))
app = modal.App(APP, image=image)

@app.function(gpu="L4", timeout=6 * 3600, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def run_match(match_id: str, video_url: str, start_s: int = 0, dur_s: int = 0, log_tail: int = 400):
    import subprocess, sys, requests, importlib
    subprocess.run(f"rm -rf /content/ipanema-analysis && git clone -q {REPO} /content/ipanema-analysis", shell=True, check=True)
    sys.path.insert(0, "/content/ipanema-analysis")
    os.makedirs(f"{ROOT}/videos", exist_ok=True)
    src = f"{ROOT}/videos/{match_id}.mp4"
    if not os.path.exists(src):
        with requests.get(video_url, stream=True, timeout=600) as r:
            r.raise_for_status(); open(src, "wb").write(b"".join(r.iter_content(1 << 20)))
        if dur_s:   # cut a segment from a full match
            seg = f"{ROOT}/videos/{match_id}_s{start_s}.mp4"
            subprocess.run(["ffmpeg", "-y", "-ss", str(start_s), "-i", src, "-t", str(dur_s), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", seg], check=True, capture_output=True)
            src = seg; match_id = f"{match_id}_s{start_s}"
    from ipanema.config import Settings
    from ipanema.run import run
    S = Settings(root=ROOT, sports_dir="/content/sports", work="/tmp/work")
    lines = []
    def log(*a): s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
    try: summary, folder, z = run(src, match_id=match_id, settings=S, log=log)
    except Exception as e:
        import traceback; log("RUN FAILED: " + traceback.format_exc()); summary = {"error": str(e)}
    vol.commit()
    # publish log + summary to the repo (results/modal/<match>.txt) so results can be read without the dashboard
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok: log("publish skipped: GITHUB_TOKEN not in the ipanema-storage secret (re-run colab_modal_setup.py)")
    else:
        try:
            import json, datetime
            d = "/content/ipanema-analysis/results/modal"; os.makedirs(d, exist_ok=True)
            open(f"{d}/{match_id}.txt", "w").write("\n".join(lines[-2000:]) + "\n\nSUMMARY " + json.dumps(summary, default=str))
            url = f"https://x-access-token:{tok}@github.com/danielzh1nt0/ipanema-analysis.git"
            r = subprocess.run(f"cd /content/ipanema-analysis && git config user.email modal@ipanema && git config user.name modal && git add results/modal && git commit -qm 'modal results for {match_id}' && git pull -q --rebase -X theirs {url} main && git push -q {url} HEAD:main", shell=True, capture_output=True, text=True)
            log("published to results/modal" if r.returncode == 0 else "publish failed: " + (r.stderr or r.stdout)[-300:])
        except Exception as e: log(f"publish failed: {e!r}")
    return {"summary": {k: (v if isinstance(v, (int, float, str, bool, dict, list, type(None))) else str(v)) for k, v in summary.items()}, "log_tail": lines[-log_tail:]}

@app.local_entrypoint()
def main(match_id: str, video_url: str = "", start_s: int = 0, dur_s: int = 0):
    out = run_match.remote(match_id, video_url, start_s, dur_s)
    for l in out["log_tail"]: print(l)
