"""list SoccerTrack v2 files with sizes, and read a sample of the mot/ box annotations (free runner)"""
import json, os, collections
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); info = api.dataset_info("atomscott/soccertrack-v2", files_metadata=True)
files = [(s.rfilename, s.size or 0) for s in info.siblings]
by = collections.defaultdict(lambda: [0, 0])
for f, s in files: top = f.split("/")[0]; by[top][0] += 1; by[top][1] += s
out = {"folders": {k: {"files": v[0], "GB": round(v[1] / 1e9, 2)} for k, v in by.items()},
       "mot_files": [(f, round(s / 1e6, 1)) for f, s in files if f.startswith("mot/")][:80],
       "video_files": [(f, round(s / 1e9, 2)) for f, s in files if f.startswith("videos/")][:25]}
os.makedirs("results/free", exist_ok=True); json.dump(out, open("results/free/soccertrack_probe.json", "w"), indent=1)
small = sorted([(s, f) for f, s in files if f.startswith("mot/") and f.endswith((".txt", ".json", ".csv", ".ini")) and s < 5e7])[:4]
out["mot_samples"] = {}
for s, f in small:
    try: p = hf_hub_download("atomscott/soccertrack-v2", f, repo_type="dataset", token=os.environ.get("HF_TOKEN") or None); out["mot_samples"][f] = open(p, errors="replace").read()[:1500]
    except Exception as e: out["mot_samples"][f] = f"gated: {type(e).__name__}"
os.makedirs("results/free", exist_ok=True); json.dump(out, open("results/free/soccertrack_probe.json", "w"), indent=1); print(json.dumps(out, indent=1)[:6000])
