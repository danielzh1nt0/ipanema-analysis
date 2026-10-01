"""B4b (1 Oct, free runner): picture sheets where the picker with and without the recurring-spot rule disagree
(results/ball/b4b/check_moments.json, old and new pick side by side per moment). Reuses tools/b4_sheet.py.
    R2_PUBLIC_URL=... python tools/b4b_sheet.py          (or LOCAL_CLIP=clip.mp4 for a dry run)"""
import os, runpy, json
os.environ.setdefault("B4_CANDS", "results/ball/b4b/check_moments.json")
os.environ.setdefault("B4_OUT", "results/free/b4b")
c = os.environ["B4_CANDS"]; D = json.load(open(c))
# b4_sheet keys moments by frame; old and new picks share a frame, so cut them as two passes
for tag in ("old", "new"):
    sub = {"clip": D["clip"], "moments": [m for m in D["moments"] if m["pick"] == tag]}
    p = f"/tmp/b4b_{tag}.json"; json.dump(sub, open(p, "w"))
    os.environ["B4_CANDS"] = p; os.environ["B4_OUT"] = os.environ.get("B4B_OUT", "results/free/b4b") + f"/{tag}"
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b4_sheet.py"), run_name="__main__")
    if not os.environ.get("LOCAL_CLIP"): os.environ["LOCAL_CLIP"] = "/tmp/b4_clip.mp4"   # download once
