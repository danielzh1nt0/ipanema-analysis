"""B4c (1 Oct, free runner): picture sheets where the picker with and without the play-on spare-ball rule disagree
(results/ball/b4c/check_moments.json, old and new pick side by side per moment). Reuses tools/b4b_sheet.py.
    R2_PUBLIC_URL=... python tools/b4c_sheet.py          (or LOCAL_CLIP=clip.mp4 for a dry run)"""
import os, runpy
os.environ.setdefault("B4_CANDS", "results/ball/b4c/check_moments.json")
os.environ.setdefault("B4B_OUT", "results/free/b4c")
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b4b_sheet.py"), run_name="__main__")
