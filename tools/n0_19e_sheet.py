"""N0 19 Oct (free runner): blind A/B picture sheets where the old and new RF-DETR + WASB fusion pick a different ball on
the SFK-BP test clip (results/qa/n0-19e/ab_moments.json; folder 'old' = side A, 'new' = side B, key in ab_key.json).
Reuses tools/b4b_sheet.py.  R2_PUBLIC_URL=... python tools/n0_19e_sheet.py   (or LOCAL_CLIP=clip.mp4 for a dry run)"""
import os, runpy
os.environ.setdefault("B4_CANDS", "results/qa/n0-19e/ab_moments.json")
os.environ.setdefault("B4B_OUT", "results/free/n0-19e")
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b4b_sheet.py"), run_name="__main__")
