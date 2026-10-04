"""4 Oct: Vallentuna frames + people boxes for the offline kit test (red vs black in hard shade; the match model put reds and
shaded blacks in one team). Free runner, CPU."""
import os, runpy
os.environ["F1C_MATCHES"] = "p15u-vs-vallentuna-2026-10-03-6cce:18"
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1c_frames.py"), run_name="__main__")
