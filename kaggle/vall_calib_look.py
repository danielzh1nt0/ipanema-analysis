# Kaggle (free, 4 Oct, C3): Vallentuna camera rows marked 'refine doubtful' but with forward/backward agreement <= 15 px:
# are they usable? Draw the pitch lines each pose predicts on the video frame (16 doubtful + 8 confident rows, random).
import json, cv2, urllib.request, subprocess, sys, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; T = "/kaggle/temp"
subprocess.run(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia", shell=True); sys.path.insert(0, f"{T}/ia")
from ipanema import lines as LN
D = json.load(open(f"{T}/ia/results/review/calib/vall_rows.json")); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5); n = 0
for i, q in enumerate(D["rows"]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(q["t"] * fps))); ok, f = cap.read()
    if not ok: continue
    f = cv2.resize(f, (1920, 1080)); g = LN.draw_pose(f, D["camera"], np.array(q["pose"], float), colour=(0, 0, 255) if q["kind"] == "doubtful" else (0, 255, 0), thick=2)
    cv2.rectangle(g, (0, 0), (900, 26), (0, 0, 0), -1); cv2.putText(g, f"r{i:02d} {q['kind']} t={q['t']:.0f}s cost={q['cost']} fwd/bwd={q['fb']} px", (4, 19), 0, 0.6, (255, 255, 255), 1)
    cv2.imwrite(f"{W}/r{i:02d}_{q['kind']}.jpg", cv2.resize(g, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 82]); n += 1
json.dump({"ok": True, "frames": n}, open(f"{W}/result.json", "w")); print("done", n)
