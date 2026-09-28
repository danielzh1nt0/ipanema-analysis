"""28 Sep: QA for tracking.fill_gaps on the SFK-BP clip (saved positions, local, no cost): draws observed (red/blue) and
filled (yellow, numbered) players on the saved clip frames so each fill can be judged by eye."""
import sys, os, json, glob, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import tracking as TR

def load(path="results/volume/runs/matches/SFKBP1109_s1200/match_data.json"):
    d = json.load(open(path)); F = d["frames"]
    per = {k: [[p["id"], p["team"], np.array(p["m"]), np.array(p["px"]), None, p["gk"]] for p in f["players"]] for k, f in enumerate(F)}
    H = {k: (np.array(f["pitch_lines"]).reshape(3, 3) if f.get("pitch_lines") else None) for k, f in enumerate(F)}
    return d, per, H

def sheets(per, out, every=5, n=8, scale=1.5):
    fs = sorted(glob.glob("results/frames_SFKBP1109_s1200/*.jpg"))[::every][:n]; tiles = []
    for f in fs:
        k = int(f.split("/f")[-1][:-4]); im = cv2.imread(f); c_ = 0
        for r in per[k]:
            if r[3] is None: continue
            x, y = np.asarray(r[3]) / scale; filled = len(r) > 6
            cv2.circle(im, (int(x), int(y)), 11, (0, 255, 255) if filled else ((0, 0, 255) if r[1] == "A" else (255, 128, 0)), 3 if filled else 2)
            if filled: c_ += 1; cv2.putText(im, str(c_), (int(x) + 10, int(y) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        t = f"f{k} observed {len(per[k]) - c_} + filled {c_}"
        cv2.putText(im, t, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6); cv2.putText(im, t, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        tiles.append(im)
    for i in range(0, len(tiles), 2): cv2.imwrite(f"{out}/fill{i // 2}.jpg", np.vstack(tiles[i:i + 2]), [cv2.IMWRITE_JPEG_QUALITY, 88])

if __name__ == "__main__":
    d, per, H = load(); fps = d["fps"]
    b = {t: float(np.median([sum(r[1] == t for r in per[k]) for k in per])) for t in "AB"}
    per, nf = TR.fill_gaps(per, fps, 1.0, H=H)
    a = {t: float(np.median([sum(r[1] == t for r in per[k]) for k in per])) for t in "AB"}
    print("filled", nf, "median per frame before", b, "after", a)
    sheets(per, sys.argv[1] if len(sys.argv) > 1 else ".")
