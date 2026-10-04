"""5 Oct (V1): tile pixel (960x540 sheet tile) at video time t -> pitch metres, using the exported frame's pitch_lines
(metres->pixels homography of the 1920x1080 frame; second half already mirrored like the players).
    PYTHONPATH=. python tools/px2m.py <match> <t> <tile_x> <tile_y>"""
import sys, json, glob, bisect, numpy as np
def load(m):
    fr = []
    for f in sorted(glob.glob(f"results/volume/runs/matches/{m}/frames_*.json")): fr += json.load(open(f))["frames"]
    return fr
def px2m(fr, t, x, y, scale=2.0):
    ts = [f["t"] for f in fr]; i = min(bisect.bisect_left(ts, t), len(ts) - 1)
    for d in range(0, 40):                     # nearest frame with a calibration
        for j in (i - d, i + d):
            if 0 <= j < len(fr) and fr[j].get("pitch_lines"):
                H = np.array(fr[j]["pitch_lines"]).reshape(3, 3); v = np.linalg.inv(H) @ np.array([x * scale, y * scale, 1.0])
                return (v[:2] / v[2]).round(1).tolist(), round(fr[j]["t"] - t, 2), fr[j].get("cal_ok")
    return None, None, None
if __name__ == "__main__":
    m, t, x, y = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]); print(px2m(load(m), t, x, y))
