"""Score our models against stored, labelled data and write results/evals.md (run on every push; costs nothing).

Each eval has a floor: if a change makes a model worse than its floor, the build fails and the run never starts.
"""
import json, os, sys, datetime, numpy as np, bisect, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import ball as BL, tracking as TR
from ipanema.calcheck import judge_frame

FLOORS = {"ball picker v2 (34 held-out frames)": 24, "ball picker v1 (34 held-out frames)": 18, "calibration judge (25 clearly labelled frames)": 25}

def ball_evals():
    z = np.load("results/picker/SFKBP1109_s1200.npz"); n = int(z["n"]); fps = float(z["fps"])
    cands = {k: [] for k in range(n)}
    for k, x, y, c in z["cands"]: cands[int(k)].append((float(x), float(y), float(c)))
    per = {k: [] for k in range(n)}
    for k, tid, team, mx, my, fx, fy, gk in z["players"]:
        per[int(k)].append([int(tid), "A" if team == 0 else "B", np.array([mx, my], float), np.array([fx, fy], float), np.zeros(4), bool(gk)])
    hk = list(z["h_frames"]); Hm = {int(k): z["h"][i].astype(np.float64) for i, k in enumerate(hk)}; valid = sorted(Hm); H = {}
    for k in range(n):
        if k in Hm: H[k] = Hm[k]
        else:
            j = bisect.bisect_left(valid, k); c = [valid[x] for x in (j - 1, j) if 0 <= x < len(valid)]; H[k] = Hm[min(c, key=lambda v: abs(v - k))]
    gt = {int(k): v for k, v in json.loads(bytes(z["gt"]).decode()).items()}
    per, _ = TR.clean(per, 120.0, 70.0, fps, log=lambda *a: None)
    score = lambda b: sum(1 for i, g in gt.items() if g is not None and i in b and np.hypot(b[i][0] - g[0], b[i][1] - g[1]) <= 30)
    total = sum(1 for g in gt.values() if g is not None)
    v2 = BL.bridge(BL.pick_v2(cands, H, 120.0, 70.0, per=per, fps=fps, log=lambda *a: None), fps)
    v1 = BL.bridge(BL.pick_global(cands, H, 120.0, 70.0, per=per, fps=fps, log=lambda *a: None), fps)
    return [("ball picker v2 (34 held-out frames)", score(v2), total), ("ball picker v1 (34 held-out frames)", score(v1), total)]

def calibration_eval():
    meta = json.load(open("results/frames/SFKBP1109/frames.json")); right, unsure = {3, 7, 27}, {4, 5, 23}; agree = 0; total = 0
    for i, r in enumerate(meta):
        if i in unsure: continue
        img = cv2.imread(f"results/frames/SFKBP1109/piece{r['i']:02d}_{r['k']:05d}.jpg")
        v = judge_frame(img, np.asarray(r["H_old"]), 120.0, 70.0)["verdict"]; total += 1
        agree += int((v == "good") == (i in right))
    return [("calibration judge (25 clearly labelled frames)", agree, total)]

def main():
    rows = ball_evals() + calibration_eval()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ver = open("ipanema/__init__.py").read().split('__version__ = "')[1].split('"')[0]
    lines = [f"# Model evals\n", f"Scored on stored labelled data, on every push. Floors fail the build.\n", f"_last run {now}, pipeline v{ver}_\n", "| eval | score | floor | status |", "|---|---|---|---|"]
    bad = []
    for name, got, total in rows:
        floor = FLOORS.get(name, 0); ok = got >= floor
        lines.append(f"| {name} | {got}/{total} | {floor} | {'ok' if ok else 'BELOW FLOOR'} |")
        if not ok: bad.append(f"{name}: {got}/{total} below floor {floor}")
    os.makedirs("results", exist_ok=True); open("results/evals.md", "w").write("\n".join(lines) + "\n")
    hist = "results/evals_history.tsv"
    with open(hist, "a") as f:
        if os.path.getsize(hist) == 0 if os.path.exists(hist) else True: f.write("when\tversion\teval\tscore\ttotal\n")
        for name, got, total in rows: f.write(f"{now}\tv{ver}\t{name}\t{got}\t{total}\n")
    print("\n".join(lines))
    if bad: print("EVAL FAILURE:", "; ".join(bad)); sys.exit(1)

if __name__ == "__main__": main()
