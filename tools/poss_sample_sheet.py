"""17 Oct (free GitHub runner): random in-play frames of a match as contact sheets, to judge who has the ball by eye (Claude).
Nothing from the pipeline is drawn: plain centre crops around the exported ball (or the frame centre), so the judgement is blind.
    R2_PUBLIC_URL=... python tools/poss_sample_sheet.py <results/kaggle/name> <match_id> [n=150] [seed=11]
-> results/qa/poss_sample/<name>/sheet_NN.jpg (12 crops each, labelled s000..) + index.json [{id, t}]"""
import os, sys, json, gzip, random, bisect, cv2, numpy as np

def main():
    src, M = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 150; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 11
    name = os.path.basename(src.rstrip("/")); out = f"results/qa/poss_sample/{name}"; os.makedirs(out, exist_ok=True)
    d = json.load(gzip.open(f"{src}/frames_lite.json.gz")); fr = d["frames"]; ts = [f["t"] for f in fr]
    per = json.load(open(f"periods/{M}.json"))["periods_s"] if os.path.exists(f"periods/{M}.json") else [[0, 1e9]]
    rng = random.Random(seed); times = []; tries = 0
    while len(times) < N and tries < 200 * N:   # 17 Oct (worker): a short match can't fit N moments 5 s apart -> stop instead of looping forever
        tries += 1; a, b = rng.choice(per); t = round(rng.uniform(a, b), 1)
        if all(abs(t - x) > 5 for x in times): times.append(t)
    times.sort(); cap = cv2.VideoCapture(f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{M}/video.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    if not cap.isOpened(): sys.exit("video not reachable")
    crops = []; index = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, img = cap.read()
        if not ok: continue
        img = cv2.resize(img, (1920, 1080)); j = bisect.bisect_left(ts, t); j = min([k for k in (j - 1, j) if 0 <= k < len(ts)], key=lambda k: abs(ts[k] - t))
        b = fr[j].get("ball"); cx, cy = (b["px"] if b else (960, 540)); W_, H_ = 560, 320
        x0 = int(min(max(cx - W_ // 2, 0), 1920 - W_)); y0 = int(min(max(cy - H_ // 2, 0), 1080 - H_))
        c = img[y0:y0 + H_, x0:x0 + W_].copy(); cv2.putText(c, f"s{i:03d}", (6, 22), 0, 0.7, (255, 255, 255), 2)
        crops.append(c); index.append({"id": f"s{i:03d}", "t": t, "ours": fr[j].get("possession"), "phase": fr[j].get("phase")})
    cap.release()
    for s in range(0, len(crops), 12):
        rows = [np.hstack(crops[s + r * 3: s + r * 3 + 3] + [np.zeros((320, 560, 3), np.uint8)] * (3 - len(crops[s + r * 3: s + r * 3 + 3]))) for r in range(4) if s + r * 3 < len(crops)]
        cv2.imwrite(f"{out}/sheet_{s // 12:02d}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump(index, open(f"{out}/index.json", "w"), indent=1); print(len(index), "crops ->", out)

if __name__ == "__main__": main()
