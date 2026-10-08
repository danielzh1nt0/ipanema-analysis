"""C3b (8 Oct, free runner): draw the line model's pose for each blind sample row (results/qa/c3b/sample.json) on the
full-size Vallentuna frame at that second. Title shows only the id and time, never the group.
    python tools/c3b_look.py                              (free runner: video from R2_PUBLIC_URL)
    VIDEO=/path/to.mp4 OUT=dir python tools/c3b_look.py   (dry run on a local file)
-> results/qa/c3b/<id>.jpg (1280x720) + results/qa/c3b/result.json"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import lines as LN

def draw(f, camera, pose, title):
    f = cv2.resize(f, (1280, 720)); g = LN.draw_pose(f, camera, np.array(pose, float), colour=(0, 255, 255), thick=1)
    cv2.rectangle(g, (0, 0), (300, 26), (0, 0, 0), -1); cv2.putText(g, title, (4, 19), 0, 0.6, (255, 255, 255), 1)
    return g

def main():
    S = json.load(open("results/qa/c3b/sample.json")); out = os.environ.get("OUT", "results/qa/c3b"); os.makedirs(out, exist_ok=True)
    src = os.environ.get("VIDEO") or f"{os.environ['R2_PUBLIC_URL'].rstrip('/')}/{S['src_key']}"
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; n = 0; miss = []
    print("video", "ok" if cap.isOpened() else "NOT OPEN", fps, flush=True)
    for q in S["rows"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(q["t"] * fps))); ok, f = cap.read()
        if not ok: miss.append(q["id"]); continue
        cv2.imwrite(f"{out}/{q['id']}.jpg", draw(f, S["camera"], q["pose"], f"{q['id']} t={q['t']:.0f}s"), [cv2.IMWRITE_JPEG_QUALITY, 82]); n += 1
        print(q["id"], flush=True)
    json.dump({"ok": n > 0, "frames": n, "missing": miss}, open(f"{out}/result.json", "w")); print("done", n)

if __name__ == "__main__": main()
