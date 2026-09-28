"""FREE player check on GitHub's own runner (no Modal): real player detector + 10 frames from every match.
Videos: R2 public URL (opened over HTTP, only the needed frames are read) or, for the Drive matches, downloaded one at a
time and deleted. Pictures + boxes -> results/qa/players/<match>/. Trigger: triggers/playerbench.txt"""
import os, sys, json, time, subprocess, traceback, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import playerbench as PB, tracking as TR, teams as T
OUT = "results/qa/players"; os.makedirs(OUT, exist_ok=True); LOG = []
def log(m): s = f"{time.strftime('%H:%M:%S')} {m}"; print(s, flush=True); LOG.append(s); open(f"{OUT}/log.txt", "w").write("\n".join(LOG) + "\n")
R2 = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
MATCHES = [{"id": "SFKBP1109", "r2": ["p15u-vs-bp-2026-09-22-2000/video.mp4", "SFKBP1109/video.mp4"]},
           {"id": "p15u-vs-aik-2026-09-21-bd09", "r2": ["p15u-vs-aik-2026-09-21-bd09/video.mp4"]}]
MATCHES += [{"id": m["id"], "drive": m["drive_id"]} for m in json.load(open("reference/training_matches.json"))["matches"]]
if os.environ.get("ONLY"): MATCHES = [m for m in MATCHES if m["id"] in os.environ["ONLY"].split(",")]
if len(sys.argv) > 1: MATCHES = [m for m in MATCHES if m["id"] in sys.argv[1].split(",")] or [{"id": "local", "path": sys.argv[1]}]

def open_video(m):
    if m.get("path"): return m["path"], None
    for key in m.get("r2", []):
        if not R2: break
        u = f"{R2}/{key}"; cap = cv2.VideoCapture(u)
        if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1000: cap.release(); return u, None
        cap.release(); log(f"  {m['id']}: not at {key}")
    if m.get("drive"):
        dst = f"/tmp/{m['id']}.mp4"; t0 = time.time()
        r = subprocess.run(["gdown", "-q", m["drive"], "-O", dst], capture_output=True, text=True)       # file id works on every gdown version (--fuzzy was removed)
        if r.returncode == 0 and os.path.exists(dst): log(f"  {m['id']}: downloaded {os.path.getsize(dst) / 1e9:.1f} GB in {time.time() - t0:.0f} s"); return dst, dst
        log(f"  {m['id']}: Drive download failed: {r.stderr[-300:]}")
    return None, None

def main(weights):
    from ultralytics import YOLO
    model = YOLO(weights); summ = {}
    for m in MATCHES:
        t0 = time.time()
        try:
            src, tmp = open_video(m)
            if src is None: summ[m["id"]] = {"error": "video not reachable"}; continue
            cap = cv2.VideoCapture(src); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); d = f"{OUT}/{m['id']}"; os.makedirs(d, exist_ok=True); items = []
            for k in PB.pick_frames(nf, 10):
                cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
                if not ok: log(f"  {m['id']}: frame {k} unreadable"); continue
                det, names = TR.detect_tiled_batch(model, [f], 0.10, TR.FOLLOW_TILES, imgsz=960, half=False)[0]
                boxes = PB.classify(det, names, f, T.referee_kit)
                cv2.imwrite(f"{d}/f{k:07d}.jpg", PB.draw(f, boxes, f"{m['id'][:28]} f{k}"), [cv2.IMWRITE_JPEG_QUALITY, 85])
                items.append({"frame": k, "size": [f.shape[1], f.shape[0]], "boxes": boxes})
            cap.release(); json.dump(items, open(f"{d}/boxes.json", "w"))
            if tmp: os.remove(tmp)
            summ[m["id"]] = {"frames": len(items), "seconds": round(time.time() - t0)}; log(f"{m['id']}: {len(items)} frames in {time.time() - t0:.0f} s")
        except Exception: summ[m["id"]] = {"error": traceback.format_exc()[-600:]}; log(f"{m['id']}: FAILED {traceback.format_exc()[-300:]}")
        json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    return summ

if __name__ == "__main__":
    w = os.environ.get("PLAYER_WEIGHTS", "football-player-detection.pt")
    if not os.path.exists(w): subprocess.run(["gdown", "-q", "-O", w, "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"], check=True)
    main(w)
