"""B3 check (3 Oct, free runner): are the SoccerTrack v2 tracking XML and the half videos in sync, and what is in the
dataset's ball/ folder? The first B3 pass (results/free/b3) put the projected ball on empty grass in most check tiles.
Fixed camera -> a median background of 40 frames; a projected player 'hits' when there is foreground within R px of
his feet. For video frames spread over the half, try XML offsets from -SPAN to +SPAN around the file's own offset;
the true offset is where hits peak. Also: keys/shapes/first rows of ball/<m>_<half>_ball.npz, and how far those
points are from our projected ball at the best offset. Writes results/free/b3/sync.json + sync_<m>.jpg."""
import json, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import soccertrack as ST
import tools.b3_soccertrack as B

OUT = os.environ.get("B3_OUT", "results/free/b3"); os.makedirs(OUT, exist_ok=True)
MATCHES = os.environ.get("ST_MATCHES", "117093,118575,132877,117092").split(",")
SPAN = int(os.environ.get("SPAN", "1500")); STEP = int(os.environ.get("STEP", "5")); R = int(os.environ.get("HIT_R", "14"))
NF = int(os.environ.get("N_FRAMES", "12")); t0 = time.time()
def log(m): print(f"{(time.time() - t0) / 60:5.1f} min  {m}", flush=True)

def background(vp, n):
    cap = cv2.VideoCapture(vp); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for k in np.linspace(tot * 0.05, tot * 0.95, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read()
        if ok: fr.append(cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cap.release(); return np.median(np.array(fr), 0).astype(np.uint8), tot

def fg_mask(img, bg):
    s = cv2.resize(img, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)
    d = cv2.absdiff(s, bg).max(2); m = (d > 40).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

def hits(mask, pts, r=R // 2):
    h, w = mask.shape; n = 0; k = 0
    for x, y in pts / 2.0:
        x, y = int(x), int(y)
        if not (r <= x < w - r and r <= y < h - r): continue
        k += 1; n += mask[y - r:y + r + 1, x - r:x + r + 1].any()
    return n, k

def main():
    files = B.listing(); res = {}
    for m in MATCHES:
        rs = res[m] = {}
        try:
            for h in ("1st", "2nd"):
                f = f"ball/{m}_{h}_ball.npz"
                if f in files:
                    d = np.load(B.fetch(f), allow_pickle=True)
                    rs[f"ball_npz_{h}"] = {k: {"shape": list(d[k].shape), "dtype": str(d[k].dtype), "head": np.asarray(d[k][:5]).tolist() if d[k].ndim else d[k].tolist()} for k in d.files}
            pitch, image, src = ST.load_keypoints(B.fetch(B.find(files, m, "keypoints", ".json")), m, os.path.join(B.CORR, f"{m}_keypoints.json"))
            per = ST.parse_xml(B.fetch(B.find(files, m, "tracker_box_data", ".xml")))["FIRST_HALF"]
            vp = B.fetch(B.find(files, m, "panorama_1st_half", ".mp4"))
            bg, tot = background(vp, 40); H, W = bg.shape[0] * 2, bg.shape[1] * 2
            cal = ST.calibrate(pitch, image, W, H); off0 = int(per["frames"][0]); fidx = {int(f): i for i, f in enumerate(per["frames"])}
            vfs = np.linspace(tot * 0.1, tot * 0.9, NF).astype(int); imgs = B.read_frames(vp, list(vfs)); masks = {k: fg_mask(v, bg) for k, v in imgs.items()}
            curve = []
            for d in range(-SPAN, SPAN + 1, STEP):
                n = k = 0
                for vf, mk in masks.items():
                    i = fidx.get(int(vf) + off0 + d)
                    if i is None or not len(per["players"][i]): continue
                    a, b = hits(mk, ST.project(cal, ST.to_metres(per["players"][i], m, "player"))); n += a; k += b
                curve.append([d, round(n / max(k, 1), 3), k])
            best = max(curve, key=lambda c: c[1]); rs.update(file_offset=off0, video_frames=tot, xml_frames=len(per["frames"]), rms=round(cal["rms"], 2),
                                                          hit_at_file_offset=[c for c in curve if c[0] == 0][0], best=best, curve=curve)
            # flipped-y check for players at the best offset (is the per-match flip right?)
            n = k = 0
            for vf, mk in masks.items():
                i = fidx.get(int(vf) + off0 + best[0])
                if i is None or not len(per["players"][i]): continue
                p = per["players"][i].copy(); p[:, 1] = 1 - p[:, 1]
                a, b = hits(mk, ST.project(cal, ST.to_metres(p, m, "player"))); n += a; k += b
            rs["hit_best_with_y_mirrored"] = round(n / max(k, 1), 3)
            # pictures: one frame at the file offset (red) and at the best offset (green)
            vf = int(vfs[len(vfs) // 2]); img = imgs[vf].copy()
            for d, col in ((0, (0, 0, 255)), (best[0], (0, 255, 0))):
                i = fidx.get(vf + off0 + d)
                if i is None: continue
                for q in ST.project(cal, ST.to_metres(per["players"][i], m, "player")): cv2.circle(img, (int(q[0]), int(q[1])), 12, col, 3)
                b = per["ball"][i]
                if np.isfinite(b).all():
                    q = ST.project(cal, ST.to_metres(b, m, "ball"))[0]; cv2.drawMarker(img, (int(q[0]), int(q[1])), (0, 255, 255) if d else (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
            s = 1600 / img.shape[1]; cv2.imwrite(f"{OUT}/sync_{m}.jpg", cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
            # 2x zoom crops around the ball at the best offset, 8 frames
            tiles = []
            for vf in list(vfs)[:8]:
                i = fidx.get(int(vf) + off0 + best[0])
                if i is None or not np.isfinite(per["ball"][i]).all(): continue
                q = ST.project(cal, ST.to_metres(per["ball"][i], m, "ball"))[0]; c = ST.crop(imgs[int(vf)], q, 120)
                c = cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST); cv2.drawMarker(c, (120, 120), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 10, 1); tiles.append(c)
            if tiles: cv2.imwrite(f"{OUT}/sync_ball_{m}.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
            B.drop(vp); log(f"{m}: file offset {off0} hit {rs['hit_at_file_offset'][1]}, best shift {best[0]} hit {best[1]}, y mirrored {rs['hit_best_with_y_mirrored']}")
        except Exception as e:
            rs["error"] = repr(e); log(f"{m}: {e!r}")
        json.dump(res, open(f"{OUT}/sync.json", "w"), indent=1)

if __name__ == "__main__":
    main()
