"""N0-21e (21 Oct, free runner): the app's RF-DETR ball finder (ipanema/ballrf.py, B7) on the key frames of the two app
test clips, guesses merged the old way and with the tile-edge rule (IPANEMA_BALL_EDGE=1: guesses whose box touches a
tile's inner side left out). One model pass per frame, both merges from the same tile guesses.
Keys: SFK-BP 34 checked balls + B4 ball moments, AIK 39 checked balls. Per frame: is a guess still on the key ball
(<= 30 px), is the top guess on it, which guesses the rule drops. Writes results/qa/n0-21e/edge_frames.json and a
picture sheet of every dropped guess (cyan = tile sides, green ring = the key ball, red ring = the dropped guess).
    R2_PUBLIC_URL=... python tools/n0_21e_balledge.py
    dry run: LOCAL_CLIP_SFK=a.mp4 LOCAL_CLIP_AIK=b.mp4 MAX_KEYS=3 FAKE_MODEL=1 python tools/n0_21e_balledge.py out_dir"""
import os, sys, json, urllib.request, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from ipanema import ballrf as BR

SFK, AIK = "SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"
HIT = 30.0


def keys(max_keys=0):
    """{clip: {frame: (x, y)}} - SFK 34 + B4 'ball' moments, AIK 39"""
    s = {int(k): tuple(v[:2]) for k, v in json.load(open(f"results/volume/reference/{SFK}/ball_gt.json")).items() if v}
    for m in json.load(open("results/ball/b4/key.json"))["moments"]:
        if m["verdict"] == "ball": s.setdefault(int(m["frame"]), (m["x"], m["y"]))
    a = {int(k): tuple(v[:2]) for k, v in json.load(open(f"reference/{AIK}/ball_gt.json")).items() if v}
    out = {SFK: dict(sorted(s.items())), AIK: dict(sorted(a.items()))}
    if max_keys: out = {c: dict(list(v.items())[:max_keys]) for c, v in out.items()}
    return out


def fetch(clip):
    loc = os.environ.get("LOCAL_CLIP_SFK" if clip == SFK else "LOCAL_CLIP_AIK")
    if loc: return loc
    dst = f"/tmp/n0_21e_{clip}.mp4"
    if os.path.exists(dst): return dst
    url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{clip}/video.mp4"; print("downloading", clip, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    return dst


def read_frames(video, want):
    cap = cv2.VideoCapture(video); i = 0; last = max(want) if want else -1; out = {}
    while i <= last:
        if i in want:
            ok, fr = cap.read()
            if not ok: break
            out[i] = fr if fr.shape[:2] == (1080, 1920) else cv2.resize(fr, (1920, 1080))
        elif not cap.grab(): break
        i += 1
    cap.release(); return out


def on(gs, xy, k=None):
    gs = gs if k is None else gs[:k]
    return any(np.hypot(g[0] - xy[0], g[1] - xy[1]) <= HIT for g in gs)


def compare(raw, xy):
    """one frame's tile guesses + the key ball -> row"""
    old, new = BR.merge(raw, False), BR.merge(raw, True)
    sn = set(new); dropped = [g for g in old if g not in sn]
    so = set(old); surfaced = [g for g in new if g not in so]
    d = lambda g: round(float(np.hypot(g[0] - xy[0], g[1] - xy[1])), 1)
    return {"ball": [round(xy[0], 1), round(xy[1], 1)], "tile_guesses": len(raw), "cut": int(sum(z[3] for z in raw)),
            "old_n": len(old), "new_n": len(new), "old_any": on(old, xy), "new_any": on(new, xy), "old_top1": on(old, xy, 1),
            "new_top1": on(new, xy, 1), "old_top3": on(old, xy, 3), "new_top3": on(new, xy, 3),
            "dropped": [[round(g[0], 1), round(g[1], 1), round(g[2], 3), d(g)] for g in dropped],
            "surfaced": [[round(g[0], 1), round(g[1], 1), round(g[2], 3), d(g)] for g in surfaced]}


def cell(fr, g, xy, label, half=48, scale=2):
    x, y = int(round(g[0])), int(round(g[1])); p = cv2.copyMakeBorder(fr, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    c = p[y:y + 2 * half, x:x + 2 * half].copy(); c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ox, oy = x - half, y - half
    for tx, ty in BR.tiles(1920, 1080):                                     # tile sides inside the cell
        for v in (tx, tx + BR.CROP):
            if 0 < v < 1920 and 0 <= v - ox < 2 * half: cv2.line(c, ((v - ox) * scale, 0), ((v - ox) * scale, 10), (255, 255, 0), 1); cv2.line(c, ((v - ox) * scale, c.shape[0] - 11), ((v - ox) * scale, c.shape[0] - 1), (255, 255, 0), 1)
        for v in (ty, ty + BR.CROP):
            if 0 < v < 1080 and 0 <= v - oy < 2 * half: cv2.line(c, (0, (v - oy) * scale), (10, (v - oy) * scale), (255, 255, 0), 1); cv2.line(c, (c.shape[1] - 11, (v - oy) * scale), (c.shape[1] - 1, (v - oy) * scale), (255, 255, 0), 1)
    cv2.circle(c, (half * scale, half * scale), 16, (0, 0, 255), 1)
    bx, by = (xy[0] - ox) * scale, (xy[1] - oy) * scale
    if 0 <= bx < c.shape[1] and 0 <= by < c.shape[0]: cv2.circle(c, (int(bx), int(by)), 20, (0, 255, 0), 1)
    cv2.rectangle(c, (0, 0), (c.shape[1], 14), (0, 0, 0), -1); cv2.putText(c, label, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return c


def sheet(cells, path, cols=8):
    if not cells: return
    while len(cells) % cols: cells.append(np.zeros_like(cells[0]))
    cv2.imwrite(path, np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


class Fake:
    """stand-in detector for the dry run: one box cut by every tile's right side, one whole box in the middle"""
    def predict(self, crops, threshold):
        class D: pass
        out = []
        for _ in crops:
            d = D(); d.xyxy = np.array([[630.0, 300, 640, 310], [300, 300, 312, 312]]); d.confidence = np.array([0.5, 0.3]); out.append(d)
        return out


def main(out, max_keys=0, model=None):
    t0 = time.time(); model = model or (Fake() if os.environ.get("FAKE_MODEL") == "1" else BR.load(BR.WEIGHTS))
    K = keys(max_keys); rows = {}; cells = []; tot = {}
    for clip, kk in K.items():
        frs = read_frames(fetch(clip), set(kk)); rows[clip] = {}
        print(f"{clip}: {len(frs)}/{len(kk)} key frames read ({time.time() - t0:.0f} s)", flush=True)
        fl = sorted(frs)
        for b in range(0, len(fl), 4):
            raw = []; BR.detect_many(model, [frs[f] for f in fl[b:b + 4]], edge=False, raw=raw)
            for f, r in zip(fl[b:b + 4], raw):
                row = compare(r, kk[f]); rows[clip][f] = row
                for j, g in enumerate(row["dropped"]):
                    cells.append(cell(frs[f], g, kk[f], f"{'S' if clip == SFK else 'A'}{f} c{g[2]:.2f} d{g[3]:.0f}"))
            if b % 40 == 0: print(f"  {b}/{len(fl)} ({time.time() - t0:.0f} s)", flush=True)
        R = list(rows[clip].values())
        tot[clip] = {"frames": len(R), **{k: int(sum(r[k] for r in R)) for k in ("tile_guesses", "cut", "old_n", "new_n", "old_any", "new_any", "old_top1", "new_top1", "old_top3", "new_top3")},
                     "dropped": int(sum(len(r["dropped"]) for r in R)), "dropped_on_ball": int(sum(any(g[3] <= HIT for g in r["dropped"]) for r in R)),
                     "lost_ball": [f for f, r in rows[clip].items() if r["old_any"] and not r["new_any"]],
                     "top1_changed": {str(f): [r["old_top1"], r["new_top1"]] for f, r in rows[clip].items() if r["old_top1"] != r["new_top1"]}}
        print(clip, json.dumps(tot[clip]), flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump({"totals": tot, "minutes": round((time.time() - t0) / 60, 1), "frames": {c: {str(f): r for f, r in v.items()} for c, v in rows.items()}}, open(f"{out}/edge_frames.json", "w"))
    for s in range(0, len(cells), 96): sheet(cells[s:s + 96], f"{out}/dropped_{s // 96:02d}.jpg")
    return tot


if __name__ == "__main__":
    if os.environ.get("FAKE_MODEL") != "1":
        try: import rfdetr  # noqa
        except ImportError:
            import subprocess
            subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
    print(json.dumps(main(sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-21e", int(os.environ.get("MAX_KEYS", "0")))))
