"""P2 (4 Oct): table from the 20-s tracking pieces in results/qa/p2/<match>_<start>/ (tools/p2_track.py).
Per piece: players seen per frame (dark/light), median seconds a player stays tracked, track ids, and on the 8 key
frames the detector's people that no tracked player sits on ('dropped' = referee / staff / spectators rightly, or a
missed player). Writes results/qa/p2/table.md + table.json and, per piece, dropped_sheet.jpg (crops of the dropped
people, numbered) to grade by eye.   python tools/p2_table.py [dir]"""
import os, sys, json, glob, gzip
import numpy as np

def feet(b): return ((b[0] + b[2]) / 2.0, b[3])

def dropped(dets, rows, tol=0.6):
    """dets: [[x1,y1,x2,y2]], rows: [[id, team, [fx, fy] | None, filled]] -> indices of detections with no tracked
    (non-filled) player whose feet lie within tol x box height of the detection's feet."""
    pts = [r[2] for r in rows if r[2] is not None and not r[3]]
    out = []
    for i, b in enumerate(dets):
        fx, fy = feet(b); h = max(b[3] - b[1], 1.0)
        if not any(np.hypot(p[0] - fx, p[1] - fy) <= tol * h for p in pts): out.append(i)
    return out

def piece_stats(d):
    s = json.load(open(f"{d}/summary.json"))["new, RF-DETR"]
    st = {"seen_dark": s["observed_per_frame"]["dark"], "seen_light": s["observed_per_frame"]["light"],
          "track_s_dark": s["median_track_s"]["dark"], "track_s_light": s["median_track_s"]["light"], "tracks": s["tracks"],
          "minutes": s.get("minutes"), "filled_rows": s.get("filled_rows")}
    if os.path.exists(f"{d}/rows_all.json.gz"):
        a = json.load(gzip.open(f"{d}/rows_all.json.gz", "rt")); rows = a["rows"]["new, RF-DETR"]
        n = sum(1 for rs in rows.values() for r in rs if not r[3])
        st["player_s"] = round(n / a["fps"], 1)                           # player-seconds tracked in the piece (both teams, not filled)
    if os.path.exists(f"{d}/keydets.json"):
        kd = json.load(open(f"{d}/keydets.json")); kr = json.load(open(f"{d}/rows_keyframes.json"))["new, RF-DETR"]
        drop = {k: dropped(v, kr.get(k, [])) for k, v in kd.items()}
        st["det_people"] = sum(map(len, kd.values())); st["dropped"] = sum(map(len, drop.values())); st["_drop"] = drop
    return st

def sheet(d, drop, size=96, cols=12):
    import cv2
    kd = json.load(open(f"{d}/keydets.json")); tiles = []
    for k, idx in sorted(drop.items(), key=lambda x: int(x[0])):
        f = cv2.imread(f"{d}/raw_{int(k):04d}.jpg")
        if f is None: continue
        for i in idx:
            x1, y1, x2, y2 = [int(v) for v in kd[k][i]]; h = y2 - y1; pad = max(h // 3, 6)
            c = f[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad]
            if c.size == 0: continue
            t = cv2.resize(c, (size, int(size * 1.5))); n = len(tiles)
            cv2.putText(t, str(n), (2, 14), 0, 0.45, (0, 0, 0), 3); cv2.putText(t, str(n), (2, 14), 0, 0.45, (255, 255, 255), 1)
            tiles.append(t)
    if not tiles: return 0
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{d}/dropped_sheet.jpg", np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
    return len(tiles)

def main(root="results/qa/p2"):
    res = {}
    for d in sorted(glob.glob(f"{root}/*/")):
        if not os.path.exists(f"{d}summary.json"): continue
        st = piece_stats(d.rstrip("/")); drop = st.pop("_drop", None)
        if drop: sheet(d.rstrip("/"), drop)
        res[os.path.basename(d.rstrip("/"))] = st
    hdr = "| piece | seen dark/light per frame | player-seconds tracked | median s tracked dark/light | track ids | detector people (8 frames) | dropped by tracking |"
    lines = [hdr, "|---|---|---|---|---|---|---|"]
    for k, s in res.items():
        lines.append(f"| {k} | {s['seen_dark']:g} / {s['seen_light']:g} | {s.get('player_s', '-')} | {s['track_s_dark']} / {s['track_s_light']} | {s['tracks']} | {s.get('det_people', '-')} | {s.get('dropped', '-')} |")
    json.dump(res, open(f"{root}/table.json", "w"), indent=1); open(f"{root}/table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); return res

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/qa/p2")
