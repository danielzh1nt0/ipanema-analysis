"""Players: detect, track, assign team, project to metres, then clean (mask, keepers, duplicates, id stitching).
Output per[k] = list of rows (id, team 'A'/'B', pos_m (x,y), foot_px (x,y), box xyxy, gk bool)."""
import os, numpy as np
from collections import defaultdict
from .video import frames
from .calibration import to_m


# Veo panorama: players are small (about 20 px tall on the far side), so detect on zoomed tiles.
# Far band (top of the picture) in 4 tiles, near band in 2; fractions of the frame (x0, y0, x1, y1), with overlap.
PANO_TILES = [(0.00, 0.20, 0.30, 0.58), (0.23, 0.20, 0.53, 0.58), (0.47, 0.20, 0.77, 0.58), (0.70, 0.20, 1.00, 0.58),
              (0.00, 0.45, 0.55, 1.00), (0.45, 0.45, 1.00, 1.00)]

def detect_tiled(model, f, conf, tiles):
    """run the detector on each tile, map boxes back to the frame, merge duplicates in the overlaps (NMS)"""
    import supervision as sv
    h, w = f.shape[:2]; boxes, confs, cls = [], [], []; names = None
    for x0, y0, x1, y1 in tiles:
        a, b, c, d = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        res = model(f[b:d, a:c], conf=conf, verbose=False)[0]; names = res.names
        det = sv.Detections.from_ultralytics(res)
        if len(det):
            xy = det.xyxy.copy(); xy[:, [0, 2]] += a; xy[:, [1, 3]] += b
            boxes.append(xy); confs.append(det.confidence); cls.append(det.class_id)
    if not boxes: return sv.Detections.empty(), names or {}
    det = sv.Detections(xyxy=np.vstack(boxes), confidence=np.concatenate(confs), class_id=np.concatenate(cls))
    return det.with_nms(0.5, class_agnostic=True), names

def track(video, weights_player, H, team_model, conf=0.3, log=print, tiles=None):
    import supervision as sv
    from ultralytics import YOLO
    from .video import info
    fps = info(video)["fps"]; model = YOLO(weights_player)
    use_bot = os.environ.get("IPANEMA_TRACKER", "bytetrack") == "botsort"
    tracker = None
    if use_bot:
        try:
            from boxmot import BotSort            # camera-motion compensation for panning footage
            import torch, pathlib
            tracker = BotSort(reid_weights=pathlib.Path("osnet_x0_25_msmt17.pt"), device=0 if torch.cuda.is_available() else "cpu", half=False, with_reid=False)
            log("tracking: BoT-SORT with camera-motion compensation")
        except Exception as e: log(f"tracking: BoT-SORT unavailable ({e!r}); using ByteTrack"); tracker = None
    if tracker is None:
        tracker = sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=90, minimum_matching_threshold=0.8, track_activation_threshold=0.25)
    votes = {}; per = {}
    for k, f in frames(video):
        if tiles:
            det, names = detect_tiled(model, f, conf, tiles)
        else:
            res = model(f, conf=conf, verbose=False)[0]; det = sv.Detections.from_ultralytics(res).with_nms(0.5, class_agnostic=True); names = res.names
        ref_id = next((i for i, nm in names.items() if "referee" in nm.lower()), None)
        if ref_id is not None: det = det[det.class_id != ref_id]
        if hasattr(tracker, "update_with_detections"): det = tracker.update_with_detections(det)
        else:
            import numpy as _np
            arr = _np.c_[det.xyxy, det.confidence if det.confidence is not None else _np.ones(len(det)), _np.zeros(len(det))]
            out = tracker.update(arr, f)          # BoT-SORT: (N, 7) x1 y1 x2 y2 id conf cls
            if len(out):
                det = sv.Detections(xyxy=out[:, :4].astype(float), confidence=out[:, 5].astype(float),
                                    class_id=out[:, 6].astype(int), tracker_id=out[:, 4].astype(int))
            else: det = sv.Detections.empty()
        rows = []
        if len(det):
            feet = np.c_[(det.xyxy[:, 0] + det.xyxy[:, 2]) / 2, det.xyxy[:, 3]]; m = to_m(H[k], feet)
            labs = team_model.predict_batch(f, det.xyxy)
            for j in range(len(det)):
                if labs[j] == "R": continue                                   # referee / bib: not a player
                tid = int(det.tracker_id[j]) if det.tracker_id is not None else -1
                v = votes.setdefault(tid, []); v.append(labs[j]); del v[:-25]
                rows.append([tid, max(set(v), key=v.count), m[j].astype(float), feet[j].astype(float), det.xyxy[j].astype(float), False])
        per[k] = rows
        if k % 500 == 0: log(f"  tracking frame {k}")
    return per, fps

def clean(per, L, W, fps, log=print):
    n = len(per)
    # 1) drop off-pitch
    dropped = 0
    for k in per:
        keep = [r for r in per[k] if 0 <= r[2][0] <= L and 0 <= r[2][1] <= W]; dropped += len(per[k]) - len(keep); per[k] = keep
    # 2) duplicates: same team within 1 m in one frame -> keep the bigger box
    merged = 0
    for k in per:
        keep = []
        for r in sorted(per[k], key=lambda r: -(r[4][2] - r[4][0]) * (r[4][3] - r[4][1])):
            if any(r[1] == q[1] and np.linalg.norm(r[2] - q[2]) < 1.0 for q in keep): merged += 1; continue
            keep.append(r)
        per[k] = keep
    # 3) stitch ids: a track starting within 2.5 s of another's end, same team, nearby -> same player
    segs = {}
    for k in range(n):
        for r in per[k]:
            sg = segs.setdefault(r[0], {"team": r[1], "start": k, "end": k, "p0": r[2], "p1": r[2]}); sg["end"] = k; sg["p1"] = r[2]
    remap = _stitch(segs, fps)
    def root(t):
        while t in remap: t = remap[t]
        return t
    for k in range(n): 
        for r in per[k]: r[0] = root(r[0])
    # 4) keepers: mostly inside a goal zone; team = the team whose outfield players are, on average, nearer that goal... use zone + majority of nearby team
    zone = defaultdict(lambda: [0, 0, 0])
    for k in range(n):
        for r in per[k]:
            z = zone[r[0]]; z[0] += 1
            if r[2][0] < 8 and abs(r[2][1] - W / 2) < 20: z[1] += 1
            if r[2][0] > L - 8 and abs(r[2][1] - W / 2) < 20: z[2] += 1
    # a keeper is whoever is in a goalmouth for most of the frames in which that goalmouth is on screen (follow-cam hides the goals often)
    keepers = {tid: ("left" if z[1] >= z[2] else "right") for tid, z in zone.items() if z[0] >= 10 and max(z[1], z[2]) >= 0.4 * z[0] and max(z[1], z[2]) >= 30}
    # which team defends which goal: the team with the lower median x over the clip defends left (works for clips showing both ends; else falls back to majority near goal)
    med = {}
    for tm in ("A", "B"):
        xs = [r[2][0] for k in range(n) for r in per[k] if r[1] == tm and r[0] not in keepers]; med[tm] = float(np.median(xs)) if xs else L / 2
    left_team = "A" if med["A"] < med["B"] else "B"; right_team = "B" if left_team == "A" else "A"
    for k in range(n):
        for r in per[k]:
            if r[0] in keepers: r[1] = left_team if keepers[r[0]] == "left" else right_team; r[5] = True
    log(f"clean: dropped {dropped} off-pitch, merged {merged} duplicates, stitched {len(remap)} ids, keepers {list(keepers)} -> {left_team} defends left")
    counts = [sum(1 for r in per[k] if r[1] == "A") for k in range(n)], [sum(1 for r in per[k] if r[1] == "B") for k in range(n)]
    log(f"clean: players per frame median A {np.median(counts[0]):.0f} / B {np.median(counts[1]):.0f}, max A {max(counts[0])} / B {max(counts[1])}")
    return per, {"left_team": left_team, "keepers": list(keepers)}


def _stitch(segs, fps, max_gap_s=2.5):
    """Join broken tracks: in order of start, a track joins the player whose latest track ended 0 < gap <= 2.5 s before it,
    same team, within 2 + 6*gap metres (nearest wins; ties go to the earlier-seen track). Only players that ended in that
    window are examined, so this is linear in the number of tracks (the old all-pairs loop was cubic)."""
    order = {t: i for i, t in enumerate(segs)}
    remap = {}; g_end = {}; g_latest = {}; by_end = {}
    win = max_gap_s * fps
    def put(r, end, latest):
        old = g_end.get(r)
        if old is not None: by_end[old].discard(r)
        g_end[r] = end; g_latest[r] = latest; by_end.setdefault(end, set()).add(r)
    for tid in sorted(segs, key=lambda t: segs[t]["start"]):
        B = segs[tid]; st = B["start"]; cands = []
        for e in range(max(0, int(np.floor(st - win))), st):
            if (st - e) / fps > max_gap_s: continue
            for r in by_end.get(e, ()):
                cands.extend((order[a], a, r, (st - e) / fps) for a in g_latest[r])
        best = None
        for _, a, r, gap in sorted(cands):
            A = segs[a]
            if A["team"] != B["team"]: continue
            dist = np.linalg.norm(A["p1"] - B["p0"])
            if dist <= 2.0 + 6.0 * gap and (best is None or dist < best[1]): best = (r, dist)
        if best:
            r = best[0]; remap[tid] = r
            if B["end"] > g_end[r]: put(r, B["end"], [tid])
            elif B["end"] == g_end[r]: g_latest[r].append(tid)
        else: put(tid, B["end"], [tid])
    return remap


def reposition(per, H):
    """recompute each detection's pitch position (metres) from its foot point in the picture with a new calibration"""
    from .calibration import to_m
    out = {}
    for k, rows in per.items():
        if not rows: out[k] = []; continue
        m = to_m(H[k], np.array([r[3] for r in rows], np.float32))
        out[k] = [[r[0], r[1], m[j].astype(float), r[3], r[4], r[5]] for j, r in enumerate(rows)]
    return out
