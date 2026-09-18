"""Players: detect, track, assign team, project to metres, then clean (mask, keepers, duplicates, id stitching).
Output per[k] = list of rows (id, team 'A'/'B', pos_m (x,y), foot_px (x,y), box xyxy, gk bool)."""
import numpy as np
from collections import defaultdict
from .video import frames
from .calibration import to_m

def track(video, weights_player, H, team_model, conf=0.3, log=print):
    import supervision as sv
    from ultralytics import YOLO
    from .video import info
    fps = info(video)["fps"]; model = YOLO(weights_player)
    tracker = sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=90, minimum_matching_threshold=0.8, track_activation_threshold=0.25)
    votes = {}; per = {}
    for k, f in frames(video):
        res = model(f, conf=conf, verbose=False)[0]; det = sv.Detections.from_ultralytics(res).with_nms(0.5, class_agnostic=True)
        ref_id = next((i for i, nm in res.names.items() if "referee" in nm.lower()), None)
        if ref_id is not None: det = det[det.class_id != ref_id]
        det = tracker.update_with_detections(det); rows = []
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
    remap = {}
    def root(t):
        while t in remap: t = remap[t]
        return t
    for tid in sorted(segs, key=lambda t: segs[t]["start"]):
        B = segs[tid]; best = None
        for aid, A in segs.items():
            if aid == tid or A["team"] != B["team"]: continue
            ra = root(aid); Aend = max(segs[q]["end"] for q in segs if root(q) == ra); gap = (B["start"] - Aend) / fps
            if not (0 < gap <= 2.5): continue
            dist = np.linalg.norm(A["p1"] - B["p0"]) if A["end"] == Aend else 99
            if dist <= 2.0 + 6.0 * gap and (best is None or dist < best[1]): best = (ra, dist)
        if best: remap[tid] = best[0]
    for k in range(n): 
        for r in per[k]: r[0] = root(r[0])
    # 4) keepers: mostly inside a goal zone; team = the team whose outfield players are, on average, nearer that goal... use zone + majority of nearby team
    zone = defaultdict(lambda: [0, 0, 0])
    for k in range(n):
        for r in per[k]:
            z = zone[r[0]]; z[0] += 1
            if r[2][0] < 8 and abs(r[2][1] - W / 2) < 20: z[1] += 1
            if r[2][0] > L - 8 and abs(r[2][1] - W / 2) < 20: z[2] += 1
    keepers = {tid: ("left" if z[1] / z[0] > 0.7 else "right") for tid, z in zone.items() if z[0] >= 10 and max(z[1], z[2]) / z[0] > 0.7}
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
