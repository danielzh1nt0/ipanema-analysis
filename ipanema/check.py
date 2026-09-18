"""ipanema.check: score runs against reference labels. reference/<clip>/ball_gt.json (from label.py) and events_gt_<clip>.json (from event_labeller.html)."""
import os, json, glob, numpy as np

def _match(pred_ts, gt_ts, tol):
    """greedy one-to-one matching within tol seconds -> (hits, misses, false)"""
    pred = sorted(pred_ts); gt = sorted(gt_ts); used = set(); hits = 0
    for g in gt:
        best = None
        for i, p in enumerate(pred):
            if i in used: continue
            if abs(p - g) <= tol and (best is None or abs(p - g) < abs(pred[best] - g)): best = i
        if best is not None: used.add(best); hits += 1
    return hits, len(gt) - hits, len(pred) - hits

def check_clip(root, match_id, tol_s=2.0, log=print):
    ref = os.path.join(root, "reference", match_id); run = os.path.join(root, "runs", "matches", match_id)
    if not os.path.exists(f"{run}/match_data.json"): log(f"{match_id}: no run found"); return None
    md = json.load(open(f"{run}/match_data.json")); st = json.load(open(f"{run}/stats.json")); summ = json.load(open(f"{run}/summary.json")) if os.path.exists(f"{run}/summary.json") else {}
    out = {"match_id": match_id}
    # ---- ball ----
    gt_path = f"{ref}/ball_gt.json"
    if os.path.exists(gt_path):
        gt = {int(k): v for k, v in json.load(open(gt_path)).items()}; fps = md["fps"]; ok = tot = 0
        for i, g in gt.items():
            if g is None: continue
            tot += 1; fr = md["frames"][i] if i < len(md["frames"]) else None
            if fr and fr.get("ball") and np.hypot(fr["ball"]["px"][0] - g[0], fr["ball"]["px"][1] - g[1]) <= 30: ok += 1
        out["ball"] = f"{ok}/{tot}"; out["ball_pct"] = round(100 * ok / tot) if tot else None
    # ---- events ----
    ev_files = glob.glob(f"{ref}/events_gt*.json") + glob.glob(f"{root}/reference/events_gt_{match_id}.json")
    if ev_files:
        gt = json.load(open(ev_files[0])); gte = gt["events"]
        # labels with A/B the wrong way round: if every matched turnover disagrees on the loser, flip the label teams
        agree = dis = 0
        for e in [e for e in gte if e["type"] == "turnover" and e.get("team")]:
            near = [t for t in md["turnovers"] if abs(t["t"] - e["t"]) <= tol_s]
            if near: agree += near[0]["lost_by"] == e["team"]; dis += near[0]["lost_by"] != e["team"]
        if dis >= 2 and agree == 0:
            out["label_teams"] = "SWAPPED in reference file (A<->B) — scored with teams flipped"
            for e in gte:
                if e.get("team") in ("A", "B"): e["team"] = "B" if e["team"] == "A" else "A"
            if gt.get("attack_right_A") is not None: gt["attack_right_A"] = not gt["attack_right_A"]
        pred_to = [t["t"] for t in md["turnovers"]]; gt_to = [e["t"] for e in gte if e["type"] == "turnover"]
        h, m, f = _match(pred_to, gt_to, tol_s); out["turnovers"] = f"{h}/{len(gt_to)} found, {f} extra (pipeline {len(pred_to)}, truth {len(gt_to)})"
        out["turnover_recall"] = round(100 * h / len(gt_to)) if gt_to else None; out["turnover_precision"] = round(100 * h / len(pred_to)) if pred_to else None
        shots = st.get("metrics", {}).get("shots", []); pred_sh = [s["t"] for s in shots]; gt_sh = [e["t"] for e in gte if e["type"] in ("shot", "goal")]
        h, m, f = _match(pred_sh, gt_sh, 3.0); out["shots"] = f"{h}/{len(gt_sh)} found, {f} extra"
        goals_pred = [s["t"] for s in shots if s.get("goal")]; goals_gt = [e["t"] for e in gte if e["type"] == "goal"]
        out["goals"] = f"{_match(goals_pred, goals_gt, 5.0)[0]}/{len(goals_gt)}"
        if gt.get("attack_right_A") is not None:
            out["direction"] = "from labels" if md["attack_right"]["A"] == gt["attack_right_A"] else "WRONG"
        # side-attributed turnovers: does the pipeline agree who lost it?
        agree = tot = 0
        for e in [e for e in gte if e["type"] == "turnover" and e.get("team")]:
            near = [t for t in md["turnovers"] if abs(t["t"] - e["t"]) <= tol_s]
            if near: tot += 1; agree += near[0]["lost_by"] == e["team"]
        out["turnover_team_agreement"] = f"{agree}/{tot}" if tot else None
    out["players_per_frame"] = summ.get("players_per_frame_median"); out["loose_pct"] = summ.get("loose_pct"); out["passes"] = summ.get("passes")
    log(f"--- {match_id} ---"); [log(f"  {k:26s} {v}") for k, v in out.items() if k != "match_id"]
    return out

def check_all(root, log=print):
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "runs", "matches", "*")) if os.path.isdir(p))
    res = [check_clip(root, m, log=log) for m in ids]
    # diagnostics: per-turnover tables, logged and written next to the results if the repo checkout is present
    out_dir = next((d for d in ("/content/ipanema-analysis/results",) if os.path.isdir(os.path.dirname(d))), None)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    for m in ids:
        try:
            tbl = turnover_table(root, m)
            if tbl:
                log(f"\n--- turnovers {m} ---"); [log(l) for l in tbl.splitlines()]
                if out_dir: open(os.path.join(out_dir, f"turnovers_{m}.txt"), "w").write(tbl)
        except Exception as e: log(f"turnover table failed for {m}: {e!r}")
    try: inventory(root, log=log)
    except Exception as e: log(f"inventory failed: {e!r}")
    try: publish_frames(root, log=log)
    except Exception as e: log(f"publish_frames failed: {e!r}")
    try: publish_mosaic(root, log=log)
    except Exception as e: log(f"publish_mosaic failed: {e!r}")
    try:
        if not os.path.isdir("/content/ipanema-analysis/results/ballcands_SFKBP1109_s1200"): publish_ball_candidates(root, log=log)
    except Exception as e: log(f"publish_ball_candidates failed: {e!r}")
    return [r for r in res if r]


def turnover_table(root, match_id, tol_s=2.0):
    """one line per detected turnover with context, for diagnosis"""
    run = os.path.join(root, "runs", "matches", match_id)
    if not os.path.exists(f"{run}/match_data.json"): return ""
    md = json.load(open(f"{run}/match_data.json")); fps = md["fps"]; fr = md["frames"]
    evf = glob.glob(f"{root}/reference/events_gt_{match_id}.json"); gt = json.load(open(evf[0]))["events"] if evf else []
    gt_to = [e for e in gt if e["type"] == "turnover"]
    lines = [f"truth turnovers: {[(e['t'], e.get('team')) for e in gt_to]}", f"{'#':>2} {'t':>7} {'lost':>4}>{'won':<4} {'hold':>5} {'prev':>5} {'speed':>6} {'dist':>5} {'nopp':>4} {'ballpx':>14}  note"]
    for i, t in enumerate(md["turnovers"]):
        k = int(round(t["t"] * fps)); f = fr[min(k, len(fr) - 1)]
        sp = []
        for j in range(max(0, k - 5), min(len(fr) - 1, k + 5)):
            a, b = fr[j].get("ball"), fr[j + 1].get("ball")
            if a and b and a.get("m") and b.get("m"): sp.append(np.hypot(b["m"][0] - a["m"][0], b["m"][1] - a["m"][1]) * fps)
        speed = round(float(np.median(sp)), 1) if sp else None
        hold = 0
        for j in range(k, min(len(fr), k + int(6 * fps))):
            if fr[j]["possession"] == t["won_by"]: hold += 1
            elif fr[j]["possession"] == t["lost_by"]: break
        prev = 0
        for j in range(k - 1, max(0, k - int(10 * fps)), -1):
            if fr[j]["possession"] == t["lost_by"]: prev += 1
            elif fr[j]["possession"] == t["won_by"]: break
        near = [e for e in gt_to if abs(e["t"] - t["t"]) <= tol_s]
        note = ("TRUTH " + str(near[0].get("team")) + " lost") if near else ""
        bp = f.get("ball", {}).get("px") if f.get("ball") else None
        lines.append(f"{i+1:>2} {t['t']:>7.1f} {t['lost_by']:>4}>{t['won_by']:<4} {hold/fps:>5.1f} {prev/fps:>5.1f} {str(speed):>6} {str(f.get('pressure_m')):>5} {str(f.get('near_opps')):>4} {str(bp):>14}  {note}")
    # truth turnovers the pipeline missed: what did it think possession was around then?
    for e in gt_to:
        if not any(abs(t["t"] - e["t"]) <= tol_s for t in md["turnovers"]):
            k = int(round(e["t"] * fps)); seq = "".join((fr[j]["possession"] or "-")[0] for j in range(max(0, k - int(4 * fps)), min(len(fr), k + int(4 * fps)), max(1, int(fps / 5))))
            lines.append(f"MISSED truth turnover at {e['t']} ({e.get('team')} lost): possession -4s..+4s = {seq}")
    return "\n".join(lines)


def inventory(root, log=print):
    """publish what sits in videos/<subfolder>/ (SFK matches): names, sizes, durations — so labels can be built from the Veo filenames"""
    import subprocess
    vids = os.path.join(root, "videos"); out = []
    for sub in sorted(d for d in os.listdir(vids) if os.path.isdir(os.path.join(vids, d))):
        out.append(f"== {sub} ==")
        for dp, dn, fn in os.walk(os.path.join(vids, sub)):
            for f in sorted(fn):
                p = os.path.join(dp, f); rel = os.path.relpath(p, os.path.join(vids, sub)); size = os.path.getsize(p) / 1e6; dur = ""
                if f.lower().endswith((".mp4", ".mov", ".mkv")):
                    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True)
                    try: dur = f"{float(r.stdout.strip()):.1f}s"
                    except Exception: dur = "?"
                out.append(f"  {rel}  {size:.1f} MB  {dur}")
    txt = "\n".join(out); log(txt)
    d = "/content/ipanema-analysis/results"
    if os.path.isdir(os.path.dirname(d)): os.makedirs(d, exist_ok=True); open(os.path.join(d, "inventory.txt"), "w").write(txt)
    return txt


def publish_frames(root, n_frames=40, log=print, extra=("SFKBP1109_s1200.mp4",)):
    """write sampled frames of each full game to the results folder so they can be labelled off-Colab"""
    import cv2
    vids = os.path.join(root, "videos"); out_root = "/content/ipanema-analysis/results"
    if not os.path.isdir(os.path.dirname(out_root)): return
    targets = [(sub, max([os.path.join(vids, sub, f) for f in os.listdir(os.path.join(vids, sub)) if f.lower().endswith((".mp4", ".mov", ".mkv"))] or [None], key=lambda p: os.path.getsize(p) if p else 0)) for sub in sorted(d for d in os.listdir(vids) if os.path.isdir(os.path.join(vids, d)))]
    targets += [(os.path.splitext(e)[0], os.path.join(vids, e)) for e in extra if os.path.exists(os.path.join(vids, e))]
    for sub, full in targets:
        if not full: continue
        od = os.path.join(out_root, f"frames_{sub}"); os.makedirs(od, exist_ok=True)
        if len(glob.glob(f"{od}/*.jpg")) >= n_frames: continue
        cap = cv2.VideoCapture(full); n = int(cap.get(7))
        for k in range(n_frames):
            i = int(round(k * (n - 1) / (n_frames - 1))); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
            if ok: cv2.imwrite(f"{od}/f{i:07d}.jpg", cv2.resize(f, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 78])
        cap.release(); log(f"published {n_frames} frames of {sub} for labelling")


def publish_mosaic(root, chunk_s=300, n_chunks=6, log=print):
    """publish sharp per-chunk panoramas of each full game (chaining over a whole match drifts)"""
    import cv2, subprocess
    from .mosaic import build
    vids = os.path.join(root, "videos"); out_root = "/content/ipanema-analysis/results"
    if not os.path.isdir(os.path.dirname(out_root)): return
    for sub in sorted(d for d in os.listdir(vids) if os.path.isdir(os.path.join(vids, d))):
        cands = [os.path.join(vids, sub, f) for f in os.listdir(os.path.join(vids, sub)) if f.lower().endswith((".mp4", ".mov", ".mkv"))]
        if not cands: continue
        full = max(cands, key=os.path.getsize)
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", full], capture_output=True, text=True)
        dur = float(r.stdout.strip() or 0)
        for i in range(n_chunks):
            start = int(i * dur / n_chunks)
            dst = os.path.join(out_root, f"mosaic_{sub}_c{i+1}.jpg")
            if os.path.exists(dst): continue
            piece = os.path.join(root, "cache", f"{sub}_chunk{i+1}.mp4"); os.makedirs(os.path.dirname(piece), exist_ok=True)
            if not os.path.exists(piece):
                subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-i", full, "-t", str(chunk_s), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", piece], capture_output=True)
            cache = os.path.join(root, "cache", f"{sub}_mosaic_c{i+1}.pkl")
            m = build(piece, cache, stride=25, canvas=(4200, 1500), log=log)
            cv2.imwrite(dst, m["mosaic"], [cv2.IMWRITE_JPEG_QUALITY, 88]); log(f"published mosaic chunk {i+1} for {sub} (from {start}s)")


def publish_ball_candidates(root, clip="SFKBP1109_s1200", n_frames=40, tile=96, zoom=2, log=print):
    """for labelling: zoomed tiles around each ball candidate in sampled frames (pick the tile index that is the ball)"""
    import cv2, pickle, json
    cands_path = os.path.join(root, "cache", clip, "ball_cands.pkl"); video = os.path.join(root, "videos", f"{clip}.mp4")
    if not os.path.exists(cands_path): log("no ball candidates cache"); return
    cands = pickle.load(open(cands_path, "rb")); out_root = "/content/ipanema-analysis/results"; od = os.path.join(out_root, f"ballcands_{clip}"); os.makedirs(od, exist_ok=True)
    cap = cv2.VideoCapture(video); n = int(cap.get(7)); idx = {}
    for j in range(n_frames):
        i = int(round(j * (n - 1) / (n_frames - 1))); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok or not cands.get(i): continue
        cs = sorted(cands[i], key=lambda z: -z[2])[:10]; tiles = []; meta = []
        for t, (x, y, cf) in enumerate(cs):
            x0, y0 = int(max(0, x - tile // 2)), int(max(0, y - tile // 2)); crop = f[y0:y0 + tile, x0:x0 + tile]
            if crop.size == 0: continue
            crop = cv2.resize(crop, (tile * zoom, tile * zoom)); cv2.putText(crop, str(t), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.circle(crop, (tile * zoom // 2, tile * zoom // 2), 14, (0, 0, 255), 1); tiles.append(crop); meta.append([t, float(x), float(y), float(cf)])
        while len(tiles) % 5: tiles.append(np.zeros((tile * zoom, tile * zoom, 3), np.uint8))
        rows = [np.hstack(tiles[k:k + 5]) for k in range(0, len(tiles), 5)]
        cv2.imwrite(os.path.join(od, f"f{i:06d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); idx[i] = meta
    cap.release(); json.dump(idx, open(os.path.join(od, "candidates.json"), "w")); log(f"published ball candidate tiles for {len(idx)} frames")
