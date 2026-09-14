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
    return [r for r in res if r]
