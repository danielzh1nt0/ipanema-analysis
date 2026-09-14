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
