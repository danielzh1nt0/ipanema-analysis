"""F1 (16 Oct): grade a Kaggle match export against the by-eye answer keys in reference/<match>/ - no hand tagging, no Modal.
    python tools/export_qa.py <export dir holding matches/<match>/ or the match folder itself> [match_id] [out.json]
Checks (all optional, each skipped when its key is missing):
  ball_key_px.json      {video frame: [x, y]} by-eye ball positions -> the exported ball within 30 px / 60 px (export is 10 frames/s)
  ball_key_graded.json  items with "owner" A/B (who has the ball, by eye) -> the exported possession at that second
  poss_sample.json      items with "owner" A/B -> the same
Prints a plain-words summary and writes qa.json {ball: {hit30, hit60, n}, owner: {agree, n, by_team}, possession_pct, summary}."""
import os, sys, json, glob, bisect

def load_export(folder):
    md = json.load(open(f"{folder}/match_data.json")); frames = list(md.get("frames") or [])
    for ch in md.get("frame_chunks") or []:
        p = f"{folder}/{ch['key']}.json"
        if os.path.exists(p): frames += json.load(open(p))["frames"]
    frames.sort(key=lambda f: f["t"]); return md, frames

def nearest(frames, ts, t):
    i = bisect.bisect_left(ts, t); c = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    j = min(c, key=lambda j: abs(ts[j] - t)); return frames[j] if abs(ts[j] - t) <= 0.2 else None

def grade(folder, ref_dir, log=print):
    md, frames = load_export(folder); ts = [f["t"] for f in frames]; fps = md.get("fps") or 29.97; out = {"n_frames": len(frames)}
    kp = f"{ref_dir}/ball_key_px.json"
    if os.path.exists(kp):
        key = json.load(open(kp)); h30 = h60 = seen = 0
        for fr, (x, y) in key.items():
            f = nearest(frames, ts, int(fr) / fps)
            b = (f or {}).get("ball")
            if not b or b.get("state") == "bridged": continue
            seen += 1; d = ((b["px"][0] - x) ** 2 + (b["px"][1] - y) ** 2) ** 0.5; h30 += d <= 30; h60 += d <= 60
        out["ball"] = {"hit30": h30, "hit60": h60, "seen": seen, "n": len(key)}
        log(f"ball: {h30} of {len(key)} by-eye positions within 30 px ({h60} within 60 px); the export had a seen ball at {seen} of them")
    owners = []
    for name in ("ball_key_graded.json", "poss_sample.json"):
        p = f"{ref_dir}/{name}"
        if os.path.exists(p): owners += [(it["t"], it["owner"]) for it in json.load(open(p)).get("items", []) if it.get("owner") in ("A", "B")]
    if owners:
        agree = 0; n = 0; by = {"A": [0, 0], "B": [0, 0]}
        for t, o in owners:
            f = nearest(frames, ts, t)
            if f is None: continue
            n += 1; got = f.get("possession"); by[o][1] += 1
            if got == o: agree += 1; by[o][0] += 1
        out["owner"] = {"agree": agree, "n": n, "by_team": by}
        log(f"who has the ball: {agree} of {n} by-eye moments agree (A {by['A'][0]}/{by['A'][1]}, B {by['B'][0]}/{by['B'][1]})")
    sp = f"{folder}/summary.json"
    if os.path.exists(sp):
        s = json.load(open(sp)); out["possession_pct"] = s.get("possession_pct"); out["summary"] = {k: s.get(k) for k in ("ball_grade", "passes", "turnovers", "shots", "goals", "players_per_frame_median", "calibration_coverage")}
        log(f"possession: {s.get('possession_pct')}  passes {s.get('passes')}  turnovers {s.get('turnovers')}  ball grade {s.get('ball_grade')}")
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    root = sys.argv[1]; match_id = (sys.argv[2] or None) if len(sys.argv) > 2 else None
    folder = root if os.path.exists(f"{root}/match_data.json") else next(iter(glob.glob(f"{root}/matches/{match_id or '*'}/match_data.json") + glob.glob(f"{root}/{match_id or '*'}/match_data.json")), "")
    if not folder: sys.exit(f"no match_data.json under {root}")
    folder = os.path.dirname(folder) if folder.endswith("match_data.json") else folder; match_id = match_id or os.path.basename(folder)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ref = f"{here}/reference/{match_id}"
    out = grade(folder, ref); out["match"] = match_id
    dst = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(folder.rstrip("/")), "qa.json")
    json.dump(out, open(dst, "w"), indent=1); print(json.dumps(out))
