"""5 Oct (V1c): turn the Shot Spot Marker clicks (artifact 2uSb4YugerBje9EnLavHXK, collection 'clicks', saved with
ArtifactData list out_dir=results/app/shotclick/db) into shot origins: pixel at the clicked video time -> metres with the
exported frame's pitch_lines (second half already mirrored), written as "t shot|goal TEAM x y" lines in
reference/veo_highlights_<match>.txt. Clicks in frames without a calibration are reported, not written.
    PYTHONPATH=. python tools/shotclick_import.py"""
import json, glob, os, re, bisect, numpy as np
clicks = [json.load(open(f)) for f in glob.glob("results/app/shotclick/db/clicks/*.json")]
clicks = [c.get("data", c) for c in clicks]
by = {}
for c in clicks: by.setdefault(c["match"], []).append(c)
for m, cs in by.items():
    fr = []
    for f in sorted(glob.glob(f"results/volume/runs/matches/{m}/frames_*.json")): fr += json.load(open(f))["frames"]
    ts = [f["t"] for f in fr]; res = {}
    for c in cs:
        if c.get("skip"): continue
        i = min(bisect.bisect_left(ts, c["video_t"]), len(ts) - 1); H = None
        for j in (i, i - 1, i + 1):
            if 0 <= j < len(fr) and abs(fr[j]["t"] - c["video_t"]) < 0.25 and fr[j].get("pitch_lines"): H = np.array(fr[j]["pitch_lines"]).reshape(3, 3)
        if H is None: print(m, c["t"], "no calibration at", c["video_t"]); continue
        v = np.linalg.inv(H) @ np.array([c["px"][0], c["px"][1], 1.0]); res[int(round(c["t"]))] = ((v[:2] / v[2]).round(1), c.get("team"))
    p = f"reference/veo_highlights_{m}.txt"; out = []
    for line in open(p):
        mm = re.match(r"\s*(\d+)\s+(shot|goal)\b(?:\s+([AB]))?", line)
        if mm and int(mm.group(1)) in res:
            (x, y), tm = res[int(mm.group(1))]; tm = mm.group(3) or tm      # a goal team checked by eye wins over the click
            if tm: line = f"{mm.group(1)} {mm.group(2)} {tm} {x} {y}\n"
        out.append(line)
    open(p, "w").writelines(out); print(m, "origins written for", len(res), "shots ")
