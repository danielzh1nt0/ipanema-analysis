"""4 Oct: build results/review/spells/zoom.json for kaggle/spell_zoom.py: for each spell in moments.json the 5 moments with the
exported ball pixel, player pixels + teams and the raw possession label from the fetched frame chunks.
    PYTHONPATH=. python tools/spell_zoom_prep.py"""
import json, glob, bisect
M = json.load(open("results/review/spells/moments.json")); D = f"results/volume/runs/matches/{M['match']}"
frames = []
for f in sorted(glob.glob(f"{D}/frames_*.json")): frames += json.load(open(f))["frames"]
ts = [f["t"] for f in frames]; print("frames", len(frames), "keys", list(frames[0])[:12])
def at(t):
    i = min(bisect.bisect_left(ts, t), len(ts) - 1); f = frames[i]
    b = f.get("ball"); ball = [b["px"][0], b["px"][1]] if b and b.get("px") else None
    pl = [[p["px"][0], p["px"][1], p.get("team")] for p in f.get("players", []) if p.get("px")]
    return {"t": round(f["t"], 2), "ball": ball, "players": pl, "poss": f.get("possession")}
out = []
for s in M["spells"]:
    tl = [(s["t_start"] - 1.5, "before"), (s["t_start"], "start"), ((s["t_start"] + s["t_end"]) / 2, "middle"), (s["t_end"], "end"), (s["t_end"] + 1.5, "after")]
    out.append({**s, "moments": [{**at(t), "label": lab} for t, lab in tl]})
json.dump({"src_key": M["src_key"], "spells": out}, open("results/review/spells/zoom.json", "w"), indent=0); print("spells", len(out), out[0]["moments"][1])
