"""N0-25b: who are the 'players' our team heat maps put on the far touchline (y < 3 m: 4-15% of a team's time on the
app clips, ~1% for pro players)? Picks the people with the most frames within 3 m of a touchline on the two app clips
(picker_inputs.pkl repo copies) plus random mid-pitch controls, shuffled; writes moments files for tools/n0_20b_frames.py
(ring = the person's feet) and the key (results/qa/n0-25b/moments_key.json), opened only after grading.
Usage: python tools/n0_25b_moments.py"""
import pickle, json, random, collections, numpy as np
CL = {"sfk": ("SFKBP1109_s1200", "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl"),
      "aik": ("p15u-vs-aik-2026-09-21-bd09_s2520", "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl")}
rng = random.Random(25); key = {}
for short, (clip, p) in CL.items():
    d = pickle.load(open(p, "rb")); per, fps, W = d["per"], d["fps"], float(d["W"]); ks = sorted(per) if isinstance(per, dict) else range(len(per))
    rows = collections.defaultdict(list)
    for k in ks:
        for r in per[k]: rows[(r[0], r[1])].append((k, float(r[2][0]), float(r[2][1]), float(r[3][0]), float(r[3][1])))
    edge = collections.Counter({t: sum(1 for q in v if q[2] < 3 or q[2] > W - 3) for t, v in rows.items()})
    pick = [(t, "edge") for t, c in edge.most_common(12) if c >= 30]
    mids = [t for t, v in rows.items() if len(v) >= 60 and all(10 < q[2] < W - 10 for q in v) and t not in dict(pick)]
    pick += [(t, "control") for t in rng.sample(mids, min(6, len(mids)))]
    rng.shuffle(pick); moms = []
    for j, (t, kind) in enumerate(pick):
        v = rows[t]; e = [q for q in v if q[2] < 3 or q[2] > W - 3] if kind == "edge" else v
        sel = [e[int(i)] for i in np.linspace(0, len(e) - 1, 4)]; mid = f"{short}{j:02d}"
        moms.append({"id": mid, "clip": clip, "team": "ring = person", "t": round(sel[0][0] / fps, 2), "frame": sel[0][0], "attacks": "-",
                     "tiles": [{"frame": q[0], "t": round(q[0] / fps, 2), "ball_px": [q[3], q[4]]} for q in sel]})
        key[mid] = {"track": t[0], "team": t[1], "kind": kind, "frames": len(v), "edge_frames": edge[t], "y_m": [round(q[2], 1) for q in sel], "x_m": [round(q[1], 1) for q in sel]}
    json.dump({"clip": clip, "moments": moms}, open(f"results/qa/n0-25b/moments_{short}.json", "w"), indent=1)
json.dump(key, open("results/qa/n0-25b/moments_key.json", "w"), indent=1)
print({k: v["kind"] for k, v in key.items()})
