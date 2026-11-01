"""N0-01d (1 Nov): blind sheets for the export's spots that lie > 4 m off the pitch (tools/n0_01d_places.py) mixed
with on-pitch controls of the same kinds. Per moment 6 frames; the exported spot is drawn as a cyan cross through each
frame's own metres->pixels homography (frames[].pitch_lines), with the pitch's outer lines in thin white.
Question by eye: at the restart / ball won / shot, where is the ball - at the cross (within ~3 m), over the line, or
somewhere else on the pitch; and for a restart: what is taken (throw-in, corner, goal kick, kick-off, play on).

    python tools/n0_01d_moments.py     # writes results/qa/n0-01d/moments.json (no kinds, no flags) + moments_key.json
"""
import os, sys, json, glob, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n0_01d_places as P

ROOT = "results/volume/runs/matches"
OUT = "results/qa/n0-01d"
WIN = {"restart": (-3.0, 2.0), "ball won": (-1.5, 1.5), "shot": (-1.0, 1.5)}


def load_frames(mid):
    fr = []
    for f in sorted(glob.glob(f"{ROOT}/{mid}/frames_*.json")): fr += json.load(open(f))["frames"]
    fr.sort(key=lambda x: x["t"]); return fr


def near(fr, t, tol=0.6):
    import bisect
    ts = [x["t"] for x in fr]; i = bisect.bisect_left(ts, t)
    c = [j for j in (i - 1, i) if 0 <= j < len(fr) and abs(fr[j]["t"] - t) <= tol]
    return fr[min(c, key=lambda j: abs(fr[j]["t"] - t))] if c else None


def items(stats):
    """every located restart / ball won / shot with its distance off the pitch"""
    L, W = stats["pitch"]["length"], stats["pitch"]["width"]; out = []
    for r in stats.get("restarts") or []:
        if r.get("x_m") is not None: out.append({"what": "restart", "t": r["t"], "kind": r["kind"], "xy": [r["x_m"], r["y_m"]]})
    for r in stats["metrics"].get("high_turnovers") or []:
        if r.get("x_m") is not None: out.append({"what": "ball won", "t": r["t"], "kind": r["zone"], "xy": [r["x_m"], r["y_m"]]})
    for r in stats["metrics"].get("shots_detected") or []:
        if r.get("x_m") is not None: out.append({"what": "shot", "t": r["t"], "kind": r.get("outcome"), "xy": [r["x_m"], r["y_m"]]})
    for o in out: o["off_m"] = round(P._off(o["xy"][0], o["xy"][1], L, W), 1)
    return out


def build(seed=1):
    rnd = random.Random(seed); moments, key = [], []
    for mid in ("SFKBP1109", "p15u-vs-vallentuna-2026-10-03-6cce"):
        st = json.load(open(f"{ROOT}/{mid}/stats.json")); fr = load_frames(mid)
        if not fr: continue
        its = [o for o in items(st) if near(fr, o["t"]) is not None]
        flagged = [o for o in its if o["off_m"] > P.MARGIN_M]
        ctrl = []
        for o in flagged:                                   # one control per flagged item: same match, same kind, on the pitch, near a line if possible
            pool = [c for c in its if c["what"] == o["what"] and c["off_m"] == 0 and c not in ctrl and (o["what"] != "restart" or c["kind"] == o["kind"])]
            pool.sort(key=lambda c: min(c["xy"][0], st["pitch"]["length"] - c["xy"][0], c["xy"][1], st["pitch"]["width"] - c["xy"][1]))
            if pool: ctrl.append(rnd.choice(pool[:6]))
        for o, flag in [(o, True) for o in flagged] + [(c, False) for c in ctrl]:
            a, b = WIN[o["what"]]; t0, t1 = o["t"] + a, o["t"] + b
            ts = [t0 + (t1 - t0) * i / 5 for i in range(6)]
            H = [];
            for t in ts:
                f = near(fr, t); H.append(f["pitch_lines"] if f and f.get("cal_ok") and f.get("pitch_lines") else None)
            moments.append({"match": mid, "what": o["what"], "t": o["t"], "ts": ts, "xy": o["xy"], "H": H, "L": st["pitch"]["length"], "W": st["pitch"]["width"]})
            key.append({"match": mid, "what": o["what"], "t": o["t"], "kind": o["kind"], "xy": o["xy"], "off_m": o["off_m"], "flagged": flag})
    order = list(range(len(moments))); rnd.shuffle(order)
    moments = [dict(moments[i], id=f"m{n:02d}") for n, i in enumerate(order)]; key = [dict(key[i], id=f"m{n:02d}") for n, i in enumerate(order)]
    return moments, key


if __name__ == "__main__":
    m, k = build(); os.makedirs(OUT, exist_ok=True)
    json.dump(m, open(f"{OUT}/moments.json", "w")); json.dump(k, open(f"{OUT}/moments_key.json", "w"), indent=1)
    print(len(m), "moments,", sum(x["flagged"] for x in k), "flagged")
