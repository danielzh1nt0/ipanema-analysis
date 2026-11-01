"""N0-01c (1 Nov, local, $0): blind film strips for 'pass to himself' (stats.passes[] with from == to) on the app copies,
mixed with control passes (completed, from != to). Per pass: the release time t, the arrival = first exported frame after
t whose carrier is the pass's 'to' id, and one crop box (pixels) that holds the passer at t and the receiver at arrival.
The sheet tool cuts 6 frames from t - 0.4 s to arrival + 0.8 s in that box and rings only the passer in the first frame,
so the grader cannot tell a self pass from a control. Key (which is which) goes to a separate file.

Question by eye: after P plays the ball, who has it next - P himself (dribble / knock-on / he keeps it), a team-mate,
an opponent, or can't tell?

    python tools/n0_01c_moments.py   -> results/qa/n0-01c/moments.json + moments_key.json
"""
import os, sys, json, random, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n0_30d_clips import load

MATCHES = {"SFKBP1109": (8, 4), "p15u-vs-vallentuna-2026-10-03-6cce": (18, 8)}   # (self passes, controls)
OUT = "results/qa/n0-01c"


def frame_at(fr, ts, t, tol=0.2):
    i = bisect.bisect_left(ts, t)
    j = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t), default=None)
    return j if j is not None and abs(ts[j] - t) <= tol else None


def px_of(f, pid):
    return next((q["px"] for q in f["players"] if q["id"] == pid and q.get("px")), None)


def arrival(fr, ts, t, to, max_s=4.5):
    """index of the first frame after t + 0.1 s whose carrier is `to`"""
    i = bisect.bisect_left(ts, t + 0.1)
    while i < len(fr) and fr[i]["t"] <= t + max_s:
        if fr[i].get("carrier") == to: return i
        i += 1
    return None


def box(a, b, W=1920, H=1080, pad=160, min_w=560):
    """16:9 crop holding both points with padding; players' px are feet, so extra room above"""
    x0, x1 = min(a[0], b[0]) - pad, max(a[0], b[0]) + pad; y0, y1 = min(a[1], b[1]) - pad - 120, max(a[1], b[1]) + pad / 2
    w = max(x1 - x0, (y1 - y0) * 16 / 9, min_w); h = w * 9 / 16; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = min(w, W), min(h, H)
    x0 = int(min(max(cx - w / 2, 0), W - w)); y0 = int(min(max(cy - h / 2, 0), H - h))
    return [x0, y0, int(w), int(h)]


def centres(fr, ts, t0, t1, n=6, W=1920, H=1080, cw=720, ch=405):
    """run 2: the camera pans, so one fixed box loses the play. Per strip frame a box centred on the exported ball
    (nearest frame within 0.5 s with a ball pixel; else the last centre; else the passer) -> [x0, y0] top-left"""
    out, last = [], None
    for i in range(n):
        t = t0 + (t1 - t0) * i / (n - 1); j = bisect.bisect_left(ts, t); best = None
        for k in range(max(0, j - 13), min(len(fr), j + 13)):
            b = fr[k].get("ball")
            if b and b.get("px") and abs(fr[k]["t"] - t) <= 0.5 and (best is None or abs(fr[k]["t"] - t) < best[0]): best = (abs(fr[k]["t"] - t), b["px"])
        c = best[1] if best else last
        last = c; out.append(c)
    first = next((c for c in out if c is not None), None)
    return [[int(min(max((c or first)[0] - cw / 2, 0), W - cw)), int(min(max((c or first)[1] - ch * 0.55, 0), H - ch))] if (c or first) else None for c in out]


def usable(st, fr, ts, want_self):
    out = []
    for i, p in enumerate(st["passes"]):
        sf = p.get("to") is not None and p["from"] == p["to"]
        if sf != want_self or (not want_self and not p["completed"]): continue
        j = frame_at(fr, ts, p["t"]);
        if j is None: continue
        a = px_of(fr[j], p["from"]); k = arrival(fr, ts, p["t"], p["to"])
        if a is None or k is None: continue
        b = px_of(fr[k], p["to"])
        if b is None: continue
        out.append({"i": i, "t": fr[j]["t"], "t_arr": fr[k]["t"], "passer_px": a, "box": box(a, b), "self": sf,
                    "length_m": p["length_m"], "completed": p["completed"], "from": p["from"], "to": p["to"]})
    return out


def build(seed=3):
    rnd = random.Random(seed); out = []
    for mid, (ns, nc) in MATCHES.items():
        st = json.load(open(f"results/volume/runs/matches/{mid}/stats.json")); _, fr = load(f"results/volume/runs/matches/{mid}/match_data.json")
        ts = [f["t"] for f in fr]
        S, C = usable(st, fr, ts, True), usable(st, fr, ts, False)
        n_self = sum(p.get("to") is not None and p["from"] == p["to"] for p in st["passes"])
        print(mid, "self passes", n_self, "usable", len(S), "controls usable", len(C))
        for r in rnd.sample(S, min(ns, len(S))) + rnd.sample(C, min(nc, len(C))):
            r["match"] = mid; r["tl"] = centres(fr, ts, r["t"] - 0.4, r["t_arr"] + 0.8); out.append(r)
    rnd.shuffle(out); key = {}
    for n, r in enumerate(out):
        r["id"] = f"s{n:02d}"; key[r["id"]] = {k: r[k] for k in ("match", "i", "t", "t_arr", "self", "from", "to", "length_m", "completed")}
    blind = [{"id": r["id"], "match": r["match"], "t0": round(r["t"] - 0.4, 2), "t1": round(r["t_arr"] + 0.8, 2), "t_rel": r["t"],
              "passer_px": r["passer_px"], "box": r["box"], "tl": r["tl"]} for r in out]
    return blind, key


if __name__ == "__main__":
    b, k = build(); os.makedirs(OUT, exist_ok=True)
    json.dump(b, open(f"{OUT}/moments.json", "w"), indent=1); json.dump(k, open(f"{OUT}/moments_key.json", "w"), indent=1)
    print(len(b), "moments")
