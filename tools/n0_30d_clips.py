"""N0-30d: the 5-s gain after each 'won the ball' event on the repo's app match copies, old rule vs new (N0-30d), rebuilt
from the exported frame files (ball metres per kept frame). The copies' frames are thinned (every Nth frame) and some
(SFK-BP) mix older exports, so 'old' here is a rebuild - its agreement with the stored event value is reported.

    python tools/n0_30d_clips.py [--out results/qa/n0-30d/clips_gain5s.json]
"""
import sys, os, json, glob, bisect, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P


def load(md_path):
    d = json.load(open(md_path)); root = os.path.dirname(md_path); fr = list(d.get("frames") or [])
    for c in d.get("frame_chunks") or []:
        q = f"{root}/{c['key']}.json"
        if os.path.exists(q): fr += json.load(open(q))["frames"]
    fr.sort(key=lambda f: f["t"]); return d, fr


def rebuild(d, fr, ev, onpitch, detail=False):
    """gain for one turnover_won event from the exported frames: a dense ballm on the export's own frame grid"""
    ts = [f["t"] for f in fr]; t0 = ev["payload"].get("t_won", ev["t"])
    i0 = bisect.bisect_left(ts, t0 - 0.05)
    if i0 >= len(ts) or abs(ts[i0] - t0) > 0.5: return (None, None, None, None) if detail else (None, None)
    dt = np.median(np.diff(ts[max(0, i0 - 20):i0 + 20])) if len(ts) > 2 else 0.04; fps = 1.0 / max(dt, 1e-3)
    seg = fr[i0:i0 + int(7 * fps)]
    ballm = {j: np.array(f["ball"]["m"]) for j, f in enumerate(seg) if f.get("ball") and f["ball"].get("m") is not None}
    per = d["periods"]; ar = next((p["attack_right"] for p in per if p["t_start"] - 1 <= t0 <= p["t_end"] + 1), d["attack_right"])
    right = ar[ev["team"]] if isinstance(ar, dict) else (ar if ev["team"] == "A" else not ar)
    g, a, b = P.gain_window(ballm, 0, len(seg), fps, right, (d["pitch"]["length"], d["pitch"]["width"]), onpitch=onpitch, detail=True)
    pick = lambda j: None if j is None else {"t": seg[j]["t"], "px": seg[j]["ball"]["px"], "m": seg[j]["ball"]["m"]}
    return (g, fps, pick(a), pick(b)) if detail else (g, fps)


def main(out):
    res = {}
    for p in sorted(glob.glob("results/volume/runs/matches/*/match_data.json")):
        mid = p.split("/")[-2]; d, fr = load(p)
        evs = [e for e in d["events"] if e["type"] == "turnover_won"]
        if len(evs) < 5 or not fr: continue
        rows = []
        for e in evs:
            old, fps, o0, o5 = rebuild(d, fr, e, False, True); new, _, n0, n5 = rebuild(d, fr, e, True, True)
            rows.append({"t_won": e["payload"].get("t_won", e["t"]), "team": e["team"], "stored": e["payload"].get("gain_5s_m"), "old": old, "new": new, "fps": fps,
                         "old_start": o0, "old_end": o5, "new_start": n0, "new_end": n5})
        R = [r for r in rows if r["fps"] is not None]
        st = [r for r in R if r["stored"] is not None and r["old"] is not None]
        ab = lambda v: [abs(x) for x in v if x is not None]
        res[mid] = {"wins": len(evs), "rebuilt": len(R), "old_matches_stored_within_2m": f"{sum(abs(r['stored'] - r['old']) <= 2 for r in st)}/{len(st)}",
                    "stored_over_40m": sum(x > 40 for x in ab(r["stored"] for r in rows)), "stored_over_106m": sum(x > 106 for x in ab(r["stored"] for r in rows)),
                    "old_over_40m": sum(x > 40 for x in ab(r["old"] for r in R)), "new_over_40m": sum(x > 40 for x in ab(r["new"] for r in R)),
                    "old_none": sum(r["old"] is None for r in R), "new_none": sum(r["new"] is None for r in R),
                    "changed_over_3m": sum(1 for r in R if (r["old"] is None) != (r["new"] is None) or (r["old"] is not None and abs(r["old"] - r["new"]) > 3)),
                    "rows": rows}
        print(mid, {k: v for k, v in res[mid].items() if k != "rows"})
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]; main(a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-30d/clips_gain5s.json")
