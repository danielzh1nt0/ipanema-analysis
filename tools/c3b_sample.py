"""C3b (8 Oct, local, $0): which of Vallentuna's untrusted camera seconds are actually right?
The line model's rows already hold a pose for 5,961 of 5,966 seconds; 23% of the match seconds are not trusted
(linecal.brave). Most are 'forward/backward disagree' (tracked forward vs backward from the next anchor) or anchors that
'jump from track'. This picks a blind sample from groups that a looser rule would let in, plus controls, and writes
  results/qa/c3b/sample.json  (id, t, pose - no group; the free runner draws these: tools/c3b_look.py)
  results/review/c3b_key.json (id -> group; opened only after grading)
    python tools/c3b_sample.py
    python tools/c3b_sample.py SFKBP1109 results/qa/c3b_sfk results/review/c3b_sfk_key.json jump_anchor:16,brave:6
(8 Oct: SFK-BP rows carry no 'anchor' flag; 'jump from track' is only ever set on an anchor, so it implies one.)"""
import json, random, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import linecal as LC

MATCH = "p15u-vs-vallentuna-2026-10-03-6cce"
GROUPS = [("fb15-25", 20), ("fb25-40", 12), ("jump_anchor", 16), ("fb100+", 8), ("brave", 8)]

def group(r):
    if r.get("pose") is None: return None
    if LC.brave(r): return "brave"
    why = r.get("why", []); fb = r.get("fwd_bwd_px")
    if (r.get("anchor") or "jump from track" in why) and why and all(w.startswith(LC.NOT_A_VETO + ("jump from track",)) for w in why): return "jump_anchor"
    if fb is None or any(not w.startswith(LC.NOT_A_VETO) for w in why): return None
    if 15 < fb <= 25: return "fb15-25"
    if 25 < fb <= 40: return "fb25-40"
    if fb > 100: return "fb100+"
    return None

def pick(rows, periods, seed=8, groups=GROUPS):
    inplay = [r for r in rows if any(a + 5 <= r["t"] <= b - 5 for a, b in periods)]
    rng = random.Random(seed); out = []
    for g, n in groups:
        pool = [r for r in inplay if group(r) == g]; out += [(g, r) for r in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(out)
    return [{"id": f"c{i:02d}", "t": r["t"], "pose": r["pose"], "group": g, "why": r.get("why", []), "fb": r.get("fwd_bwd_px"), "hard": r.get("hard", [])}
            for i, (g, r) in enumerate(out)]

def main(argv=sys.argv[1:]):
    match = argv[0] if argv else MATCH; out = argv[1] if len(argv) > 1 else "results/qa/c3b"; keyp = argv[2] if len(argv) > 2 else "results/review/c3b_key.json"
    groups = [(g, int(n)) for g, n in (x.split(":") for x in argv[3].split(","))] if len(argv) > 3 else GROUPS
    d = json.load(open(f"calibration/{match}_lines_match.json")); per = json.load(open(f"periods/{match}.json"))["periods_s"]
    s = pick(sorted(d["rows"], key=lambda r: r["t"]), per, groups=groups)
    os.makedirs(out, exist_ok=True); os.makedirs(os.path.dirname(keyp), exist_ok=True)
    json.dump({"match": match, "src_key": f"{match}/video.mp4", "camera": d["camera"], "rows": [{"id": q["id"], "t": q["t"], "pose": q["pose"]} for q in s]},
              open(f"{out}/sample.json", "w"), indent=0)
    json.dump({q["id"]: {k: q.get(k) for k in ("group", "t", "why", "fb", "hard")} for q in s}, open(keyp, "w"), indent=1)
    print(len(s), "rows sampled")

if __name__ == "__main__": main()
