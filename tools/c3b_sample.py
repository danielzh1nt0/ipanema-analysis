"""C3b (8 Oct, local, $0): which of Vallentuna's untrusted camera seconds are actually right?
The line model's rows already hold a pose for 5,961 of 5,966 seconds; 23% of the match seconds are not trusted
(linecal.brave). Most are 'forward/backward disagree' (tracked forward vs backward from the next anchor) or anchors that
'jump from track'. This picks a blind sample from groups that a looser rule would let in, plus controls, and writes
  results/qa/c3b/sample.json  (id, t, pose - no group; the free runner draws these: tools/c3b_look.py)
  results/review/c3b_key.json (id -> group; opened only after grading)
    python tools/c3b_sample.py"""
import json, random, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import linecal as LC

MATCH = "p15u-vs-vallentuna-2026-10-03-6cce"
GROUPS = [("fb15-25", 20), ("fb25-40", 12), ("jump_anchor", 16), ("fb100+", 8), ("brave", 8)]

def group(r):
    if r.get("pose") is None: return None
    if LC.brave(r): return "brave"
    why = r.get("why", []); fb = r.get("fwd_bwd_px")
    if r.get("anchor") and why and all(w.startswith(LC.NOT_A_VETO + ("jump from track",)) for w in why): return "jump_anchor"
    if fb is None or any(not w.startswith(LC.NOT_A_VETO) for w in why): return None
    if 15 < fb <= 25: return "fb15-25"
    if 25 < fb <= 40: return "fb25-40"
    if fb > 100: return "fb100+"
    return None

def pick(rows, periods, seed=8):
    inplay = [r for r in rows if any(a + 5 <= r["t"] <= b - 5 for a, b in periods)]
    rng = random.Random(seed); out = []
    for g, n in GROUPS:
        pool = [r for r in inplay if group(r) == g]; out += [(g, r) for r in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(out)
    return [{"id": f"c{i:02d}", "t": r["t"], "pose": r["pose"], "group": g, "why": r.get("why", []), "fb": r.get("fwd_bwd_px"), "hard": r.get("hard", [])}
            for i, (g, r) in enumerate(out)]

def main():
    d = json.load(open(f"calibration/{MATCH}_lines_match.json")); per = json.load(open(f"periods/{MATCH}.json"))["periods_s"]
    s = pick(sorted(d["rows"], key=lambda r: r["t"]), per)
    os.makedirs("results/qa/c3b", exist_ok=True); os.makedirs("results/review", exist_ok=True)
    json.dump({"match": MATCH, "src_key": f"{MATCH}/video.mp4", "camera": d["camera"], "rows": [{"id": q["id"], "t": q["t"], "pose": q["pose"]} for q in s]},
              open("results/qa/c3b/sample.json", "w"), indent=0)
    json.dump({q["id"]: {k: q[k] for k in ("group", "t", "why", "fb", "hard")} for q in s}, open("results/review/c3b_key.json", "w"), indent=1)
    print(len(s), "rows sampled")

if __name__ == "__main__": main()
