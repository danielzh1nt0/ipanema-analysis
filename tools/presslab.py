"""2 Oct: pressures_applied counts one per SECOND a player is within 2 m of the opposing carrier (855 per team on AIK).
Compare with one per presser per carrier spell (the usual 'pressure event'), on the exact app inputs of both clips. Free."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P
q = lambda *a: None
def load(clip):
    P_ = pickle.load(open(f"results/volume/cache/{clip}/picker_inputs.pkl", "rb")); fps = P_["fps"]; H = P_["H"]; L, W = P_["L"], P_["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
    ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q), fps)
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0)
    state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
    return per, frames_, state, fps
def count(per, frames_, state, fps, press_r=2.0, per_spell=False, skip_dead=False, gap_s=0.5):
    tracks = {}; team = {}
    for k in range(len(per)):
        for r in per[k]: tracks.setdefault(r[0], {})[k] = r[2]; team[r[0]] = r[1]
    tot = {"A": 0, "B": 0}
    for tid, tk in tracks.items():
        lastp = -99; last_carrier = None; last_k = -99
        for k in sorted(tk):
            f = frames_[k]; c = f["carrier"]
            if c is None or f["team"] == team[tid] or c not in tracks or k not in tracks[c]: continue
            if skip_dead and int(state[k]) >= 2: continue
            if np.linalg.norm(tk[k] - tracks[c][k]) > press_r: continue
            if per_spell:
                new = (c != last_carrier) or (k - last_k > gap_s * fps)
                if new: tot[team[tid]] += 1
                last_carrier, last_k = c, k
            elif k - lastp > fps: tot[team[tid]] += 1; lastp = k
    return tot
if __name__ == "__main__":
    for clip in ("SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"):
        per, frames_, state, fps = load(clip); mins = len(per) / fps / 60
        print(clip, f"{mins:.1f} min")
        for name, kw in (("per second (current)", {}), ("per spell", {"per_spell": True}), ("per spell, gap 1 s", {"per_spell": True, "gap_s": 1.0}), ("per spell, no dead/loose", {"per_spell": True, "skip_dead": True}), ("per spell, 1.5 m", {"per_spell": True, "press_r": 1.5})):
            t = count(per, frames_, state, fps, **kw); print(f"  {name:28s} A {t['A']:4d} B {t['B']:4d} | per 90 min: A {t['A'] * 90 / mins:5.0f} B {t['B'] * 90 / mins:5.0f}")

def spells(state, fps, min_s=0.0):
    """team possession spells from the state: list of (team, k0, k1); a spell ends when the other team takes it"""
    out = []; cur = None; k0 = 0
    for k, s in enumerate(state):
        s = int(s); t = "A" if s == 0 else "B" if s == 1 else None
        if t is None: continue
        if t != cur:
            if cur is not None: out.append((cur, k0, k))
            cur, k0 = t, k
    if cur is not None: out.append((cur, k0, len(state)))
    return out
def count_by_spell(per, frames_, state, fps, press_r=2.0):
    tracks = {}; team = {}
    for k in range(len(per)):
        for r in per[k]: tracks.setdefault(r[0], {})[k] = r[2]; team[r[0]] = r[1]
    sp = spells(state, fps); idx = np.full(len(state), -1)
    for i, (t, a, b) in enumerate(sp): idx[a:b] = i
    tot = {"A": 0, "B": 0}; seen = set()
    for k, f in enumerate(frames_):
        c = f["carrier"]
        if c is None or idx[k] < 0 or sp[idx[k]][0] != f["team"] or c not in tracks or k not in tracks[c]: continue
        for r in per[k]:
            if r[1] == f["team"] or r[0] not in tracks: continue
            if np.linalg.norm(r[2] - tracks[c][k]) <= press_r and (r[0], idx[k]) not in seen: seen.add((r[0], idx[k])); tot[r[1]] += 1
    return tot, len(sp)
if __name__ == "__main__":
    print("--- one pressure per presser per TEAM possession spell (state A/B runs)")
    for clip in ("SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"):
        per, frames_, state, fps = load(clip); mins = len(per) / fps / 60
        t, ns = count_by_spell(per, frames_, state, fps); print(f"  {clip}: spells {ns} ({ns / mins:.0f}/min) | A {t['A']} B {t['B']} | per 90: A {t['A'] * 90 / mins:.0f} B {t['B'] * 90 / mins:.0f}")

def count_by_seq(per, frames_, seqs, press_r=2.0):
    """one pressure per presser per E5 sequence (the de-flickered possession spells the sequences already use)"""
    tracks = {}
    for k in range(len(per)):
        for r in per[k]: tracks.setdefault(r[0], {})[k] = r[2]
    tot = {"A": 0, "B": 0}
    for i, s in enumerate(seqs):
        seen = set()
        for k in range(s["start"], s["end"] + 1):
            if k >= len(frames_): break
            f = frames_[k]; c = f["carrier"]
            if c is None or f["team"] != s["team"] or c not in tracks or k not in tracks[c]: continue
            for r in per[k]:
                if r[1] != s["team"] and r[0] not in seen and np.linalg.norm(r[2] - tracks[c][k]) <= press_r: seen.add(r[0]); tot[r[1]] += 1
    return tot
if __name__ == "__main__":
    print("--- one pressure per presser per E5 sequence")
    for clip in ("SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"):
        P_ = pickle.load(open(f"results/volume/cache/{clip}/picker_inputs.pkl", "rb")); fps = P_["fps"]; H = P_["H"]; L, W = P_["L"], P_["W"]
        per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P_["per"].items()}
        ball = BL.bridge(BL.pick_v2(P_["cands"], H, L, W, per=per, fps=fps, log=q), fps); frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0)
        state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q); ar, _ = P.direction(state, ballm, log=q)
        seqs = P.sequences(state, ballm, bspeed, fps, L, ar, **pinfo["seq"]); mins = len(per) / fps / 60
        t = count_by_seq(per, frames_, seqs); print(f"  {clip}: sequences {len(seqs)} ({len(seqs) / mins:.0f}/min) | A {t['A']} B {t['B']} | per 90: A {t['A'] * 90 / mins:.0f} B {t['B'] * 90 / mins:.0f}")
