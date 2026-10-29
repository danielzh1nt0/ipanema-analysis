"""N0-29f: the app's headline possession % (stats.json teams[].possession_pct, the KPI 'Possession' and the summary line)
for a whole half, checked on pro data with a frame-by-frame truth. The nightly only says 'worst half 8 pts off'
(against an event-gap truth); nobody looked at why, or at other ways to count it.

What the app writes (analytics.stats): A's share of the frames where the RAW possession state (possession_simple) says
A or B; loose frames left out.

Truth (Metrica's 2 pro games, 4 halves): n0_25d_tilt_windows.truth_own - per frame the team of the last on-ball event,
ball out / a foul until the restart = nobody's. Also the nightly's own truth (metricalab.truth) for reference.

Variants of ours (same captured state, nothing re-tracked):
  raw          - today (the app)
  spell        - the spell state the counted stats use (P.spell_state)
  raw_fill     - raw, a loose frame credited to the team that had it last (ball in flight = the passer's, like the truth)
  raw_nostop   - raw, frames inside the restart step's stoppages left out (P.stoppage_spans)
  spell_fill   - spell + fill
Balls: perfect, 'flicker' (the dot hops to a nearby opponent, like our clips), 'streak', 'veo' (65% of players, 25%
balls missing, 10% wrong picks) - 2 seeds for the noisy ones.

    PYTHONPATH=. python tools/n0_29f_possession.py /tmp/metrica/data [--out results/qa/n0-29f/metrica_possession.json]
    PYTHONPATH=. python tools/n0_29f_possession.py --dry
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P

FPS, L, W = 25.0, 105.0, 68.0
VARIANTS = ("raw", "spell", "raw_fill", "raw_nostop", "spell_fill", "raw_gap1", "raw_gap2", "raw_gap5", "raw_gapspell2")


def share(st, keep=None):
    """A's % of the frames whose state is A (0) or B (1); keep = boolean mask of frames that count"""
    st = np.asarray(st)
    m = np.isin(st, [0, 1]) if keep is None else (np.isin(st, [0, 1]) & keep)
    a, b = int(((st == 0) & m).sum()), int(((st == 1) & m).sum())
    return round(100.0 * a / (a + b), 1) if a + b else None


def fill_loose(st):
    """a loose / dead frame (>= 2) takes the last team's value; frames before anyone had it stay loose"""
    st = np.asarray(st).copy(); last = None
    for i, v in enumerate(st):
        if v in (0, 1): last = v
        elif last is not None: st[i] = last
    return st


def stop_mask(n, spans):
    keep = np.ones(n, bool)
    for a, b in spans: keep[max(0, a):min(n, b + 1)] = False
    return keep


def gap_mask(has_ball, fps, gap_s):
    """True on frames inside a run of >= gap_s seconds with no ball (possession_simple keeps the last team there)"""
    hb = np.asarray(has_ball, bool); out = np.zeros(len(hb), bool); i = 0; g = int(round(gap_s * fps))
    while i < len(hb):
        if hb[i]: i += 1; continue
        j = i
        while j < len(hb) and not hb[j]: j += 1
        if j - i >= g: out[i:j] = True
        i = j
    return out


def variants(state, cstate, spans, has_ball=None, fps=FPS):
    """-> {variant: A share %} from the captured states"""
    n = len(state); keep = stop_mask(n, spans)
    v = {"raw": share(state), "spell": share(cstate), "raw_fill": share(fill_loose(state), keep),
         "raw_nostop": share(state, keep), "spell_fill": share(fill_loose(cstate), keep)}
    if has_ball is not None:
        for g in (1, 2, 5): v[f"raw_gap{g}"] = share(state, ~gap_mask(has_ball, fps, g))
        gm = gap_mask(has_ball, fps, 2); st2 = np.asarray(state).copy(); st2[gm] = np.asarray(cstate)[gm]
        v["raw_gapspell2"] = share(st2)
    return v


def truth_share(own, k0, n, keep=None):
    ok = (lambda i: True) if keep is None else (lambda i: keep[i])
    a = sum(1 for i in range(n) if ok(i) and own.get(k0 + i) == "A"); b = sum(1 for i in range(n) if ok(i) and own.get(k0 + i) == "B")
    return round(100.0 * a / (a + b), 1) if a + b else None


def capture(fr, per, noisy, seed):
    """metricalab.ours (the pipeline's steps) with the raw state, spell state and stoppages captured"""
    import metricalab as ML
    got = {}
    o_ps, o_sp, o_car = P.pipeline_state, P.spell_state, P.carriers
    def ps_(*a, **k): r = o_ps(*a, **k); got["state"], got["dstate"] = r[0], r[2]; return r
    def sp_(*a, **k): r = o_sp(*a, **k); got.setdefault("cstate", r); return r
    def car_(*a, **k): r = o_car(*a, **k); got["ballm"] = r[1]; return r
    P.pipeline_state, P.spell_state, P.carriers = ps_, sp_, car_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.pipeline_state, P.spell_state, P.carriers = o_ps, o_sp, o_car
    spans = P.stoppage_spans(got["dstate"], got["ballm"], FPS, L, W)
    hb = np.array([i in got["ballm"] for i in range(len(got["state"]))])
    return got["state"], got["cstate"], spans, o, hb


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "flicker", "streak", True), seeds=(0, 1)):
    import metricalab as ML, n0_24d_ppda as PP, n0_25d_tilt_windows as TW
    rows = []
    for g in games:
        fr, ev = PP.load_game(d, g)
        nt = ML.truth(ML.load_events(f"{d}/{g}/{g}_RawEventsData.csv"), {p: [k for k in fr if fr[k][0] == p] for p in halves})
        for per in halves:
            ks = sorted(k for k in fr if fr[k][0] == per); k0, n = ks[0], len(ks)
            own = TW.truth_own(ev, fr, per); tr = truth_share(own, k0, n)
            seen = ~gap_mask([fr[k][2] is not None for k in ks], FPS, 2)   # Metrica blanks its ball for long in-play
            tr_seen = truth_share(own, k0, n, seen)                       # stretches; our picker never does (N0-29f) ->
            # the 'seen' score = truth and ours on the same frames, outside Metrica's own ball gaps of 2 s+
            for nz in noises:
                for sd in (seeds if nz else (0,)):
                    st, cst, spans, o, hb = capture(fr, per, nz, sd)
                    v = variants(st, cst, spans, hb)
                    name = {False: "perfect", True: "veo"}.get(nz, nz)
                    vs = {"raw": share(st, seen), "spell": share(cst, seen)}
                    rows.append({"game": g, "half": per, "ball": name, "seed": sd, "truth": tr, "truth_seen": tr_seen, "ours_seen": vs,
                                 "err_seen": {k: round(x - tr_seen, 1) for k, x in vs.items()}, "truth_nightly": nt[per]["possession_pct"]["Home"],
                                 "ours": v, "err": {k: (round(x - tr, 1) if x is not None else None) for k, x in v.items()},
                                 "loose_pct": round(100.0 * float((np.asarray(st) >= 2).mean()), 1), "ball_pct": round(100.0 * float(hb.mean()), 1), "gap2_pct": round(100.0 * float(gap_mask(hb, FPS, 2).mean()), 1), "stoppages": len(spans)})
                    print(g, per, name, sd, "truth", tr, {k: x for k, x in v.items()}, flush=True)
    return rows


def summary(rows):
    out = {}
    for b in sorted({r["ball"] for r in rows}):
        R = [r for r in rows if r["ball"] == b]; out[b] = {}
        for v in VARIANTS:
            if any(r["err"].get(v) is None for r in R): continue
            e = [abs(r["err"][v]) for r in R if r["err"][v] is not None]
            side = sum(1 for r in R if r["err"][v] is not None and (r["ours"][v] - 50) * (r["truth"] - 50) > 0)
            out[b][v] = {"mean_abs": round(float(np.mean(e)), 1), "worst": round(float(np.max(e)), 1), "bias": round(float(np.mean([r["err"][v] for r in R])), 1),
                         "right_side": f"{side}/{len(R)}"}
        for v in ("raw", "spell"):
            e = [abs(r["err_seen"][v]) for r in R]
            out[b]["seen_" + v] = {"mean_abs": round(float(np.mean(e)), 1), "worst": round(float(np.max(e)), 1),
                                   "bias": round(float(np.mean([r["err_seen"][v] for r in R])), 1)}
    return out


def dry():
    """stand-in states: fill and share behave"""
    st = np.array([2, 0, 0, 2, 2, 1, 1, 1, 2, 0]); assert share(st) == 50.0
    f = fill_loose(st); assert list(f) == [2, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    v = variants(st, st, [(3, 4)]); assert v["raw_fill"] == 42.9 and v["raw_nostop"] == 50.0
    hb = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1]; gm = gap_mask(hb, 1.0, 2); assert list(gm) == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    v = variants(st, st, [], hb, fps=1.0); assert v["raw_gap2"] == 25.0, v   # frames 1-3 dropped: A only frame 9, B 5-7
    print("dry ok", v)


if __name__ == "__main__":
    if "--dry" in sys.argv: dry(); sys.exit()
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results/qa/n0-29f/metrica_possession.json"
    if "--game" in sys.argv:                                   # one game per process (memory), rows only
        g = sys.argv[sys.argv.index("--game") + 1]; rows = main(sys.argv[1], games=(g,))
        os.makedirs(os.path.dirname(out), exist_ok=True); json.dump({"rows": rows}, open(out, "w"), indent=1); sys.exit()
    if "--merge" in sys.argv:                                  # --merge a.json b.json
        rows = [r for f in sys.argv[sys.argv.index("--merge") + 1:] if f.endswith(".json") and f != out for r in json.load(open(f))["rows"]]
    else: rows = main(sys.argv[1])
    s = summary(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"rows": rows, "summary": s}, open(out, "w"), indent=1)
    print(json.dumps(s, indent=1))
