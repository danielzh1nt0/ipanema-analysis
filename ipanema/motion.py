"""1 Oct: per-player speed (km/h) and distance (m) for the app's speed layer, from the tracked pitch positions.
Raw positions wobble (calibration, box jitter, ID swaps), so:
  1. split a track wherever it jumps (> JUMP_M in one frame, or a gap > 0.5 s): a jump is a tracking error,
     not running;
  2. smooth positions with a centred SMOOTH_S window;
  3. speed over a STEP_S baseline; readings above MAX_MS (36 km/h) are dropped (shown as no speed), not clipped;
  4. distance = sum of smoothed steps on valid stretches; a new piece starts from the distance the track had.
Output: {frame: {track_id: (kmh or None, metres_so_far)}}. Identity is per TRACK: tracks are short (median 4-8 s), so
these are not per-player match totals until tracks are joined into players."""
import numpy as np

MAX_MS = 7.0           # 25 km/h: above this a youth reading is a tracking error (1 Oct check: all >25 readings were wrong)
SMOOTH_S = 1.0         # 1 Oct: 0.5 s gave 117-134 m per player-minute (jitter adds metres); 1.0 s gives 96-115
STEP_S = 0.5
MIN_PIECE_S = 0.6      # shorter pieces get no speed (too little to smooth)
JUMP_M = 3.0           # one-frame jump this big = ID swap / bad calibration, not running


def _pieces(ks, P, fps):
    cut = [0]
    for i in range(1, len(ks)):
        dt = (ks[i] - ks[i - 1]) / fps
        if dt > 0.5 or np.linalg.norm(P[i] - P[i - 1]) > JUMP_M: cut.append(i)          # frame-to-frame jitter is big far away; only a real jump splits
    cut.append(len(ks))
    return [(cut[j], cut[j + 1]) for j in range(len(cut) - 1)]


def compute(per, fps):
    tracks = {}
    for k, rows in per.items():
        for r in rows:
            if r[2] is None: continue
            tracks.setdefault(int(r[0]), []).append((int(k), np.asarray(r[2], float)))
    out = {}; w = max(1, int(round(SMOOTH_S * fps / 2))); st = max(1, int(round(STEP_S * fps)))
    for tid, v in tracks.items():
        v.sort(key=lambda z: z[0]); ks = np.array([a for a, _ in v]); P = np.array([b for _, b in v]); dist = 0.0
        for a, b in _pieces(ks, P, fps):
            pk, pp = ks[a:b], P[a:b]
            if (pk[-1] - pk[0]) / fps < MIN_PIECE_S:
                for k in pk: out.setdefault(int(k), {})[tid] = (None, round(dist, 1))
                continue
            cs = np.vstack([np.zeros(2), np.cumsum(pp, 0)]); idx = np.arange(len(pp))
            lo, hi = np.maximum(0, idx - w), np.minimum(len(pp), idx + w + 1)
            S = (cs[hi] - cs[lo]) / (hi - lo)[:, None]                                       # centred moving average
            step = np.r_[0.0, np.linalg.norm(np.diff(S, axis=0), axis=1)]
            j0, j1 = np.maximum(0, idx - st // 2), np.minimum(len(pp) - 1, idx + st - st // 2)
            dt = (pk[j1] - pk[j0]) / fps
            sp = np.where(dt > 0, np.linalg.norm(S[j1] - S[j0], axis=1) / np.where(dt > 0, dt, 1), np.nan)
            for i, k in enumerate(pk):
                if sp[i] <= MAX_MS: dist += step[i]
                kmh = None if not np.isfinite(sp[i]) or sp[i] > MAX_MS else round(float(sp[i]) * 3.6, 1)
                out.setdefault(int(k), {})[tid] = (kmh, round(dist, 1))
    return out


def summary(mo, fps):
    """sanity numbers for the run log: speed percentiles, share without a speed, longest distance"""
    sp = [v[0] for f in mo.values() for v in f.values() if v[0] is not None]
    none = sum(1 for f in mo.values() for v in f.values() if v[0] is None); tot = none + len(sp)
    best = {}
    for f in mo.values():
        for t, v in f.items(): best[t] = max(best.get(t, 0), v[1])
    return {"kmh_p50_p95_max": [round(float(x), 1) for x in np.percentile(sp, [50, 95, 100])] if sp else None,
            "no_speed_pct": round(100 * none / max(1, tot), 1), "longest_track_m": round(max(best.values()), 1) if best else 0}
