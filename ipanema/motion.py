"""1 Oct: per-player speed (km/h) and distance (m) for the app's speed layer, from the tracked pitch positions.
Raw positions wobble (calibration, box jitter, ID swaps), so:
  1. split a track wherever it jumps (> JUMP_M in one frame, or a gap > 0.5 s): a jump is a tracking error,
     not running;
  2. smooth positions with a centred SMOOTH_S window;
  3. speed over a STEP_S baseline; readings above MAX_MS (36 km/h) are dropped (shown as no speed), not clipped;
  4. distance = sum of smoothed steps on valid stretches; a new piece starts from the distance the track had.
Output: {frame: {track_id: (kmh or None, metres_so_far)}}. Identity is per TRACK: tracks are short (median 4-8 s), so
these are not per-player match totals until tracks are joined into players.

M1b (1 Oct): when the calibration H is given, step 2-3 is a Kalman smoother instead: each foot point's noise in metres
= PX_NOISE pixels times the local metres-per-pixel of the calibration (0.05 m/px near the camera, 0.3-0.9 m/px at the
far touchline / when the camera is zoomed out), so far, noisy points are trusted less and the jitter no longer reads as
running. Speed = the smoothed velocity."""
import numpy as np

MAX_MS = 7.0           # 25 km/h: above this a youth reading is a tracking error (1 Oct check: all >25 readings were wrong)
SMOOTH_S = 1.0         # 1 Oct: 0.5 s gave 117-134 m per player-minute (jitter adds metres); 1.0 s gives 96-115
STEP_S = 0.5
MIN_PIECE_S = 0.6      # shorter pieces get no speed (too little to smooth)
JUMP_M = 3.0           # one-frame jump this big = ID swap / bad calibration, not running
PX_NOISE = 2.0         # M1b: foot-point jitter in pixels (box bottom + follow-cam calibration), measured ~2 px on SFK-BP
ACC_MS2 = 1.0          # M1b: how fast a player changes velocity (m/s per s); only PX_NOISE/ACC_MS2 matters: 2 px per m/s² was best of 5 ratios on the 21 by-eye moments


def _pieces(ks, P, fps):
    cut = [0]
    for i in range(1, len(ks)):
        dt = (ks[i] - ks[i - 1]) / fps
        if dt > 0.5 or np.linalg.norm(P[i] - P[i - 1]) > JUMP_M: cut.append(i)          # frame-to-frame jitter is big far away; only a real jump splits
    cut.append(len(ks))
    return [(cut[j], cut[j + 1]) for j in range(len(cut) - 1)]


def jacobians(per, H):
    """{(frame, track): 2x2 metres-per-pixel matrix at the foot point} from the calibration (numeric, any H with to_m)"""
    from .calibration import to_m
    J = {}
    for k, rows in per.items():
        rr = [r for r in rows if r[2] is not None and len(r) > 3 and r[3] is not None]
        if not rr or H is None or H.get(k) is None: continue
        p = np.array([r[3] for r in rr], float).reshape(-1, 2)
        try: m = to_m(H[k], np.vstack([p, p + [1.0, 0.0], p + [0.0, 1.0]])).reshape(3, -1, 2).astype(float)
        except Exception: continue
        for i, r in enumerate(rr):
            j = np.column_stack([m[1, i] - m[0, i], m[2, i] - m[0, i]])
            if np.all(np.isfinite(j)): J[(int(k), int(r[0]))] = j
    return J


def kalman(pk, pp, Js, fps, px_noise=None, acc=None):
    """constant-velocity Kalman filter + RTS smoother on one piece. pk frames, pp (n,2) metres, Js (n,2,2) metres/pixel.
    Returns smoothed positions (n,2) and velocities (n,2) in m/s."""
    px_noise = PX_NOISE if px_noise is None else px_noise; acc = ACC_MS2 if acc is None else acc
    n = len(pk); x = np.zeros((n, 4)); Pc = np.zeros((n, 4, 4)); xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4)); F = np.zeros((n, 4, 4))
    Hm = np.zeros((2, 4)); Hm[0, 0] = Hm[1, 1] = 1
    R = [px_noise ** 2 * (j @ j.T) + 1e-4 * np.eye(2) for j in Js]
    xc = np.r_[pp[0], 0.0, 0.0]; Pcur = np.diag([R[0][0, 0], R[0][1, 1], 9.0, 9.0])
    for i in range(n):
        if i:
            dt = (pk[i] - pk[i - 1]) / fps; Fi = np.eye(4); Fi[0, 2] = Fi[1, 3] = dt
            q = acc ** 2; Q = np.zeros((4, 4)); Q[np.ix_([0, 2], [0, 2])] = Q[np.ix_([1, 3], [1, 3])] = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
            xc = Fi @ xc; Pcur = Fi @ Pcur @ Fi.T + Q; F[i] = Fi
        xp[i], Pp[i] = xc, Pcur
        S = Pcur[:2, :2] + R[i]; K = Pcur[:, :2] @ np.linalg.inv(S)
        xc = xc + K @ (pp[i] - xc[:2]); Pcur = (np.eye(4) - K @ Hm) @ Pcur
        x[i], Pc[i] = xc, Pcur
    for i in range(n - 2, -1, -1):                                                        # RTS backward pass
        C = Pc[i] @ F[i + 1].T @ np.linalg.inv(Pp[i + 1])
        x[i] = x[i] + C @ (x[i + 1] - xp[i + 1]); Pc[i] = Pc[i] + C @ (Pc[i + 1] - Pp[i + 1]) @ C.T
    return x[:, :2], x[:, 2:]


def compute(per, fps, H=None):
    """H: per-frame calibration {frame: H}. Given -> noise-aware Kalman smoother (M1b); None -> 1 s moving average."""
    J = jacobians(per, H) if H is not None else {}
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
            if J:
                Js = [J.get((int(k), tid)) for k in pk]
                fill = next((j for j in Js if j is not None), None)
                if fill is not None:
                    Js = np.array([fill if j is None else j for j in Js])
                    S, V = kalman(pk, pp, Js, fps); sp = np.linalg.norm(V, axis=1)
                    step = np.r_[0.0, np.linalg.norm(np.diff(S, axis=0), axis=1)]
                    for i, k in enumerate(pk):
                        if sp[i] <= MAX_MS: dist += step[i]
                        out.setdefault(int(k), {})[tid] = (None if sp[i] > MAX_MS else round(float(sp[i]) * 3.6, 1), round(dist, 1))
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
