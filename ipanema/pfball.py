"""PF1 (3 Oct): particle-filter ball picker, an alternative to the Viterbi path in ball.pick_v2.

Each particle is a guess of where the ball is, in picture pixels (camera pan removed between frames like pick_v2), with a
speed and a mode:
  0 ground  - rolls on, slowing a little; a guess projected off the pitch is unlikely
  1 air     - flies on with more freedom; off-pitch guesses are fine (a ball in the air projects outside the pitch)
  2 player  - stuck to one tracked player's feet (ball at feet / hidden in a crowd); the finder often misses it there
Every frame: move the particles, switch modes now and then, weigh them by the finder's guesses (same candidate rows and
static-clutter rule as pick_v2), resample, and drop a few fresh particles on strong guesses so a lost ball is found again.
The answer for frame t is read `lag` frames later from the particles' own histories (fixed-lag smoothing), snapped to the
finder guess with the most particle support; no guess near enough -> the particle cloud's centre if the cloud is tight.
Returns {frame: [x, y]} like pick_v2 (feed it to ball.bridge).
Result (tools/pf1lab.py, results/ball/pf1/README.md): best settings (these defaults) AIK 26-28/39, SFK-BP 26-27/34, B4 279-283
vs Viterbi 30, 29, 284 -> NOT used; option only. It drops all 8 fake passes but also 8-10 of 20 real ones.)"""
import numpy as np, cv2
from .calibration import to_m
from . import ball as BL

GROUND, AIR, PLAYER = 0, 1, 2


def _pan_vec(H, i, j, pos, vel):
    """carry positions and speeds from frame i pixels to frame j pixels"""
    if hasattr(H[i], "to_m") or hasattr(H[j], "to_m"): return pos, vel
    G = (H[j] @ np.linalg.inv(H[i])).astype(np.float32)
    a = cv2.perspectiveTransform(pos.astype(np.float32).reshape(-1, 1, 2), G).reshape(-1, 2)
    b = cv2.perspectiveTransform((pos + vel).astype(np.float32).reshape(-1, 1, 2), G).reshape(-1, 2)
    return a.astype(float), (b - a).astype(float)


def _feet(per, i):
    """{track id: foot pixel} for frame i"""
    return {r[0]: np.asarray(r[3], float) for r in (per.get(i) or []) if r[3] is not None} if per else {}


def pick_pf(cands, H, L, W, per=None, fps=30.0, n_part=600, seed=0, margin=1.5, min_conf=0.08, top_k=12,
            sigma_px=20.0, conf_pow=3.0, eps=0.02, pmiss=(0.35, 0.6, 0.7), off_pen=0.15, air_lik=0.6,
            vg=2.5, va=5.0, damp=0.97, p_ga=0.01, p_ag=0.04, p_gp=0.15, p_pg=0.04, foot_r=40.0, foot_sd=10.0,
            inject=0.01, lag=15, snap_px=40.0, min_support=0.15, spread_px=30.0, keep_free=False, log=print):
    rng = np.random.default_rng(seed)
    n = len(cands)
    C = BL.v2_rows(cands, H, L, W, per=per, margin=margin, min_conf=min_conf, top_k=top_k)
    C, dropped = BL.drop_static(C, fps)
    N = n_part
    pos = np.zeros((N, 2)); vel = np.zeros((N, 2)); mode = np.zeros(N, int); pid = np.full(N, -1, int)
    hist = np.zeros((N, lag + 1, 2))           # each particle's own last lag+1 positions (frame pixels of that time)
    hmode = np.zeros((N, lag + 1), int)
    born = np.zeros(N, int)                    # frame a particle (its line) was created: older history is made up
    w = np.full(N, 1.0 / N); alive = False
    ball = {}; mode_count = np.zeros(3)

    def seed_at(idx, i, rows):
        """put particles idx on guesses of frame i, chosen by confidence"""
        if not rows or not len(idx): return False
        q = np.array([max(r[2], 1e-3) for r in rows]) ** conf_pow; q /= q.sum()
        j = rng.choice(len(rows), size=len(idx), p=q)
        pos[idx] = np.array([[rows[k][3], rows[k][4]] for k in j]) + rng.normal(0, 2.0, (len(idx), 2))
        vel[idx] = rng.normal(0, 3.0, (len(idx), 2))
        mode[idx] = np.where([rows[k][6] for k in j], GROUND, AIR); pid[idx] = -1
        return True

    for i in range(n):
        rows = C[i]; feet = _feet(per, i)
        if not alive:
            if seed_at(np.arange(N), i, rows):
                alive = True; born[:] = i; w[:] = 1.0 / N; hist[:] = pos[:, None, :]; hmode[:] = mode[:, None]
        else:
            # --- move ---
            pos, vel = _pan_vec(H, i - 1, i, pos, vel)
            u = rng.random(N)
            g = mode == GROUND; a = mode == AIR; pl = mode == PLAYER
            # mode switches
            to_air = g & (u < p_ga); to_ground = a & (u < p_ag)
            mode[to_air] = AIR; mode[to_ground] = GROUND
            if feet:
                ids = list(feet); F = np.array([feet[k] for k in ids])
                gi = np.flatnonzero(g & ~to_air)
                if len(gi):
                    d = np.linalg.norm(pos[gi, None, :] - F[None, :, :], axis=2); jn = d.argmin(1)
                    take = (d[np.arange(len(gi)), jn] < foot_r) & (rng.random(len(gi)) < p_gp)
                    mode[gi[take]] = PLAYER; pid[gi[take]] = np.array(ids)[jn[take]]
            kick = (mode == PLAYER) & (rng.random(N) < p_pg)
            mode[kick] = GROUND; vel[kick] = rng.normal(0, 8.0, (int(kick.sum()), 2))
            # motion per mode
            g = mode == GROUND; a = mode == AIR; pl = mode == PLAYER
            vel[g] = damp * vel[g] + rng.normal(0, vg, (int(g.sum()), 2))
            vel[a] = vel[a] + rng.normal(0, va, (int(a.sum()), 2))
            pos[g | a] += vel[g | a] + rng.normal(0, 2.0, (int((g | a).sum()), 2))
            pk = np.flatnonzero(pl)
            if len(pk):
                f_idx = {k: j for j, k in enumerate(feet)}; F = np.array(list(feet.values())) if feet else np.zeros((0, 2))
                jj = np.array([f_idx.get(int(p), -1) for p in pid[pk]])
                lost = pk[jj < 0]; mode[lost] = GROUND; pid[lost] = -1
                ok = pk[jj >= 0]
                if len(ok):
                    new = F[jj[jj >= 0]] + rng.normal(0, foot_sd, (len(ok), 2)); vel[ok] = new - pos[ok]; pos[ok] = new
            # --- weigh ---
            lik = np.full(N, eps)
            pm = np.array(pmiss)[mode]
            if rows:
                Q = np.array([[r[3], r[4]] for r in rows]); cf = np.array([max(r[2], 1e-3) for r in rows]) ** conf_pow
                on = np.array([r[6] for r in rows], bool)
                d2 = ((pos[:, None, :] - Q[None, :, :]) ** 2).sum(2)
                k_ = np.exp(-d2 / (2 * sigma_px ** 2))
                gw = np.where(on, 1.0, off_pen)[None, :] * np.ones((N, 1))       # ground: off-pitch guesses unlikely
                gw[mode == AIR] = air_lik                                           # air: anywhere, but seen less sharply
                lik = pm * eps + (1 - pm) * (k_ * gw * cf[None, :]).sum(1)
            else:
                lik = pm * eps
            # a ground particle whose own position projects far off the pitch is unlikely
            gi = np.flatnonzero(mode != AIR)
            if len(gi):
                m = to_m(H[i], pos[gi])
                off = ~((m[:, 0] > -margin) & (m[:, 0] < L + margin) & (m[:, 1] > -margin) & (m[:, 1] < W + margin))
                lik[gi[off]] *= off_pen
            w = w * lik; s = w.sum()
            if not np.isfinite(s) or s <= 0: w[:] = 1.0 / N
            else: w /= s
            # --- resample (systematic) when the weights collapse ---
            if 1.0 / (w ** 2).sum() < N / 2:
                cs = np.cumsum(w); cs[-1] = 1.0
                idx = np.searchsorted(cs, (rng.random() + np.arange(N)) / N)
                pos, vel, mode, pid, hist, hmode, born = pos[idx], vel[idx], mode[idx].copy(), pid[idx].copy(), hist[idx], hmode[idx], born[idx]
                w[:] = 1.0 / N
            # --- fresh particles on strong guesses (a lost ball is found again) ---
            if rows and inject > 0:
                k = int(inject * N); idx = rng.choice(N, size=k, replace=False)
                if seed_at(idx, i, rows):
                    hist[idx] = pos[idx][:, None, :]; hmode[idx] = mode[idx][:, None]; born[idx] = i
                    w[idx] = 1.0 / N; w /= w.sum()
            hist = np.roll(hist, -1, axis=1); hist[:, -1] = pos
            hmode = np.roll(hmode, -1, axis=1); hmode[:, -1] = mode
        if not alive: continue
        # --- answer for frame t = i - lag, from the particles' histories ---
        t = i - lag
        if t >= 0: _answer(t, hist[:, 0], hmode[:, 0], w * (born <= t), C[t], ball, mode_count, snap_px, sigma_px, min_support, spread_px, keep_free)
    # the last lag frames: read straight from the histories
    if alive:
        for back in range(lag, 0, -1):
            t = n - back
            if t >= 0: _answer(t, hist[:, lag - back + 1], hmode[:, lag - back + 1], w * (born <= t), C[t], ball, mode_count, snap_px, sigma_px, min_support, spread_px, keep_free)
    tot = max(1.0, mode_count.sum())
    log(f"ball pf: {len(ball)}/{n} frames on the path, {dropped} static-clutter candidates dropped, "
        f"particle modes ground {mode_count[0] / tot:.0%} air {mode_count[1] / tot:.0%} player {mode_count[2] / tot:.0%}")
    return ball


def _answer(t, P, M, w, rows, ball, mode_count, snap_px, sigma_px, min_support, spread_px, keep_free):
    s = w.sum()
    if s <= 0: return
    w = w / s
    for m in range(3): mode_count[m] += w[M == m].sum()
    if rows:
        Q = np.array([[r[3], r[4]] for r in rows])
        d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
        sup = (w[:, None] * (d < snap_px)).sum(0)
        j = int(sup.argmax())
        if sup[j] >= min_support: ball[t] = [rows[j][3], rows[j][4]]; return
    if keep_free:
        c = (w[:, None] * P).sum(0)
        if np.sqrt((w * ((P - c) ** 2).sum(1)).sum()) < spread_px: ball[t] = [float(c[0]), float(c[1])]
