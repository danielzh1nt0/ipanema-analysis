"""P3 (5 Oct): tracklet joining after tracking (idea from gta-link, MIT: join broken pieces of the same player by
look + where/when, never two pieces that are on screen at the same time). Option only - not wired into the pipeline.

Input: rows per frame {k: [(id, team, px(x,y) or None, filled, box_h or None, m(x,y) or None), ...]} (observed rows only
are used). Works on calibrated grounds (metres) and on uncalibrated follow-cam pieces (pixels with the camera pan taken
out = median shift of the people seen on both frames, px -> m from the box height, a player ~1.8 m).

join(...) -> {old_id: new_id}; every new_id is the earliest piece of its chain."""
import numpy as np

PLAYER_M = 1.8


def pieces(rows, fps):
    """rows -> {id: dict(k=[...], p=[[x, y]...] (metres or stabilised px), h=[box_h], team=majority)} using observed rows."""
    out = {}
    for k in sorted(rows):
        for r in rows[k]:
            if r[3]: continue
            d = out.setdefault(r[0], {"k": [], "px": [], "h": [], "m": [], "teams": {}})
            d["k"].append(k); d["px"].append(r[2]); d["h"].append(r[4] if len(r) > 4 else None); d["m"].append(r[5] if len(r) > 5 else None)
            d["teams"][r[1]] = d["teams"].get(r[1], 0) + 1
    for d in out.values():
        d["team"] = max(d["teams"], key=d["teams"].get); d["start"], d["end"] = d["k"][0], d["k"][-1]
    return out


def camera_shift(rows):
    """cumulative screen offset per frame: median px shift of the ids seen on both k-1 and k (follow-cam pan)."""
    ks = sorted(rows); off = {}; cur = np.zeros(2); prev = None
    for k in ks:
        now = {r[0]: np.asarray(r[2], float) for r in rows[k] if not r[3] and r[2] is not None}
        if prev is not None:
            common = [i for i in now if i in prev]
            if len(common) >= 3: cur = cur + np.median([now[i] - prev[i] for i in common], axis=0)
        off[k] = cur.copy(); prev = now
    return off


def _ends(d, fps, use_m, off, side, n=None):
    """position + velocity (units/s) at a piece's start or end, from up to ~0.5 s of it."""
    n = n or max(2, int(0.5 * fps))
    idx = list(range(len(d["k"])))[-n:] if side == "end" else list(range(len(d["k"])))[:n]
    ks = np.array([d["k"][i] for i in idx], float)
    if use_m: P = np.array([d["m"][i] for i in idx], float)
    else: P = np.array([np.asarray(d["px"][i], float) - off[d["k"][i]] for i in idx])
    j = -1 if side == "end" else 0
    v = (P[-1] - P[0]) / max((ks[-1] - ks[0]) / fps, 1e-6) if len(ks) > 1 else np.zeros(2)
    return P[j], v


def _scale(d):
    h = [x for x in d["h"] if x]
    return (np.median(h) / PLAYER_M) if h else 60.0


def candidates(P, fps, rows=None, use_m=False, max_gap_s=5.0, speed=7.0, slack_m=2.0, vmax=8.0):
    """all (a, b, gap_s, dist_m) where b starts after a ends (same team), within the reach of a running player."""
    off = camera_shift(rows) if (rows is not None and not use_m) else None
    E = {i: _ends(d, fps, use_m, off, "end") for i, d in P.items()}
    S = {i: _ends(d, fps, use_m, off, "start") for i, d in P.items()}
    out = []
    for a, A in P.items():
        for b, B in P.items():
            if a == b or A["team"] != B["team"]: continue
            gap = (B["start"] - A["end"]) / fps
            if not (0 < gap <= max_gap_s): continue
            pa, va = E[a]; pb, _ = S[b]
            sc = 1.0 if use_m else (_scale(A) + _scale(B)) / 2
            vn = np.linalg.norm(va) / sc
            va = va * (min(vn, vmax) / vn) if vn > 1e-6 else va
            pred = pa + va * min(gap, 1.0)                                  # carry the last velocity for at most 1 s
            dist = min(np.linalg.norm(pb - pred), np.linalg.norm(pb - pa)) / sc
            if dist <= slack_m + speed * gap: out.append((a, b, round(gap, 2), round(float(dist), 2)))
    return out


def join(P, cands, emb=None, max_app=None, w_app=1.0, w_dist=0.15, w_gap=0.2):
    """greedy merge, cheapest first: cost = w_app * look distance + w_dist * metres + w_gap * seconds.
    A pair is used only if the chains never share a frame and a's chain end / b's chain start are still free.
    emb: {id: unit vector} (look); pairs whose look distance > max_app are refused (None = look not used)."""
    def app(a, b):
        if emb is None or a not in emb or b not in emb: return 0.0
        return float(1 - np.dot(emb[a], emb[b]))
    scored = []
    for a, b, gap, dist in cands:
        ad = app(a, b)
        if max_app is not None and emb is not None and (a not in emb or b not in emb or ad > max_app): continue
        scored.append((w_app * ad + w_dist * dist + w_gap * gap, a, b, ad))
    scored.sort()
    nxt, prv, root = {}, {}, {i: i for i in P}
    span = {i: [(P[i]["start"], P[i]["end"])] for i in P}
    def r(i):
        while root[i] != i: i = root[i]
        return i
    used = []
    for c, a, b, ad in scored:
        if a in nxt or b in prv: continue
        ra, rb = r(a), r(b)
        if ra == rb: continue
        if any(s1 <= e2 and s2 <= e1 for s1, e1 in span[ra] for s2, e2 in span[rb]): continue
        nxt[a] = b; prv[b] = a; root[rb] = ra; span[ra] += span.pop(rb); used.append((a, b, round(c, 3), round(ad, 3)))
    new = {}
    for i in P:                                                             # chain head = earliest piece: walk back
        j = i
        while j in prv: j = prv[j]
        new[i] = j
    return new, used


def measure(P, fps, n_frames, remap=None, min_s=0.0):
    """pieces per 20 s, median s a piece lasts, share of player-time in pieces >= 5 s."""
    remap = remap or {i: i for i in P}
    ch = {}
    for i, d in P.items():
        ch.setdefault(remap[i], []).append(d)
    lens = [sum(len(d["k"]) for d in ds) / fps for ds in ch.values()]
    spans = [(max(d["end"] for d in ds) - min(d["start"] for d in ds) + 1) / fps for ds in ch.values()]
    tot = sum(lens)
    per20 = len(ch) * (20.0 * fps / max(n_frames, 1))
    return {"pieces_per_20s": round(per20, 1), "median_s": round(float(np.median(lens)), 2) if lens else None,
            "median_span_s": round(float(np.median(spans)), 2) if spans else None,
            "share_time_in_5s_plus": round(sum(l for l in lens if l >= 5) / tot, 3) if tot else None}
