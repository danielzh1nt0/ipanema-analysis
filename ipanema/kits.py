"""28 Sep: kits learned per match, whatever the colours or the light (Daniel: must hold on ALL our footage).
Every person's torso colour (grass removed, in Lab so light and colour are separate) is clustered within ONE match.
The two biggest groups are the teams; a person far from both team colours is 'other' (referee, keepers, staff).
Nothing is hard-coded about orange, black or white: a floodlit yellow cast shifts everyone the same way."""
import os, numpy as np, cv2

def on_grass(frame, box, grass=None, d=22.0):
    """P4: are this person's feet on the pitch? (the strip just below the box is grass-coloured) - no calibration needed;
    spectators / bench behind the line stand on track, fence or dark ground"""
    x1, y1, x2, y2 = [int(v) for v in box]; h = frame.shape[0]; w = max(4, x2 - x1)
    s = frame[min(h - 1, y2):min(h, y2 + max(3, int(0.08 * (y2 - y1)))), max(0, x1 - w // 4):x2 + w // 4]
    if s.size == 0: return True                                               # at the bottom edge: near side, on the pitch
    g = grass_lab(frame) if grass is None else grass
    lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    return bool((np.linalg.norm(lab - g, axis=1) <= d).mean() >= 0.5)

def pitch_top(frame, s=4, chroma_d=14.0, dark=0.45):
    """P7 (29 Sep): per image column, the row where the pitch's grass ends at the top (far touchline / run-off edge).
    Grass = colour close to this pitch's grass in a*b* (lightness mostly ignored: floodlit grass mid-pitch is much brighter
    than the reference, which made on_grass drop almost every on-pitch player at night) and not too dark (fence, hedges,
    dark ground behind the line). The grass area touching the bottom of the frame is the pitch. No calibration needed."""
    H, W = frame.shape[:2]; small = cv2.resize(frame, (max(8, W // s), max(8, H // s)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(float); g = grass_lab(frame)
    m = ((np.hypot(lab[..., 1] - g[1], lab[..., 2] - g[2]) <= chroma_d) & (lab[..., 0] >= dark * g[0])).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m); h, w = m.shape
    bottom = [i for i in set(lb[h - 1].tolist()) if i > 0]
    if not bottom: return np.zeros(W)                                          # no grass at the bottom: don't drop anyone
    comp = lb == max(bottom, key=lambda i: st[i, cv2.CC_STAT_AREA])
    top = np.where(comp.any(0), comp.argmax(0), h).astype(np.float32)
    top = cv2.medianBlur(top.reshape(1, -1), 5).ravel()
    return np.interp(np.arange(W), np.arange(w) * s + s / 2, top * s)

def feet_on_pitch(frame, box, top=None, margin=0.011):
    """P7: feet at least ~1% of the frame height below the pitch's top edge. Graded by eye on the 24 Reymersholm frames
    (night): people behind the far line are at or above the edge (median -5 px), players on the pitch well below."""
    t = pitch_top(frame) if top is None else top
    x = int(min(len(t) - 1, max(0, (box[0] + box[2]) / 2)))
    return bool(box[3] - t[x] >= margin * frame.shape[0])

def grass_lab(frame):
    """this pitch's grass colour: median Lab of the lower third of the frame (mostly pitch on the follow-cam)"""
    h = frame.shape[0]; lab = cv2.cvtColor(frame[int(0.66 * h):], cv2.COLOR_BGR2LAB).reshape(-1, 3)[::7].astype(float)
    return np.median(lab, axis=0)

def torso_feature(frame, box, grass=None, min_px=6, grass_d=16.0, green_kit=False, stat=None):
    """median torso colour with THIS pitch's grass removed (28 Sep: removing all green would delete a green kit).
    green_kit (P4, 28 Sep night): if most of the torso reads as grass, it IS the shirt (green kit under floodlights) ->
    keep those pixels instead of what is left (shorts, numbers, skin looked like the white team)."""
    x1, y1, x2, y2 = [float(v) for v in box]; w, h = x2 - x1, y2 - y1
    c = frame[int(y1 + 0.20 * h):int(y1 + 0.48 * h), int(x1 + 0.30 * w):int(x1 + 0.70 * w) + 1]
    if c.size == 0: return None
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    g = grass_lab(frame) if grass is None else grass
    keep = np.linalg.norm(lab - g, axis=1) > grass_d
    if green_kit and keep.mean() < 0.4 and (~keep).sum() >= min_px: lab = lab[~keep]
    else: lab = lab[keep]
    if len(lab) < min_px: return None
    stat = stat or os.environ.get("IPANEMA_TORSO_STAT", "median")
    if stat == "mean": L, a, b = lab.mean(axis=0)
    elif stat == "trim": lo, hi = np.percentile(lab, [10, 90], axis=0); L, a, b = [lab[(lab[:, i] >= lo[i]) & (lab[:, i] <= hi[i]), i].mean() for i in range(3)]
    else: L, a, b = np.median(lab, axis=0)
    return np.array([L / 2.5, a - 128.0, b - 128.0])                          # lightness counts less than colour (shadows)

def body_hist(frame, box, bins=5):
    """P8 (29 Sep): colour histogram of the upper body (shirt area incl. its edges), lightness scaled by this frame's grass,
    square-rooted. A blurred floodlit white shirt has a torso MEDIAN between the kits, but still many bright pixels -
    the histogram keeps that. 5x5x5 bins (L, a, b)."""
    x1, y1, x2, y2 = [float(v) for v in box]; w, h = x2 - x1, y2 - y1
    c = frame[int(max(0, y1 + 0.12 * h)):int(max(0, y1 + 0.55 * h)), int(max(0, x1 + 0.15 * w)):int(max(0, x1 + 0.85 * w)) + 1]
    if c.size == 0: return None
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float); gL = grass_lab(frame)[0]
    Ln = np.clip(lab[:, 0] / (2 * gL + 1e-6), 0, 0.999); a = np.clip((lab[:, 1] - 88) / 80, 0, 0.999); b = np.clip((lab[:, 2] - 88) / 80, 0, 0.999)
    idx = (Ln * bins).astype(int) * bins * bins + (a * bins).astype(int) * bins + (b * bins).astype(int)
    return np.sqrt(np.bincount(idx, minlength=bins ** 3) / len(idx))

def _logreg(X, y, C=10.0, iters=30):
    """small L2 logistic regression (Newton steps), classes weighted equally; -> (w, b)"""
    Xb = np.c_[X, np.ones(len(X))]; wt = np.where(y == 1, len(y) / (2 * max(1, y.sum())), len(y) / (2 * max(1, (1 - y).sum())))
    th = np.zeros(Xb.shape[1]); reg = np.eye(Xb.shape[1]) / C; reg[-1, -1] = 1e-8
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ th, -30, 30))); g = Xb.T @ (wt * (p - y)) + reg @ th
        Hm = (Xb * (wt * p * (1 - p))[:, None]).T @ Xb + reg; step = np.linalg.solve(Hm, g); th -= step
        if np.abs(step).max() < 1e-6: break
    return th[:-1], th[-1]

def fit_player_cls(H, lab, ks=(4, 5, 6, 7, 8), seeds=(0, 1, 2), gain=0.05, max_off=0.15):
    """P8 (29 Sep): per-match team classifier on body_hist, no labels needed. For each k-means run on the histograms of
    the on-pitch people the colour model reads as a team: team A seed = the biggest group it mostly calls A, team B seed =
    the biggest group it mostly calls B; a logistic regression is trained on those two groups.
    Check (both teams have about as many players on the pitch): a run is kept only if its A/B split of these people is
    nearer 50/50 than the colour model's (by >= 5 points) and within 15 points of 50/50. No run kept -> colour stays.
    Graded by eye (results/qa/p8): Reymersholm night 239/252 players right vs 195 with colour alone; Spånga unchanged
    (without the check it got worse, 177 vs 181/279). H: histograms (n, d); lab: 'A'/'B' per person. -> list of (w, b)"""
    H = np.asarray(H, float); lab = np.asarray(lab); models = []
    if len(H) < 30 or not ((lab == "A").any() and (lab == "B").any()): return models
    colour_off = abs((lab == "B").mean() - 0.5)
    for k in ks:
        if len(H) < 3 * k: continue
        for s in seeds:
            best = None
            for r in range(4):                                                  # a few restarts, keep the tightest
                C, cl = _kmeans(H, k, 100 * s + r); cost = ((H - C[cl]) ** 2).sum()
                if best is None or cost < best[0]: best = (cost, cl)
            cl = best[1]; sz = np.bincount(cl, minlength=k); order = np.argsort(-sz); vote = {}
            for c in order:
                v = lab[cl == c]
                if len(v): vote[c] = "B" if (v == "B").sum() > (v == "A").sum() else "A"
            ca = [c for c in order if vote.get(c) == "A"]; cb = [c for c in order if vote.get(c) == "B"]
            if not ca or not cb: continue
            idx = np.isin(cl, [ca[0], cb[0]]); w, b = _logreg(H[idx], (cl[idx] == cb[0]).astype(float))
            off = abs((H @ w + b > 0).mean() - 0.5)
            if off <= colour_off - gain and off <= max_off: models.append((w, b))
    return models

def player_cls_prob(models, h):
    """mean probability (over the voting runs) that this person is team B"""
    return float(np.mean([1 / (1 + np.exp(-np.clip(h @ w + b, -30, 30))) for w, b in models]))

def _kmeans(X, k, seed=0, iters=40):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(C2, C): break
        C = C2
    return C, lab

def _fit_far(X, C, lab, sizes, order, merge_d, min_share=0.25, l_weight=0.5):
    """P7 (29 Sep): team A = biggest colour group; team B = the big group (>= 25% of A's size) LEAST like A. Other groups join
    the nearer team if close, lightness counting half (sun/shade, floodlit/dark splits of one kit differ mainly in
    lightness), else they are 'other'. The old rule took the two biggest groups as the teams; at night on Reymersholm
    those were dark-green and floodlit-green players (same kit) and the whites were pushed out."""
    a = order[0]; big = [j for j in order[1:] if sizes[j] >= min_share * sizes[a]] or order[1:2]
    b = max(big, key=lambda j: np.linalg.norm(C[j] - C[a])); team_of = {a: 0, b: 1}
    w = np.array([l_weight, 1.0, 1.0] + [1.0] * (C.shape[1] - 3))
    for j in order:
        if j in team_of or sizes[j] == 0: continue
        da, db = np.linalg.norm((C[j] - C[a]) * w), np.linalg.norm((C[j] - C[b]) * w)
        if min(da, db) <= merge_d: team_of[j] = 0 if da <= db else 1
    ta = X[np.isin(lab, [j for j, t in team_of.items() if t == 0])].mean(0); tb = X[np.isin(lab, [j for j, t in team_of.items() if t == 1])].mean(0)
    d = np.minimum(np.linalg.norm(X - ta, axis=1), np.linalg.norm(X - tb, axis=1)); near = d[np.isin(lab, list(team_of))]
    return {"teams": [ta, tb], "sizes": sorted(sizes.tolist(), reverse=True), "spread": float(np.percentile(near, 80)) if len(near) else 10.0,
            "spreads": _team_spreads(X, lab, team_of, ta, tb)}

def fit(features, k=4, seeds=8, merge_d=18.0, pair=None):
    """-> {"teams": [centre A, centre B], "spread": typical distance of a team member to its centre}
    pair: "big" (the two biggest groups, default) or "far" (P7 option: 200 vs 195/252 night players right on Reymersholm,
    not enough to change the default without a check on the day grounds). Default from IPANEMA_KIT_PAIR."""
    import os
    pair = pair or os.environ.get("IPANEMA_KIT_PAIR", "big")
    X = np.array([f for f in features if f is not None], float)
    best = None
    for s in range(seeds):
        C, lab = _kmeans(X, k, s); cost = ((X - C[lab]) ** 2).sum()
        if best is None or cost < best[0]: best = (cost, C, lab)
    _, C, lab = best; sizes = np.bincount(lab, minlength=k); order = list(np.argsort(-sizes))
    if pair == "far": return _fit_far(X, C, lab, sizes, order, merge_d)
    # a team often splits into two groups (sun/shade, near/far): merge any group whose colour is close to a team's
    team_of = {order[0]: 0}
    for j in order[1:]:
        dd = [np.linalg.norm(C[j] - C[t]) for t in team_of if team_of[t] == 0] + [np.inf]
        if 1 not in team_of.values() and min(dd) > merge_d: team_of[j] = 1
        elif min(dd) <= merge_d: team_of[j] = 0
        else:
            d1 = min(np.linalg.norm(C[j] - C[t]) for t in team_of if team_of[t] == 1)
            if d1 <= merge_d: team_of[j] = 1
    if 1 not in team_of.values(): team_of[order[1]] = 1
    ta = X[np.isin(lab, [j for j, t in team_of.items() if t == 0])].mean(0); tb = X[np.isin(lab, [j for j, t in team_of.items() if t == 1])].mean(0)
    order = [j for j, t in team_of.items() if t in (0, 1)]
    d = np.minimum(np.linalg.norm(X - ta, axis=1), np.linalg.norm(X - tb, axis=1))
    near = d[np.isin(lab, order)]
    sep = np.linalg.norm(ta - tb); big = [j for j in range(k) if sizes[j] >= 0.25 * sizes.max()]
    missed = max([min(np.linalg.norm(C[j] - ta), np.linalg.norm(C[j] - tb)) / max(sep, 1e-6) for j in big] + [0.0])
    return {"teams": [ta, tb], "sizes": sorted(sizes.tolist(), reverse=True), "spread": float(np.percentile(near, 80)) if len(near) else 10.0,
            "spreads": _team_spreads(X, lab, team_of, ta, tb), "missed": float(missed)}

def _team_spreads(X, lab, team_of, ta, tb):
    """P1 (29 Sep): each team's own spread (80th percentile distance of its members to its centre): a striped kit
    reads much less steadily than a plain one"""
    out = []
    for t, c in ((0, ta), (1, tb)):
        m = np.isin(lab, [j for j, tt in team_of.items() if tt == t]); dd = np.linalg.norm(X[m] - c, axis=1)
        out.append(float(np.percentile(dd, 80)) if len(dd) else 10.0)
    return out

def classify(model, f, other_factor=2.0):
    """'A' / 'B' (index of the team centre) or 'other' when far from both, or None when no colour could be read"""
    if f is None: return None
    ta, tb = model["teams"]; da, db = np.linalg.norm(f - ta), np.linalg.norm(f - tb)
    if os.environ.get("IPANEMA_KIT_SPREAD", "one") == "team" and "spreads" in model:      # P1 option: per-team spread
        sa, sb = [max(8.0, other_factor * x) for x in model["spreads"]]
        if da > sa and db > sb: return "other"
        return "A" if (da / sa if da > sa or db > sb else da) <= (db / sb if da > sa or db > sb else db) else "B"
    if min(da, db) > max(8.0, other_factor * model["spread"]): return "other"
    return "A" if da <= db else "B"

class KitTeamModel:
    """drop-in for teams.TeamModel (predict_batch, dark_share, strips): kits learned from THIS match's frames.
    'A' = the darker team (pipeline convention), 'B' = the lighter, 'K' = neither (referee / keepers / staff)."""
    offpitch = False    # P7: True on grounds with no calibration -> people whose feet are off the pitch read "O" (not counted)
    def fit_frames(self, frames_boxes, log=print, pitch_only=True, green_kit=False, pitch_test=None, player_cls=None):   # pitch_only default since 28 Sep (P4)
        """pitch_test: "edge" (P7, pitch_top/feet_on_pitch) or "grass" (P4 on_grass, wrong at night: dropped 321/414 people
        at Reymersholm, most of them players on the floodlit pitch). Default "grass": kits learned from the people it keeps
        still grade better (195/252 labelled night players vs 149 with "edge", results/qa/p7). Env IPANEMA_PITCH_TEST."""
        """P1 (29 Sep): striped kits. When the default kit fit leaves out a big colour group that is further from both
        team colours than the two teams are from each other (model["missed"] >= 1: at Spånga the bright striped team was
        not one of the two teams; SFK 0.36, Reymersholm 0.74), it tries the far-pair rule with the median and with a MEAN
        torso colour (a striped shirt's median jumps between its stripes) and keeps the one leaving the fewest of the
        fitting people as 'neither' (must beat the default by 3 points). Env IPANEMA_KIT_AUTO=0 turns it off.
        A first version that tried the alternatives on every match made SFK and one Reymersholm piece worse on Kaggle."""
        auto_gain = 0.03; missed_max = 1.0
        pitch_test = pitch_test or os.environ.get("IPANEMA_PITCH_TEST", "grass")
        self.green_kit = green_kit; dropped = 0; kept = []
        for f, boxes in frames_boxes:
            g = grass_lab(f); top = pitch_top(f) if (pitch_only and pitch_test == "edge") else None
            for b in boxes:
                if b[3] - b[1] < 22: continue
                if pitch_only and not (feet_on_pitch(f, b, top) if top is not None else on_grass(f, b, g)): dropped += 1; continue
                kept.append((f, b, g))
        auto = os.environ.get("IPANEMA_KIT_AUTO", "1") == "1"
        combos = [(None, None), ("far", "median"), ("far", "mean")] if auto else [(None, None)]
        best = None; self.choice = {}; self.choice_missed = 0.0
        for pair, stat in combos:
            if self.choice and self.choice_missed < missed_max: break              # default teams look right: keep them
            smp = [(f, b, torso_feature(f, b, g, green_kit=green_kit, stat=stat)) for f, b, g in kept]; smp = [x for x in smp if x[2] is not None]
            if len(smp) < 8: continue
            m = fit([x[2] for x in smp], pair=pair)
            if not self.choice: self.choice_missed = m.get("missed", 0.0)
            other = float(np.mean([classify(m, x[2]) == "other" for x in smp]))
            self.choice[f"{pair or 'default'}/{stat or 'default'}"] = round(other, 3)
            if best is None or other < best[0] - auto_gain: best = (other, m, smp, stat, pair)
        _, self.model, self.samples, self.stat, self.pair = best; feats = [x[2] for x in self.samples]; ta, tb = self.model["teams"]
        self.swap = ta[0] > tb[0]                                              # teams[0] lighter -> it is "B"
        self.dark_share = {"A": round(float(min(ta[0], tb[0]) * 2.5 / 255), 2), "B": round(float(max(ta[0], tb[0]) * 2.5 / 255), 2)}
        self.cls = []
        if (os.environ.get("IPANEMA_KIT_CLS", "1") if player_cls is None else ("1" if player_cls else "0")) == "1": self._fit_cls(frames_boxes, log)
        self._strips(); log(f"kits: learned from {len(feats)} people ({dropped} off the pitch left out), group sizes {self.model['sizes']}"
                             + f", default fit leaves out a group at {self.choice_missed:.2f} x the team gap" + (f", reading {self.pair}/{self.stat} (neither-share tried: {self.choice})" if len(self.choice) > 1 else "")); return self
    def _fit_cls(self, frames_boxes, log=print):
        """P8: per-player classifier on body histograms, trained on this match's on-pitch people (edge test)"""
        Hs, labs = [], []
        for f, boxes in frames_boxes:
            g = grass_lab(f); top = pitch_top(f)
            for b in boxes:
                if b[3] - b[1] < 22 or not feet_on_pitch(f, b, top): continue
                h, c = body_hist(f, b), self._colour_lab(torso_feature(f, b, g, green_kit=self.green_kit, stat=getattr(self, "stat", None)))
                if h is not None and c in ("A", "B"): Hs.append(h); labs.append(c)
        self.cls = fit_player_cls(Hs, labs)
        log(f"kits: per-player classifier from {len(Hs)} on-pitch people, {len(self.cls)} of 15 runs kept" + ("" if self.cls else " -> colour only"))
    def _colour_lab(self, f):
        c = classify(self.model, f)
        if c is None: return None
        if c == "other": return "K"
        return {"A": "B", "B": "A"}[c] if self.swap else c
    def _lab(self, f, h=None):
        """colour reading; with the P8 classifier the team (A/B) comes from the body histogram, 'other' still from colour"""
        c = self._colour_lab(f)
        if c in ("A", "B") and h is not None and getattr(self, "cls", None): return "B" if player_cls_prob(self.cls, h) > 0.5 else "A"
        return c
    def predict_batch(self, frame, xyxys):
        g = grass_lab(frame); top = pitch_top(frame) if self.offpitch and len(xyxys) else None; gk = getattr(self, "green_kit", False)
        cls = getattr(self, "cls", None)
        return ["O" if top is not None and not feet_on_pitch(frame, b, top) else self._lab(torso_feature(frame, b, g, green_kit=gk, stat=getattr(self, "stat", None)), body_hist(frame, b) if cls else None) for b in xyxys]
    def _strips(self):
        import cv2
        self.strips = {}
        for t in ("A", "B"):
            cs = [cv2.resize(f[int(b[1]):int(b[3]), int(b[0]):int(b[2])], (48, 72)) for f, b, ft in self.samples if self._lab(ft, body_hist(f, b) if self.cls else None) == t and b[3] - b[1] > 30][:24]
            cs = cs or [np.zeros((72, 48, 3), np.uint8)]
            while len(cs) % 12: cs.append(np.zeros_like(cs[0]))
            self.strips[t] = np.vstack([np.hstack(cs[r:r + 12]) for r in range(0, len(cs), 12)])
