"""28 Sep: kits learned per match, whatever the colours or the light (Daniel: must hold on ALL our footage).
Every person's torso colour (grass removed, in Lab so light and colour are separate) is clustered within ONE match.
The two biggest groups are the teams; a person far from both team colours is 'other' (referee, keepers, staff).
Nothing is hard-coded about orange, black or white: a floodlit yellow cast shifts everyone the same way."""
import numpy as np, cv2

def on_grass(frame, box, grass=None, d=22.0):
    """P4: are this person's feet on the pitch? (the strip just below the box is grass-coloured) - no calibration needed;
    spectators / bench behind the line stand on track, fence or dark ground"""
    x1, y1, x2, y2 = [int(v) for v in box]; h = frame.shape[0]; w = max(4, x2 - x1)
    s = frame[min(h - 1, y2):min(h, y2 + max(3, int(0.08 * (y2 - y1)))), max(0, x1 - w // 4):x2 + w // 4]
    if s.size == 0: return True                                               # at the bottom edge: near side, on the pitch
    g = grass_lab(frame) if grass is None else grass
    lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    return bool((np.linalg.norm(lab - g, axis=1) <= d).mean() >= 0.5)

def grass_lab(frame):
    """this pitch's grass colour: median Lab of the lower third of the frame (mostly pitch on the follow-cam)"""
    h = frame.shape[0]; lab = cv2.cvtColor(frame[int(0.66 * h):], cv2.COLOR_BGR2LAB).reshape(-1, 3)[::7].astype(float)
    return np.median(lab, axis=0)

def torso_feature(frame, box, grass=None, min_px=6, grass_d=16.0, green_kit=False):
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
    L, a, b = np.median(lab, axis=0)
    return np.array([L / 2.5, a - 128.0, b - 128.0])                          # lightness counts less than colour (shadows)

def _kmeans(X, k, seed=0, iters=40):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(C2, C): break
        C = C2
    return C, lab

def fit(features, k=4, seeds=8, merge_d=18.0):
    """-> {"teams": [centre A, centre B], "spread": typical distance of a team member to its centre}"""
    X = np.array([f for f in features if f is not None], float)
    best = None
    for s in range(seeds):
        C, lab = _kmeans(X, k, s); cost = ((X - C[lab]) ** 2).sum()
        if best is None or cost < best[0]: best = (cost, C, lab)
    _, C, lab = best; sizes = np.bincount(lab, minlength=k); order = list(np.argsort(-sizes))
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
    return {"teams": [ta, tb], "sizes": sorted(sizes.tolist(), reverse=True), "spread": float(np.percentile(near, 80)) if len(near) else 10.0}

def classify(model, f, other_factor=2.0):
    """'A' / 'B' (index of the team centre) or 'other' when far from both, or None when no colour could be read"""
    if f is None: return None
    ta, tb = model["teams"]; da, db = np.linalg.norm(f - ta), np.linalg.norm(f - tb)
    if min(da, db) > max(8.0, other_factor * model["spread"]): return "other"
    return "A" if da <= db else "B"

class KitTeamModel:
    """drop-in for teams.TeamModel (predict_batch, dark_share, strips): kits learned from THIS match's frames.
    'A' = the darker team (pipeline convention), 'B' = the lighter, 'K' = neither (referee / keepers / staff)."""
    def fit_frames(self, frames_boxes, log=print, pitch_only=True, green_kit=False):   # pitch_only default since 28 Sep (P4: cleaner on Reymersholm, Spånga, SFK)
        feats = []; self.samples = []; self.green_kit = green_kit; dropped = 0
        for f, boxes in frames_boxes:
            g = grass_lab(f)
            for b in boxes:
                if b[3] - b[1] < 22: continue
                if pitch_only and not on_grass(f, b, g): dropped += 1; continue
                ft = torso_feature(f, b, g, green_kit=green_kit)
                if ft is not None: feats.append(ft); self.samples.append((f, b, ft))
        self.model = fit(feats); ta, tb = self.model["teams"]
        self.swap = ta[0] > tb[0]                                              # teams[0] lighter -> it is "B"
        self.dark_share = {"A": round(float(min(ta[0], tb[0]) * 2.5 / 255), 2), "B": round(float(max(ta[0], tb[0]) * 2.5 / 255), 2)}
        self._strips(); log(f"kits: learned from {len(feats)} people ({dropped} off the pitch left out), group sizes {self.model['sizes']}"); return self
    def _lab(self, f):
        c = classify(self.model, f)
        if c is None: return None
        if c == "other": return "K"
        return {"A": "B", "B": "A"}[c] if self.swap else c
    def predict_batch(self, frame, xyxys):
        g = grass_lab(frame); return [self._lab(torso_feature(frame, b, g, green_kit=getattr(self, "green_kit", False))) for b in xyxys]
    def _strips(self):
        import cv2
        self.strips = {}
        for t in ("A", "B"):
            cs = [cv2.resize(f[int(b[1]):int(b[3]), int(b[0]):int(b[2])], (48, 72)) for f, b, ft in self.samples if self._lab(ft) == t and b[3] - b[1] > 30][:24]
            cs = cs or [np.zeros((72, 48, 3), np.uint8)]
            while len(cs) % 12: cs.append(np.zeros_like(cs[0]))
            self.strips[t] = np.vstack([np.hstack(cs[r:r + 12]) for r in range(0, len(cs), 12)])
