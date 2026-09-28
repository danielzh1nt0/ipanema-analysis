"""T0 (28 Sep): a small ball-vs-not scorer for finder guesses. Input: 3 frames (t-1, t, t+1) cropped 32x32 around a
guess -> 9 channels. Output: probability the guess is the ball. Trained on CPU from crops at Daniel's clicks (ball) and
the finders' other guesses (not ball). MIT-clean: our own small CNN, our own data."""
import numpy as np


def to_tensor(X):
    """(N, 3, S, S, 3) uint8 -> (N, 9, S, S) float32 in 0..1, plus the frame differences the model sees as motion"""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(X)).float() / 255.0          # N,3,S,S,3
    x = x.permute(0, 1, 4, 2, 3).reshape(X.shape[0], 9, X.shape[2], X.shape[3])
    return x


def model(width=32):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(9, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
        nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),          # 16
        nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(), nn.MaxPool2d(2),  # 8
        nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * width, 1))


def augment(x, rng):
    """flips, small shifts, brightness: the ball can be anywhere in its crop and in any light"""
    import torch
    if rng.random() < 0.5: x = torch.flip(x, [3])
    if rng.random() < 0.5: x = torch.flip(x, [2])
    dx, dy = rng.integers(-3, 4, 2); x = torch.roll(x, (int(dy), int(dx)), (2, 3))
    return (x * float(rng.uniform(0.75, 1.25))).clamp(0, 1)


def train(X, y, epochs=15, lr=2e-3, seed=0, batch=128, log=print):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = model(); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    xt = to_tensor(X); yt = torch.tensor(y, dtype=torch.float32)
    pw = torch.tensor([(yt == 0).sum().item() / max(1, (yt == 1).sum().item())])                   # balance ball vs not
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(yt) + batch - 1) // batch))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(yt)); tot = 0.0
        for i in range(0, len(yt), batch):
            b = perm[i:i + batch]; xb = torch.stack([augment(xt[j:j + 1], rng)[0] for j in b.tolist()])
            loss = lossf(net(xb).squeeze(1), yt[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        if ep % 5 == 4 or ep == epochs - 1: log(f"  epoch {ep + 1}: loss {tot / len(yt):.3f}")
    return net.eval()


def score(net, X, batch=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch): out.append(torch.sigmoid(net(to_tensor(X[i:i + batch])).squeeze(1)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def top1_per_frame(meta, p, finder_w=0.0):
    """per frame, pick the guess with the highest (scorer + finder_w * finder score); right if it is a ball crop.
    Clicks added as positives when no finder guess was on the ball (finder score -1) are NOT pickable."""
    frames = {}
    for m, pi in zip(meta, p):
        frames.setdefault(m[0], []).append((pi + finder_w * max(0.0, float(m[5])), int(m[6]), float(m[5]) >= 0))
    right = has_ball = 0
    for rows in frames.values():
        if not any(l == 1 for _, l, _ in rows): continue
        has_ball += 1; pick = [r for r in rows if r[2]]
        if pick and max(pick, key=lambda r: r[0])[1] == 1: right += 1
    return right, has_ball
