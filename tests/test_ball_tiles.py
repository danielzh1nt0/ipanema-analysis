"""Tiled ball detection: coordinates must map back to the full frame and training targets must sit on the ball.
Uses a stand-in network that 'detects' the brightest spot, so the geometry is tested without the model."""
import sys, types, contextlib, numpy as np, cv2, tempfile, os, importlib

def _fake_torch():
    class T:
        def __init__(s, a): s.a = np.asarray(a)
        def to(s, d): return s
        def float(s): return s
        def cpu(s): return s
        def numpy(s): return s.a
    t = types.ModuleType("torch"); t.from_numpy = lambda a: T(a); t.sigmoid = lambda x: T(1 / (1 + np.exp(-x.a)))
    t.cuda = types.SimpleNamespace(is_available=lambda: False); t.no_grad = contextlib.nullcontext; t.T = T
    return t

def test_tiled_geometry(monkeypatch):
    torch = _fake_torch(); monkeypatch.setitem(sys.modules, "torch", torch)
    import ipanema.wasb as W, ipanema.wasb_train as WT
    def net(x):
        a = x.a; out = np.full((a.shape[0], 3, 288, 512), -10.0, np.float32); yy, xx = np.mgrid[0:288, 0:512]
        for b in range(a.shape[0]):
            for j in range(3):
                v = (a[b, 3 * j:3 * j + 3] * W.STD[:, None, None] + W.MEAN[:, None, None]).mean(0)
                if v.max() > 0.85:
                    py, px = np.unravel_index(np.argmax(cv2.GaussianBlur(v, (5, 5), 0)), v.shape); out[b, j][(xx - px) ** 2 + (yy - py) ** 2 <= 4] = 10.0
        return {0: torch.T(out)}
    class Net:
        def __call__(self, x): return net(x)
        def eval(self): pass
    monkeypatch.setattr(W, "ensure", lambda root, log=print: None); monkeypatch.setattr(W, "_model", lambda root, dev, finetuned=True: net)
    monkeypatch.setattr(WT, "ensure_finetuned", lambda root, vd, log=print: False)
    d = tempfile.mkdtemp(); path = f"{d}/v.mp4"; truth = {}
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    pts = [(40, 40), (960, 540), (1000, 560), (1880, 1040), (100, 1000), (1900, 60), (700, 300), (1300, 800), (962, 300), (500, 545)]
    for k in range(60):
        f = np.full((1080, 1920, 3), (40, 120, 50), np.uint8); x, y = pts[k % len(pts)]; x += (k // len(pts)) * 3; truth[k] = (x, y)
        cv2.circle(f, (x, y), 6, (255, 255, 255), -1); vw.write(f)
    vw.release(); os.makedirs(f"{d}/cache")
    out = W.candidates(path, d, f"{d}/cache/c.pkl", log=lambda *a: None)
    errs = [np.hypot(out[k][0][0] - x, out[k][0][1] - y) for k, (x, y) in truth.items() if out.get(k)]
    assert len(errs) == 60 and max(errs) < 4.0, (len(errs), max(errs) if errs else None)
    cap = cv2.VideoCapture(path); fr = [W.to_base(cap.read()[1]) for _ in range(3)]; cap.release()
    s = {"frames": fr, "ball": (truth[1][0] / 2, truth[1][1] / 2)}; rng = np.random.RandomState(0)
    for _ in range(100):
        x, t = WT.random_crop(s, rng)
        if t.sum() > 0:
            v = (x[3:6] * W.STD[:, None, None] + W.MEAN[:, None, None]).mean(0); py, px = np.unravel_index(np.argmax(cv2.GaussianBlur(v, (5, 5), 0)), v.shape)
            ty, tx = np.argwhere(t > 0).mean(0); assert np.hypot(px - tx, py - ty) < 1.5
    assert WT.hit_rate(Net(), "cpu", [s, {"frames": fr, "ball": None}, {"frames": fr, "ball": (50.0, 50.0)}], [0, 1, 2]) == (1, 2)
