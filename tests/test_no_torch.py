"""N0-22a: the nightly test runner has no torch. ballrf.autocast() / _fp16() imported torch even when fp16 is off, so every
stand-in ball finder test (test_ballrf_edge.py, 8 of them) failed there. Now torch is imported only with IPANEMA_RF_FP16=1.
Also a guard: the whole stand-in ball finder test file passes in a Python where 'import torch' fails."""
import os, sys, contextlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ipanema import ballrf as BR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_autocast_and_fp16_need_no_torch_when_off(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)          # 'import torch' now raises ImportError
    monkeypatch.delenv("IPANEMA_RF_FP16", raising=False)
    assert isinstance(BR.autocast(), contextlib.nullcontext)
    assert BR._fp16() == {}
    monkeypatch.setenv("IPANEMA_RF_FP16", "0")
    with BR.autocast(): pass
    assert BR._fp16() == {}


def test_fp16_on_still_asks_torch(monkeypatch):
    """on = old behaviour: torch is imported (here it is missing, so it must fail rather than silently do nothing)"""
    monkeypatch.setitem(sys.modules, "torch", None); monkeypatch.setenv("IPANEMA_RF_FP16", "1")
    for f in (BR.autocast, BR._fp16):
        try: f(); assert False, "expected ImportError"
        except ImportError: pass


def test_detect_many_runs_without_torch(monkeypatch):
    class D:
        def __init__(s, b, c): s.xyxy = np.array(b, float).reshape(-1, 4); s.confidence = np.array(c, float)
    class Stub:
        def predict(s, crops, threshold): return [D([(5, 5, 15, 15)] if i % 8 == 3 else [], [0.9] if i % 8 == 3 else []) for i in range(len(crops))]
    monkeypatch.setitem(sys.modules, "torch", None); monkeypatch.delenv("IPANEMA_RF_FP16", raising=False)
    o = BR.detect_many(Stub(), [np.zeros((1080, 1920, 3), np.uint8)], edge=False)[0]
    assert len(o) == 1 and abs(o[0][0] - 1290) < 1e-6 and abs(o[0][1] - 10) < 1e-6


def test_ballrf_edge_file_passes_without_torch(tmp_path):
    """run test_ballrf_edge.py in a child Python whose 'import torch' fails, like the nightly runner"""
    (tmp_path / "sitecustomize.py").write_text("import sys\nsys.modules['torch'] = None\n")
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{ROOT}"); env.pop("IPANEMA_RF_FP16", None)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_ballrf_edge.py"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
