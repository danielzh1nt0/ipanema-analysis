"""The watcher must stop failing runs and never stop healthy ones (simulated Modal)."""
import sys, types, time, tempfile, os, importlib, pytest

def test_watcher(monkeypatch):
    LOG = os.path.join(tempfile.mkdtemp(), "live.log")
    import sys, types, time
    # --- a stand-in for the modal library: exceptions, a call that runs for a few polls, a volume with log files ---
    modal = types.ModuleType("modal"); exc = types.ModuleType("modal.exception")
    class Error(Exception): pass
    class TimeoutError_(Error): pass
    class FunctionTimeoutError(TimeoutError_): pass
    class OutputExpiredError(TimeoutError_): pass
    exc.Error = Error; exc.TimeoutError = TimeoutError_; exc.FunctionTimeoutError = FunctionTimeoutError; exc.OutputExpiredError = OutputExpiredError
    modal.exception = exc; monkeypatch.setitem(sys.modules, "modal", modal); monkeypatch.setitem(sys.modules, "modal.exception", exc)
    import scripts_watch as W
    W.push = lambda msg: pushes.append(msg)
    class Call:
        def __init__(s, polls, result=None, raise_at_end=None): s.n = polls; s.result = result; s.raise_at_end = raise_at_end; s.cancelled = False
        def get(s, timeout=None):
            if s.n > 0: s.n -= 1; raise TimeoutError_("still running")
            if s.raise_at_end: raise s.raise_at_end
            return s.result
        def cancel(s, terminate_containers=False): s.cancelled = True
    class E:
        def __init__(s, path, mtime): s.path = path; s.mtime = mtime
    class Vol:
        def __init__(s, files): s.files = files        # path -> (mtime, text)
        def listdir(s, path, recursive=False): return [E(p, m) for p, (m, _) in s.files.items() if p.startswith(path)]
        def read_file(s, path): yield s.files[path][1].encode()
    now = time.time(); base = "match_analysis/logs/M/"
    def case(name, call, files, check_fallback=True):
        global pushes; pushes = []
        res, err = W.watch(call, Vol(files), "M", now, check_fallback, every=0, poll=0, write=True, log_path=LOG)
        first = open(LOG).readline().strip()
        return call.cancelled, err, first
    r = [case("healthy run finishes", Call(3, {"summary": {}, "log_tail": []}), {base + "run_full.log": (now + 1, "pieces: 21 to run\n"), base + "piece_000.log": (now + 2, "calibration: from panorama, 8992/8992 frames registered\n")}),
         case("piece crashes calibration -> stop", Call(9), {base + "piece_003.log": (now + 2, "mosaic calibration failed: ValueError('matmul ...')\n")}),
         case("fallback calibration although panorama exists -> stop", Call(9), {base + "piece_007.log": (now + 2, "calibration: 8992 frames, keypoints on 39%, 5325 carried\n")}),
         case("same fallback line, match WITHOUT panorama -> no stop", Call(2, {"summary": {}, "log_tail": []}), {base + "piece_007.log": (now + 2, "calibration: 8992 frames, keypoints on 39%\n")}, check_fallback=False),
         case("OLD failure log from a previous run -> ignored", Call(2, {"summary": {}, "log_tail": []}), {base + "piece_003.log": (now - 3600, "mosaic calibration failed: old run\n")}),
         case("run hits its time limit -> reported, not looped", Call(1, raise_at_end=FunctionTimeoutError("150 min")), {base + "run_full.log": (now + 1, "joining\n")})]
    names = ["healthy", "calibration crash", "fallback with panorama", "fallback without panorama", "old log", "time limit"]
    got = dict(zip(names, r))
    assert got["healthy"][0] is False and got["healthy"][1] is None
    assert got["calibration crash"][0] is True and "stopped automatically" in got["calibration crash"][1]
    assert got["fallback with panorama"][0] is True
    assert got["fallback without panorama"][0] is False and got["fallback without panorama"][1] is None
    assert got["old log"][0] is False and got["old log"][1] is None
    assert got["time limit"][0] is False and "time limit" in got["time limit"][1]

def test_watcher_budget(monkeypatch):
    import types as _t, sys as _s
    exc = _t.ModuleType("modal.exception")
    class Error(Exception): pass
    class TE(Error): pass
    class FTE(TE): pass
    class OEE(TE): pass
    exc.TimeoutError, exc.FunctionTimeoutError, exc.OutputExpiredError = TE, FTE, OEE
    mod = _t.ModuleType("modal"); mod.exception = exc
    monkeypatch.setitem(_s.modules, "modal", mod); monkeypatch.setitem(_s.modules, "modal.exception", exc)
    import scripts_watch as W; monkeypatch.setattr(W, "push", lambda msg: None)
    class Call:
        cancelled = False
        def get(self, timeout=None): raise TE("running")
        def cancel(self, terminate_containers=False): self.cancelled = True
    class Vol:
        def listdir(self, p, recursive=False): return []
    c = Call(); import time as _time
    res, err = W.watch(c, Vol(), "M", _time.time() - 3600, False, every=0, poll=0, write=False, budget_min=30)
    assert c.cancelled and "time budget" in err
