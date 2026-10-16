"""K5 tools/pseudo_ball.py: on a small made-up export, good balls become labels and each reject rule fires."""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "fake-match"


def _frame(t, ball_m, conf=0.9, state="observed", player_m=None):
    x, y = ball_m
    return {"t": round(t, 3),
            "ball": {"m": [x, y], "px": [100 + 10 * x, 50 + 10 * y], "conf": conf, "state": state},
            "players": [{"m": list(player_m or (x + 1.0, y))}]}


def _run(tmp, frames, keys=None):
    d = tmp / "results/volume/runs/matches" / M
    d.mkdir(parents=True)
    json.dump({"frames": frames}, open(d / "frames_000.json", "w"))
    json.dump({"fps": 25, "periods": [{"t_start": 0.0, "t_end": 100.0}],
               "pitch": {"length": 105, "width": 68}}, open(d / "match_data.json", "w"))
    if keys:
        (tmp / "reference" / M).mkdir(parents=True)
        json.dump({str(k): [0, 0] for k in keys}, open(tmp / "reference" / M / "ball_key_px.json", "w"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools/pseudo_ball.py"), M], cwd=tmp,
                       capture_output=True, text=True, check=True)
    return json.load(open(tmp / "results/ball/pseudo" / f"{M}.json")), r.stdout


def test_moving_ball_near_player_is_labelled(tmp_path):
    # ball rolls at 5 m/s along y=30 from t=5 s to t=15 s, 10 frames a second
    frames = [_frame(5 + i / 10, (20 + 0.5 * i, 30)) for i in range(101)]
    out, _ = _run(tmp_path, frames)
    assert 15 <= len(out) <= 21          # one label every >= 0.5 s over ~10 s
    ts = sorted(int(k) / 25 for k in out)
    assert all(b - a >= 0.5 - 1 / 25 - 1e-6 for a, b in zip(ts, ts[1:]))  # frame rounding
    # pixel position is the exported one
    k = min(out, key=int)
    t = int(k) / 25; i = round((t - 5) * 10)
    assert out[k] == [round(100 + 10 * (20 + 0.5 * i), 1), round(50 + 10 * 30, 1)]


def test_reject_rules(tmp_path):
    fr = []
    # 5-15 s: lying still ball (spare ball) -> no labels after the first 2.5 s
    fr += [_frame(5 + i / 10, (50, 30)) for i in range(101)]
    # 20-25 s: low confidence
    fr += [_frame(20 + i / 10, (20 + 0.5 * i, 30), conf=0.5) for i in range(51)]
    # 30-35 s: nobody near
    fr += [_frame(30 + i / 10, (20 + 0.5 * i, 30), player_m=(90, 60)) for i in range(51)]
    # 40-45 s: along the touchline
    fr += [_frame(40 + i / 10, (20 + 0.5 * i, 0.5)) for i in range(51)]
    # 50-55 s: bridged (not seen) ball
    fr += [_frame(50 + i / 10, (20 + 0.5 * i, 30), state="bridged") for i in range(51)]
    # 60-65 s: jumping ball (8 m a frame = 80 m/s)
    fr += [_frame(60 + i / 10, (20 + 8 * (i % 2), 30)) for i in range(51)]
    # 100-105 s: out of play (period ends at 100)
    fr += [_frame(100 + i / 10, (20 + 0.5 * i, 30)) for i in range(51)]
    out, log = _run(tmp_path, fr)
    ts = [int(k) / 25 for k in out]
    assert all(5 <= t <= 7.6 for t in ts), ts   # only the start of the still spell, before 2.5 s of stillness
    for reason in ["lying still", "ball not seen confidently", "nobody near", "near a line",
                   "too fast (jump)", "out of play"]:
        assert reason in log, reason


def test_by_eye_keys_are_held_out(tmp_path):
    frames = [_frame(5 + i / 10, (20 + 0.5 * i, 30)) for i in range(101)]
    key_frame = 10 * 25                     # a by-eye key at t = 10 s
    out, log = _run(tmp_path, frames, keys=[key_frame])
    assert all(abs(int(k) / 25 - 10) > 1.0 for k in out)
    assert "held out" in log
