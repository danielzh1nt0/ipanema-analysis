"""N0-21d (21 Oct): frames outside match time (before kick-off, half-time, after the whistle) and frames with an untrusted
camera have no players and no ball; possession_simple keeps the previous state there, so the team on the ball at the
whistle was given the whole break. Now those frames are nobody's possession (IPANEMA_PLAY_STATE=0 = old)."""
import os, json, types, tempfile
import numpy as np
import pytest
from ipanema import possession as P


def test_out_of_play_unit():
    st = np.array([0, 0, 1, 1, 2, 1, 0])
    assert (P.out_of_play(st, None) == st).all()
    play = np.array([1, 1, 1, 0, 0, 1, 1], bool)
    out = P.out_of_play(st, play)
    assert out.tolist() == [0, 0, 1, 2, 2, 1, 0]
    assert st.tolist() == [0, 0, 1, 1, 2, 1, 0]          # input not changed
    with pytest.raises(ValueError):
        P.out_of_play(st, play[:-1])


def _match(root, fps=10.0, dur_s=40.0, periods_s=((0, 15), (25, 40))):
    """two teams of 10; the ball is at A0's feet except 10-15 s where B0 has it (B on the ball at the half-time whistle)"""
    import cv2
    from ipanema import fullmatch as FM
    n = int(dur_s * fps); L, W = 120.0, 70.0
    Hm = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1.0]], np.float32)
    per, H, cands = {}, {}, {}
    for k in range(n):
        t = k / fps; rows = []
        for ti, (team, base) in enumerate((("A", 40.0), ("B", 80.0))):
            for j in range(10):
                m = np.array([base + 4 * np.sin(0.2 * t + j) + 2 * (j - 5), 5 + 6.5 * j])
                f = (Hm @ np.array([m[0], m[1], 1.0]))[:2]
                rows.append([1 + j + 10 * ti, team, m, f, np.array([f[0] - 5, f[1] - 20, f[0] + 5, f[1]]), False])
        per[k] = rows; H[k] = Hm
        c = rows[10] if 10 <= t < 15 else rows[0]
        cands[k] = [(float(c[3][0]) + 2, float(c[3][1]) - 1, 0.9)]
    per, H, cands, play, recs = FM.apply_periods(per, H, cands, list(periods_s), fps, L, W)
    vp = f"{root}/m.mp4"; vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    for _ in range(n): vw.write(np.full((90, 160, 3), 60, np.uint8))
    vw.release()
    return {"match_id": "PLAYTEST", "video": vp, "vi": {"n": n, "fps": fps, "width": 1200, "height": 700}, "H": H, "L": L, "W": W,
            "cal": {"coverage": 1.0, "frozen": 0}, "tm": types.SimpleNamespace(dark_share={"A": 0.5, "B": 0.2}, strips=None),
            "per": per, "fps": fps, "cands": cands, "t0": 0, "play_mask": play, "periods": recs, "cands_alt": None}


def _run(monkeypatch, on):
    from ipanema import run as R
    from ipanema.config import Settings
    monkeypatch.setattr(R, "_PLAY_STATE", on)
    monkeypatch.setattr(R, "_ball_gt", lambda *a, **k: None)
    root = tempfile.mkdtemp(); os.makedirs(f"{root}/cache/PLAYTEST", exist_ok=True)
    S = Settings(root=root, sports_dir="/tmp/nosports", work=f"{root}/work")
    logs = []
    summary, folder, _ = R.analyse(_match(root), S, log=lambda *a: logs.append(" ".join(map(str, a))),
                                   export_kw={"frame_stride": 1, "copy_video": False, "make_zip": False}, gt_path="/nonexistent")
    st = json.load(open(f"{folder}/stats.json")); md = json.load(open(f"{folder}/match_data.json"))
    frames = md.get("frames") or [f for c in md.get("frame_chunks", []) for f in json.load(open(f"{folder}/{c['key']}.json"))["frames"]]
    return summary, st, frames, logs


def test_break_is_nobodys_possession(monkeypatch):
    s, st, frames, logs = _run(monkeypatch, True)
    pos = {t["team"]: t["possession_pct"] for t in st["teams"]}
    # match time: A has the ball 25 s (0-10, 25-40), B 5 s (10-15) -> B about 17%, never more than 25%
    assert pos["B"] <= 25 and pos["A"] >= 75, pos
    assert s["possession_pct"]["B"] <= 25, s["possession_pct"]
    gap = [f for f in frames if 16.0 <= f["t"] <= 24.0]
    assert gap and all(f["possession"] is None for f in gap)
    assert any("set to nobody's" in l for l in logs)
    assert s["loose_pct"] <= 15, s["loose_pct"]                # of match time, not of the whole recording


def test_old_behaviour_switch(monkeypatch):
    s, st, frames, logs = _run(monkeypatch, False)
    pos = {t["team"]: t["possession_pct"] for t in st["teams"]}
    # the old rule: B keeps the ball through the 10 s break -> B about 15 of 40 s
    assert pos["B"] >= 30, pos
    gap = [f for f in frames if 16.0 <= f["t"] <= 24.0]
    assert gap and all(f["possession"] == "B" for f in gap)
