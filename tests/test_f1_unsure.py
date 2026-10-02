"""F1 (2 Oct): full matches keep a line-calibrated piece's unsure frames (as the 5-min clips do) instead of blanking them."""
import os, sys, numpy as np, pytest
from ipanema import fullmatch as FM

def test_trusted_frames_keeps_line_unsure_by_default(monkeypatch):
    monkeypatch.delenv("IPANEMA_FULL_UNSURE", raising=False)
    m = np.array([True, False, False, True])
    entries = [(0, 0, m, True), (1, 4, m, False), (2, 8, None, False)]
    t, missing = FM.trusted_frames(12, entries)
    assert t[:4].all()                                   # line piece: unsure frames kept
    assert t[4:8].tolist() == m.tolist()                 # other calibration: its check still applies
    assert missing == [2] and t[8:].all()
    monkeypatch.setenv("IPANEMA_FULL_UNSURE", "blank")
    t2, _ = FM.trusted_frames(12, entries)
    assert t2[:4].tolist() == m.tolist()                 # old behaviour on request

def test_join_collects_unsure_on_the_global_timeline():
    def piece(n, uns):
        return {"per": {k: [] for k in range(n)}, "H": {k: np.eye(3) for k in range(n)}, "cands": {k: [] for k in range(n)}, "coverage": 1.0, "frozen": 0,
                "L": 106.0, "W": 64.0, "width": 1920, "height": 1080, "dark_share": None, "strips": None, "unsure": uns}
    plan_ = [{"i": 0, "offset": 0}, {"i": 1, "offset": 10}]
    _, _, _, meta = FM.join([piece(10, [2, 3]), piece(10, [0, 9])], plan_, 18, 29.97)
    assert meta["unsure"] == {2, 3, 10}                  # 19 is past the end

@pytest.mark.skipif(not os.path.exists("results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl"), reason="needs the clip's app inputs")
def test_sfk_clip_inputs_blanking_costs_balls():
    sys.path.insert(0, "tools"); import f1_sim
    r = f1_sim.main(out="/tmp/f1_sim_test.json", with_stats=False)["variants"]
    keep, now = r["clip (nothing blanked)"], r["full match now (players + ball guesses blanked)"]
    assert r["clip (nothing blanked)"]["sfk34"] == 29 and r["clip (nothing blanked)"]["b4"] == 284   # reproduces the app clip run
    assert now["sfk34"] < keep["sfk34"] and now["b4"] < keep["b4"]
    assert {1383, 2536, 4150} <= set(now["missed"]) and not ({1383, 2536, 4150} & set(keep["missed"]))

def test_f1_check_dry_run(tmp_path):
    """the free-runner script runs end to end with a stand-in detector on a short synthetic video"""
    pytest.importorskip("supervision")
    import cv2, subprocess, json
    v = str(tmp_path / "v.mp4"); w = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 360))
    for _ in range(60):
        f = np.full((360, 640, 3), (40, 140, 50), np.uint8); f[:80] = 90
        for j, a in enumerate(np.linspace(0.1, 0.85, 10)): cv2.rectangle(f, (int(640 * a) + 2, 190), (int(640 * a) + 9, 200), (30, 30, 200) if j % 2 else (235, 235, 235), -1)
        w.write(f)
    w.release()
    env = dict(os.environ, DRY="1", LOCAL_VIDEO=v, F1_OUT=str(tmp_path / "out"))
    r = subprocess.run([sys.executable, "tools/f1_check.py"], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.load(open(tmp_path / "out" / "summary.json"))
    assert "counts" in s["A_aik_kits"] and (tmp_path / "out" / "aik_kits_match_vs_clip.jpg").exists()
