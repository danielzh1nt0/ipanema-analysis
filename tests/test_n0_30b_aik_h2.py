import os, sys, json, subprocess
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n0_30b_aik_h2_cut as c


def test_times_are_second_half_after_kickoff_and_fixed():
    ts = c.sample_times()
    assert len(ts) == 90 and ts == sorted(ts)
    assert all(3995.0 <= t <= 6899.0 for t in ts)
    assert ts == c.sample_times()                     # same seed -> same frames (blind sample is reproducible)


def test_cut_on_a_local_stand_in_video(tmp_path):
    cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    # stand-in video written by OpenCV itself: the nightly runner has no ffmpeg binary, and calling a missing ffmpeg
    # raised FileNotFoundError instead of skipping (30 Oct card: 'tests failing (1)', worker N0-30c)
    v = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    for k in range(180):                                # 6 s at 30 fps, each frame a different grey so seeks are visible
        w.write(np.full((180, 320, 3), k, np.uint8))
    w.release()
    meta = c.cut(str(v), [1.0, 2.5, 99.0], str(tmp_path / "out"), cv2, log=lambda *a: None)
    assert [f["file"] for f in meta["frames"]] == ["q00.jpg", "q01.jpg"]  # 99 s is past the end -> skipped
    assert meta["frames"][1]["frame"] == 75
    assert json.load(open(tmp_path / "out" / "times.json"))["seed"] == 11
    assert (tmp_path / "out" / "q00.jpg").stat().st_size > 0
    assert abs(float(cv2.imread(str(tmp_path / "out" / "q01.jpg")).mean()) - 75) < 4   # the seek landed on frame 75


def test_make_view_stacks_full_frame_and_zoom():
    np = pytest.importorskip("numpy"); cv2 = pytest.importorskip("cv2")
    import n0_30b_aik_h2 as h
    img = np.zeros((1080, 1920, 3), np.uint8)
    v = h.make_view(img, [1900, 10], cv2)              # ball near a corner: the zoom window is clamped inside the frame
    assert v.shape[1] == 1280 and 720 + 600 < v.shape[0] < 720 + 660
    assert h.make_view(img, None, cv2).shape == v.shape


def test_committed_grades_and_result():
    import n0_30b_aik_h2 as h
    root = os.path.join(os.path.dirname(__file__), "..")
    gp = os.path.join(root, "results/qa/n0-30b/aik_h2_grades.json")
    if not os.path.exists(gp):
        pytest.skip("grades not committed")
    G = json.load(open(gp))
    T = {f["file"]: f["t"] for f in json.load(open(os.path.join(root, "results/free/n0_30b/h2/times.json")))["frames"]}
    assert len(G["items"]) == 90 and all(T[x["file"]] == x["t"] for x in G["items"])
    assert all(set(x) >= {"eye", "sure", "dead", "note"} and x["eye"] in ("A", "B", None) for x in G["items"])
    assert not any("ours" in x for x in G["items"])      # blind: no pipeline answer stored with the grades
    fr_path = os.path.join(root, "results/kaggle/full_match_52d795-join/frames_lite.json.gz")
    if not os.path.exists(fr_path):
        pytest.skip("export not in repo")
    cwd = os.getcwd(); os.chdir(root)
    try:
        res = h.compare(out_path=os.devnull)
    finally:
        os.chdir(cwd)
    assert res["all_decided"]["eye_decided"] == 48 and res["all_decided"]["eye_sfk"] == 16
    assert res["export_second_half_sfk_pct"] == 51.7
    assert res["pooled_both_halves"]["eye_decided"] == 75
    lo, hi = res["all_decided"]["eye_ci95"]
    assert hi < res["export_second_half_sfk_pct"]       # the export's second half is outside the by-eye range
