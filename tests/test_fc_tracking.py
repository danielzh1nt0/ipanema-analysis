"""Follow-cam calibration by fixed-base pan/tilt/zoom tracking: on real consecutive frames (the 22 Sep sample), the
tracked calibration must be judged correct on nearly every frame, including the window every earlier method got wrong."""
import os, json, glob, zipfile, tempfile, numpy as np, cv2, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = f"{ROOT}/results/samples/SFKBP1109_two_windows.zip"

@pytest.mark.skipif(not os.path.exists(ZIP), reason="frame sample not present")
@pytest.mark.parametrize("prefix", ["fc_1274", "fc_1372"])
def test_tracking_on_real_frames(prefix):
    from ipanema import ptz, fccam as FC
    from ipanema.calcheck import judge_frame
    d = tempfile.mkdtemp(); z = zipfile.ZipFile(ZIP)
    names = sorted(n for n in z.namelist() if n.startswith(prefix))[:45]          # first 3 s of the window
    for n in names: z.extract(n, d)
    C = json.load(open(f"{ROOT}/calibration/panorama/SFKBP1109_base.json"))["C"]; L, W = 106.0, 64.0
    fcs = [cv2.imread(f"{d}/{n}") for n in names]; h, w = fcs[0].shape[:2]
    H, info = FC.fit(fcs[0], C, L, W, tilt_range=(0.5, 30), coarse=(3.0, 1.5, 8))
    assert judge_frame(fcs[0], H, L, W)["verdict"] == "good"
    pose0 = np.array([np.radians(info["pan_deg"]), np.radians(info["tilt_deg"]), np.radians(info["roll_deg"]), info["zoom"]])
    poses, _ = ptz.track_sequence(fcs, C, pose0, w, h, refine_lines=lambda f, p: FC.refine(f, C, L, W, p))
    verd = [judge_frame(fc, ptz.homography(p, C, w, h), L, W)["verdict"] for fc, p in zip(fcs, poses)]
    assert verd.count("good") >= 0.9 * len(verd), verd
