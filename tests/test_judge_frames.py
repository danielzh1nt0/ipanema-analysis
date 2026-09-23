"""The calibration check must agree with the eye on 28 real SFK-BP frames (labelled 21 Sep): every visibly wrong frame
rejected, every visibly right frame kept. The frames and their calibration are in results/frames/SFKBP1109."""
import os, json, cv2, numpy as np
from ipanema.calcheck import judge_frame
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIGHT, UNSURE = {3, 7, 27}, {4, 5, 23}

def test_judge_matches_visual_labels():
    meta = json.load(open(f"{ROOT}/results/frames/SFKBP1109/frames.json")); kept_wrong, rejected_right = [], []
    for n, r in enumerate(meta):
        img = cv2.imread(f"{ROOT}/results/frames/SFKBP1109/piece{r['i']:02d}_{r['k']:05d}.jpg")
        v = judge_frame(img, np.asarray(r["H_old"]), 120.0, 70.0)["verdict"]
        if n in RIGHT and v != "good": rejected_right.append(n)
        if n not in RIGHT and n not in UNSURE and v == "good": kept_wrong.append(n)
    assert not kept_wrong and not rejected_right, {"wrong frames passed": kept_wrong, "right frames rejected": rejected_right}

def test_judge_matches_labels_at_half_size():
    """tracking runs on half-size frames: there the judge must also pass every right frame and fail every wrong one"""
    meta = json.load(open(f"{ROOT}/results/frames/SFKBP1109/frames.json")); S = np.diag([0.5, 0.5, 1.0]); kept_wrong, rejected_right = [], []
    for n, r in enumerate(meta):
        if n in UNSURE: continue
        img = cv2.resize(cv2.imread(f"{ROOT}/results/frames/SFKBP1109/piece{r['i']:02d}_{r['k']:05d}.jpg"), (960, 540))
        v = judge_frame(img, S @ np.asarray(r["H_old"]), 120.0, 70.0)["verdict"]
        if n in RIGHT and v != "good": rejected_right.append(n)
        if n not in RIGHT and v == "good": kept_wrong.append(n)
    assert not kept_wrong and not rejected_right, {"wrong passed": kept_wrong, "right rejected": rejected_right}
