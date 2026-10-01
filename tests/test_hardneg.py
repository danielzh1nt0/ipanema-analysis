"""1 Oct: HARD_NEG block of kaggle/ballfinder_rfdetr.py copies training crops where the old finder is confidently wrong
(guess > 20 px from the labelled ball, or any guess on a ball-free crop) and leaves the rest alone. Stub model, no rfdetr."""
import os, re, sys, types, shutil, textwrap, numpy as np, cv2


class D:
    def __init__(self, boxes): self.xyxy = np.array([b[:4] for b in boxes]).reshape(-1, 4); self.confidence = np.array([b[4] for b in boxes])


def test_hardneg(tmp_path, monkeypatch):
    ds = tmp_path / "ds"; (ds / "train/images").mkdir(parents=True); (ds / "train/labels").mkdir(parents=True)
    for n, lab in (("a", "0 0.5 0.5 0.03 0.03\n"), ("b", "0 0.5 0.5 0.03 0.03\n"), ("c", "")):
        cv2.imwrite(str(ds / f"train/images/{n}.jpg"), np.zeros((640, 640, 3), np.uint8)); (ds / f"train/labels/{n}.txt").write_text(lab)
    preds = {0: D([(310, 310, 330, 330, 0.9)]),                    # a: guess on the ball -> not hard
             1: D([(310, 310, 330, 330, 0.9), (50, 50, 60, 60, 0.7)]),  # b: extra guess far away -> hard
             2: D([(100, 100, 110, 110, 0.5)])}                    # c: ball-free crop with a guess -> hard
    class M:
        def predict(self, imgs, threshold): return [preds[i] for i in range(len(imgs))]
    fake = types.ModuleType("ipanema.ballrf"); fake.load = lambda p: M(); fake.WEIGHTS = "w.pth"
    monkeypatch.setitem(sys.modules, "ipanema.ballrf", fake); import ipanema; monkeypatch.setattr(ipanema, "ballrf", fake, raising=False)
    s = open(os.path.join(os.path.dirname(__file__), "..", "kaggle", "ballfinder_rfdetr.py")).read()
    blk = re.search(r'    HARD_NEG = os.environ.*?save\(\)\n', s, re.S).group(0)
    monkeypatch.setenv("HARD_NEG", "1")
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    g = dict(os=os, cv2=cv2, np=np, shutil=shutil, time=__import__("time"), torch=torch, DS=str(ds), CROP=640, REPO=str(tmp_path), REPORT={},
             log=lambda m: None, save=lambda: None)
    exec(textwrap.dedent(blk), g)
    hn = g["REPORT"]["hard_neg"]
    assert hn["hard_crops"] == 2 and hn["hard_with_ball"] == 1 and hn["hard_empty"] == 1
    assert sorted(os.listdir(ds / "train/images")) == ["a.jpg", "b.jpg", "b_hn0.jpg", "b_hn1.jpg", "c.jpg", "c_hn0.jpg", "c_hn1.jpg"]
    assert (ds / "train/labels/c_hn0.txt").read_text() == ""
