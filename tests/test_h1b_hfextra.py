"""H1b (1 Oct): the HF_EXTRA block of kaggle/ballfinder_rfdetr.py cuts one 640 crop per outside picture with the ball inside
and its own box size. Runs the block's source on a fake YOLO set (no rfdetr, no GPU)."""
import os, re, random, textwrap, numpy as np, cv2


def test_hf_extra_crops(tmp_path, monkeypatch):
    root = tmp_path / "hf"; (root / "train/images").mkdir(parents=True); (root / "train/labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: ['ball']\n")
    for i, cx in enumerate((0.1, 0.5, 0.9)):
        cv2.imwrite(str(root / f"train/images/a{i}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
        (root / f"train/labels/a{i}.txt").write_text(f"0 {cx} 0.5 0.0064 0.0113\n")
    s = open(os.path.join(os.path.dirname(__file__), "..", "kaggle", "ballfinder_rfdetr.py")).read()
    pb = re.search(r'    def put_box\(.*?counts\[f"\{split\}_ball"\] \+= 1\n', s, re.S).group(0)
    hb = re.search(r'    HF_EXTRA = os.environ.*?log\(f"HF extra: \{REPORT\[\'hf\'\]\}"\)\n', s, re.S).group(0)
    ds = tmp_path / "ds"; (ds / "train/images").mkdir(parents=True); (ds / "train/labels").mkdir(parents=True)
    monkeypatch.setenv("HF_EXTRA", "1"); monkeypatch.setenv("HF_ROOT", str(root))
    g = dict(os=os, cv2=cv2, rng=random.Random(0), CROP=640, DS=str(ds), NEG_FRAC=0.0, SMOKE=False, TMP=str(tmp_path), REPORT={},
             counts={"train_ball": 0, "train_empty": 0, "valid_ball": 0, "valid_empty": 0}, log=lambda m: None, sh=lambda c: 0,
             put=lambda *a: None, ball=None)
    exec(textwrap.dedent(pb) + textwrap.dedent(hb), g)
    assert g["REPORT"]["hf"]["used"] == 3 and g["counts"]["train_ball"] == 3
    for f in os.listdir(ds / "train/labels"):
        c, x, y, w, h = map(float, open(ds / "train/labels" / f).read().split())
        assert 0 < x < 1 and 0 < y < 1 and abs(w * 640 - 12.3) < 0.5 and abs(h * 640 - 12.2) < 0.5
        assert cv2.imread(str(ds / "train/images" / (f[:-4] + ".jpg"))).shape == (640, 640, 3)


def test_default_off():
    s = open(os.path.join(os.path.dirname(__file__), "..", "kaggle", "ballfinder_rfdetr.py")).read()
    assert 'os.environ.get("HF_EXTRA") == "1"' in s and "_hf_TESTONLY" in s
