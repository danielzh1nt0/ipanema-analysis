import os, json, numpy as np, cv2, pytest
from ipanema import hfball as H


def _img(w=1280, h=720, ball=(600, 300, 12)):
    im = np.full((h, w, 3), (40, 120, 40), np.uint8); x, y, r = ball; cv2.circle(im, (x, y), r // 2, (255, 255, 255), -1); return im


def make_yolo(root):
    for sp in ("train", "valid"):
        os.makedirs(f"{root}/{sp}/images"); os.makedirs(f"{root}/{sp}/labels")
        for i in range(3):
            cv2.imwrite(f"{root}/{sp}/images/a{i}.jpg", _img())
            open(f"{root}/{sp}/labels/a{i}.txt", "w").write(f"0 {600/1280:.6f} {300/720:.6f} {12/1280:.6f} {12/720:.6f}\n")
    open(f"{root}/data.yaml", "w").write("nc: 1\nnames: ['ball']\n")


def make_coco(root):
    os.makedirs(f"{root}/train"); cv2.imwrite(f"{root}/train/a.jpg", _img())
    json.dump({"images": [{"id": 1, "file_name": "a.jpg", "width": 1280, "height": 720}],
               "categories": [{"id": 0, "name": "balls"}, {"id": 1, "name": "ball"}],
               "annotations": [{"image_id": 1, "category_id": 1, "bbox": [594, 294, 12, 12]}]}, open(f"{root}/train/_annotations.coco.json", "w"))


def make_parquet(root, xyxy=False):
    pd = pytest.importorskip("pandas"); pytest.importorskip("pyarrow")
    os.makedirs(f"{root}/data")
    ok, b = cv2.imencode(".jpg", _img())
    box = [594.0, 294.0, 606.0, 306.0] if xyxy else [594.0, 294.0, 12.0, 12.0]
    pd.DataFrame([{"image": {"bytes": b.tobytes(), "path": None}, "objects": {"bbox": [box], "category": [0]}}] * 4).to_parquet(f"{root}/data/train-0000.parquet")


@pytest.mark.parametrize("maker,kind", [(make_yolo, "yolo"), (make_coco, "coco"), (make_parquet, "parquet"), (lambda r: make_parquet(r, True), "parquet")])
def test_every_layout_gives_the_same_ball(tmp_path, maker, kind):
    root = str(tmp_path / "ds"); os.makedirs(root); maker(root)
    k, items, names = H.load(root)
    assert k == kind and items
    it = items[0]; im = H.image(it); b = H.balls(it, im, names)
    assert len(b) == 1
    x, y, w, h = b[0]
    assert abs(x + w / 2 - 600) < 1.5 and abs(y + h / 2 - 300) < 1.5 and abs(w - 12) < 1.5


def test_splits_and_ball_class_by_name(tmp_path):
    root = str(tmp_path / "y"); os.makedirs(root); make_yolo(root)
    k, items, _ = H.load(root)
    assert sorted({i["split"] for i in items}) == ["train", "valid"]
    assert H.ball_ids({0: "player", 1: "Ball"}) == {1} and H.ball_ids({0: "x"}) == {0} and H.ball_ids({0: "a", 1: "b"}) == set()


def test_stats_compare_with_veo():
    rows = [{"split": "train", "w": 1920, "h": 1080, "balls": [(0, 0, 16, 16)]}, {"split": "train", "w": 640, "h": 360, "balls": [(0, 0, 32, 32)]},
            {"split": "valid", "w": 640, "h": 360, "balls": []}]
    s = H.stats(rows)
    assert s["images"] == 3 and s["images_with_ball"] == 2 and s["by_split"] == {"train": 2, "valid": 1}
    assert s["share_as_small_as_veo"] == 0.5                                # 16/1920 is Veo-like, 32/640 = 5% of the width is not


def test_run_writes_sheets_and_veo_strip(tmp_path):
    root = str(tmp_path / "y"); os.makedirs(root); make_yolo(root)
    ex = tmp_path / "exam"; ex.mkdir(); cv2.imwrite(str(ex / "t1.jpg"), _img(960, 540, (300, 250, 6)))
    st = H.run(root, str(tmp_path / "out"), exam_dir=str(ex), exam_rows=[{"file": "gone.jpg", "truth": [1, 1]}, {"file": "t1.jpg", "truth": [600, 500]}, {"file": "t2.jpg", "truth": None}])
    assert st["images"] == 6 and st["balls"] == 6 and st["layout"] == "yolo"
    assert all(os.path.exists(f) for f in st["sheets"]) and os.path.exists(st["veo_reference"])
    assert cv2.imread(st["sheets"][0]).shape[1] == 2 * (480 + 270)
    assert cv2.imread(st["veo_reference"]).shape[1] == 270                     # the missing exam file is skipped, not counted
