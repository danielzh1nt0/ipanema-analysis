"""volume -> R2 copy (29 Sep): key mapping, unapproved matches never copied (also inside zips), same-size skip, dry run."""
import os, zipfile
from ipanema import r2copy as RC

class FakeS3:
    def __init__(self): self.store = {}; self.uploads = 0
    def head_object(self, Bucket, Key):
        if Key not in self.store: raise KeyError(Key)
        return {"ContentLength": self.store[Key]}
    def upload_file(self, src, bucket, key, ExtraArgs=None): self.store[key] = os.path.getsize(src); self.uploads += 1

def _root(tmp):
    def w(rel, data=b"x" * 10):
        p = tmp / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    for m in RC.APPROVED: w(f"labels/{m}_trainset_clicks.json", b"{}")
    w("labels/p15u-vs-reymersholm-2026-09-18_trainset_clicks.json", b"{}")
    w("labels/SFKBP1109_ball_auto.json", b"{}")
    w("models/wasb_finetuned.pth"); w("models/ball/clicks_latest.pt"); w("models/ball/best.json", b"{}")
    w("videos/SFKBP1109/full.mp4", b"v" * 100)
    for name, inner in (("SFKBP1109_ball_ds_auto0_x4.zip", "images/train/x_p15u_vs_spanga_1.jpg"), ("SFKBP1109_ball_ds_auto0_x6.zip", "images/train/x_p15u_vs_reymersholm_1.jpg")):
        (tmp / "labels").mkdir(exist_ok=True)
        with zipfile.ZipFile(tmp / "labels" / name, "w") as z: z.writestr(inner, b"img")
    return str(tmp)

def test_keys():
    assert RC.key_for("videos/SFKBP1109/full.mp4") == "SFKBP1109/video.mp4"
    assert RC.key_for("videos/SFKBP1109.mp4") == "SFKBP1109/video.mp4"
    assert RC.key_for("models/ball/clicks_latest.pt") == "models/ball/clicks_latest.pt"

def test_plan_blocks_unapproved(tmp_path):
    root = _root(tmp_path); rows = RC.plan(root, RC.DEFAULT_PATTERNS + ["labels/*reymersholm*.json"])
    keys = {r["key"] for r in rows if "key" in r}
    assert all("reymersholm" not in k and "solberga" not in k for k in keys)
    assert "labels/SFKBP1109_ball_ds_auto0_x4.zip" in keys and "labels/SFKBP1109_ball_ds_auto0_x6.zip" not in keys
    assert {"models/wasb_finetuned.pth", "models/ball/clicks_latest.pt", "models/ball/best.json", "SFKBP1109/video.mp4"} <= keys
    assert sum(1 for k in keys if k.endswith("_trainset_clicks.json")) == 4
    assert any(r.get("missing") for r in rows)                 # e.g. models/ball_finetuned.pt not in this fake root

def test_copy_dry_then_real_then_skip(tmp_path):
    root = _root(tmp_path); s3 = FakeS3()
    rows = RC.copy(root, RC.DEFAULT_PATTERNS, s3, "b", "https://pub.example", dry=True, log=lambda m: None)
    assert s3.uploads == 0 and all(r["status"] == "would upload" for r in rows if "key" in r)
    rows = RC.copy(root, RC.DEFAULT_PATTERNS, s3, "b", dry=False, log=lambda m: None)
    n = sum(1 for r in rows if "key" in r); assert s3.uploads == n and all(r["status"] == "uploaded" for r in rows if "key" in r)
    RC.copy(root, RC.DEFAULT_PATTERNS, s3, "b", log=lambda m: None); assert s3.uploads == n      # same size: not uploaded again
    assert RC.listing(root)["models/ball"] == {"best.json": 2, "clicks_latest.pt": 10}
