"""Full match: pieces joined on the exact 29.97 fps timeline, players carried across the cuts, split export, held-out labels remapped."""
import pytest

def test_full_match_join_and_export(capsys):
    import sys, os, json, types, tempfile, numpy as np, cv2, glob, pickle
    from ipanema import fullmatch as FM
    from ipanema.config import Settings
    from ipanema.run import analyse
    fps = 29.97; n_total = int(round(90 * fps)); L, W = 120.0, 70.0
    root = tempfile.mkdtemp(); S = Settings(root=root, sports_dir="/tmp/nosports", work=f"{root}/work")
    Hm = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1.0]])          # pitch metres -> image px (x10)
    plan_ = FM.plan(n_total, fps, piece_s=30)
    print("plan:", [(p["i"], round(p["start_s"], 1), round(p["dur_s"], 2), p["offset"]) for p in plan_])
    # world truth: 10 players per team moving smoothly; the ball follows player A0, then A3
    def world(g):
        t = g / fps; rows = []
        for team, base in (("A", 30.0), ("B", 80.0)):
            for j in range(10):
                x = base + 10 * np.sin(0.05 * t + j) + (j - 5); y = 7 + 6 * j + 3 * np.cos(0.07 * t + j)
                rows.append((f"{team}{j}", team, np.array([x, y])))
        return rows
    pieces = []
    for pl in plan_:
        per, H, cands = {}, {}, {}
        n_local = int(round(pl["dur_s"] * fps))
        for k in range(n_local):
            g = pl["offset"] + k
            if g >= n_total: break
            rows = []
            for name, team, m in world(g):
                tid = int(name[1:]) + (0 if team == "A" else 10) + 1          # ids restart in every piece, like a fresh tracker
                feet = Hm @ np.array([m[0], m[1], 1.0]); feet = feet[:2] / feet[2]
                rows.append([tid, team, m.astype(float), feet.astype(float), np.array([feet[0] - 5, feet[1] - 20, feet[0] + 5, feet[1]]), False])
            per[k] = rows; H[k] = Hm.astype(np.float32)
            carrier = [r for r in rows if r[0] == (1 if g < n_total // 2 else 4)][0]
            cands[k] = [(carrier[3][0] + 3, carrier[3][1] - 2, 0.9), (500.0, 300.0, 0.3)]
        pieces.append({"per": per, "H": H, "cands": cands, "fps": fps, "n": n_local, "L": L, "W": W, "coverage": 1.0, "frozen": 0,
                       "dark_share": {"A": 0.5, "B": 0.2}, "strips": None, "width": 1200, "height": 700})
    per, H, cands, meta = FM.join(pieces, plan_, n_total, fps)
    print(f"joined: {len(per)} frames (expected {n_total}), H {len(H)}, cands {len(cands)}, empty frames {sum(1 for v in per.values() if not v)}")
    ids = sorted({r[0] for rows in per.values() for r in rows}); print(f"unique ids before cleanup: {len(ids)} (3 pieces x 20)")
    # a small video for the thumbnail
    vp = f"{root}/full.mp4"; vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    for g in range(n_total): vw.write(np.full((180, 320, 3), 60, np.uint8))
    vw.release()
    # held-out ball labels from a "segment" cut at 30 s, mapped into the full match
    os.makedirs(f"{root}/reference/TEST_s30"); seg_gt = {}
    for kl in (10, 200, 500):
        g = int(round(30 * fps)) + kl; c = [r for r in per[g] if r[0] % FM.ID_STRIDE == 1][0]; seg_gt[str(kl)] = [float(c[3][0] + 3), float(c[3][1] - 2)]
    json.dump(seg_gt, open(f"{root}/reference/TEST_s30/ball_gt.json", "w"))
    gt = {}
    for d in glob.glob(f"{root}/reference/TEST_s*/ball_gt.json"):
        start = int(d.split("_s")[-1].split("/")[0]); FM.remap_gt(d, start, fps, "/tmp/gt_t.json"); gt.update(json.load(open("/tmp/gt_t.json")))
    json.dump(gt, open("/tmp/gt_full_t.json", "w")); print("gt keys mapped:", sorted(gt, key=int))
    ctx = {"match_id": "TEST", "video": vp, "vi": {"n": n_total, "fps": fps, "width": 1200, "height": 700}, "H": H, "L": L, "W": W,
           "cal": {"coverage": 1.0, "frozen": 0}, "tm": types.SimpleNamespace(dark_share=meta["dark_share"], strips=None), "per": per, "fps": fps, "cands": cands, "t0": 0, "cands_alt": None, "picker_gt": ["/tmp/gt_full_t.json"]}
    logs = []
    summary, folder, z = analyse(ctx, S, log=lambda *a: logs.append(" ".join(map(str, a))), export_kw={"frame_stride": 3, "split_s": 30, "copy_video": False, "make_zip": False, "video_url": "https://r2/TEST/video.mp4"}, gt_path="/tmp/gt_full_t.json")
    print([l for l in logs if l.startswith("clean:") or "frame files" in l or "match_data" in l or l.startswith("ball check") or l.startswith("picker test")])
    md = json.load(open(f"{folder}/match_data.json")); lib = json.load(open(f"{root}/runs/library.json"))["matches"][-1]
    print("frame_chunks:", [(c["key"], c["t_start"], c["t_end"]) for c in md["frame_chunks"]])
    f1 = json.load(open(f"{folder}/frames_001.json"))["frames"]; fr = f1[0]
    g = int(round(fr["t"] * fps)); truth = {f"{t}{j}": m for (n_, t, m) in world(g) for j in [int(n_[1:])] if n_ == f"{t}{j}"}
    print(f"frames_001 first frame t={fr['t']} (global frame {g}); players {len(fr['players'])}; ball {fr['ball'] is not None}")
    print("video in library entry:", lib["files"]["video"], "| video_url:", lib.get("video_url"), "| frame file keys:", [k for k in lib["files"] if k.startswith("frames_")])
    ids_after = sorted({p["id"] for c in md["frame_chunks"] for p in [pp for f in json.load(open(f"{folder}/{c['key']}.json"))["frames"] for pp in f["players"]]})
    print(f"unique player ids after cleanup: {len(ids_after)} (ideal 20: every player carried across both cuts)")
    print("summary: duration", summary["duration_s"], "| ball_check", summary["ball_check"], "| players/frame", summary["players_per_frame_median"])
    assert len(per) == n_total and not [k for k in range(n_total) if k not in per]
    assert len(ids_after) == 20, ids_after
    assert [c["key"] for c in md["frame_chunks"]] == ["frames_000", "frames_001", "frames_002"]
    assert summary["ball_check"]["total"] == 3
    assert sorted(gt, key=int) == ["909", "1099", "1399"]
    assert summary["ball_reliable"] == summary["ball_grade"]["possession_ok"]


def test_full_match_with_periods(capsys):
    import sys, os, json, types, tempfile, numpy as np, cv2, glob, pickle
    from ipanema import fullmatch as FM
    from ipanema.config import Settings
    from ipanema.run import analyse
    fps = 29.97; n_total = int(round(90 * fps)); L, W = 120.0, 70.0
    root = tempfile.mkdtemp(); S = Settings(root=root, sports_dir="/tmp/nosports", work=f"{root}/work")
    Hm = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1.0]])          # pitch metres -> image px (x10)
    plan_ = FM.plan(n_total, fps, piece_s=30)
    print("plan:", [(p["i"], round(p["start_s"], 1), round(p["dur_s"], 2), p["offset"]) for p in plan_])
    # world truth: 10 players per team moving smoothly; the ball follows player A0, then A3
    def world(g):
        t = g / fps; rows = []
        for team, base in (("A", 30.0), ("B", 80.0)):
            for j in range(10):
                x = base + 10 * np.sin(0.05 * t + j) + (j - 5); y = 7 + 6 * j + 3 * np.cos(0.07 * t + j)
                rows.append((f"{team}{j}", team, np.array([x, y])))
        return rows
    pieces = []
    for pl in plan_:
        per, H, cands = {}, {}, {}
        n_local = int(round(pl["dur_s"] * fps))
        for k in range(n_local):
            g = pl["offset"] + k
            if g >= n_total: break
            rows = []
            for name, team, m in world(g):
                tid = int(name[1:]) + (0 if team == "A" else 10) + 1          # ids restart in every piece, like a fresh tracker
                feet = Hm @ np.array([m[0], m[1], 1.0]); feet = feet[:2] / feet[2]
                rows.append([tid, team, m.astype(float), feet.astype(float), np.array([feet[0] - 5, feet[1] - 20, feet[0] + 5, feet[1]]), False])
            per[k] = rows; H[k] = Hm.astype(np.float32)
            carrier = [r for r in rows if r[0] == (1 if g < n_total // 2 else 4)][0]
            cands[k] = [(carrier[3][0] + 3, carrier[3][1] - 2, 0.9), (500.0, 300.0, 0.3)]
        pieces.append({"per": per, "H": H, "cands": cands, "fps": fps, "n": n_local, "L": L, "W": W, "coverage": 1.0, "frozen": 0,
                       "dark_share": {"A": 0.5, "B": 0.2}, "strips": None, "width": 1200, "height": 700})
    per, H, cands, meta = FM.join(pieces, plan_, n_total, fps)
    per, H, cands, play_mask, periods = FM.apply_periods(per, H, cands, [(0, 50), (60, 90)], fps, L, W)
    print(f"joined: {len(per)} frames (expected {n_total}), H {len(H)}, cands {len(cands)}, empty frames {sum(1 for v in per.values() if not v)}")
    ids = sorted({r[0] for rows in per.values() for r in rows}); print(f"unique ids before cleanup: {len(ids)} (3 pieces x 20)")
    # a small video for the thumbnail
    vp = f"{root}/full.mp4"; vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    for g in range(n_total): vw.write(np.full((180, 320, 3), 60, np.uint8))
    vw.release()
    # held-out ball labels from a "segment" cut at 30 s, mapped into the full match
    os.makedirs(f"{root}/reference/TEST_s30"); seg_gt = {}
    for kl in (10, 200, 500):
        g = int(round(30 * fps)) + kl; c = [r for r in per[g] if r[0] % FM.ID_STRIDE == 1][0]; seg_gt[str(kl)] = [float(c[3][0] + 3), float(c[3][1] - 2)]
    json.dump(seg_gt, open(f"{root}/reference/TEST_s30/ball_gt.json", "w"))
    gt = {}
    for d in glob.glob(f"{root}/reference/TEST_s*/ball_gt.json"):
        start = int(d.split("_s")[-1].split("/")[0]); FM.remap_gt(d, start, fps, "/tmp/gt_t.json"); gt.update(json.load(open("/tmp/gt_t.json")))
    json.dump(gt, open("/tmp/gt_full_t.json", "w")); print("gt keys mapped:", sorted(gt, key=int))
    ctx = {"match_id": "TEST", "video": vp, "vi": {"n": n_total, "fps": fps, "width": 1200, "height": 700}, "H": H, "L": L, "W": W,
           "cal": {"coverage": 1.0, "frozen": 0}, "tm": types.SimpleNamespace(dark_share=meta["dark_share"], strips=None), "per": per, "fps": fps, "cands": cands, "t0": 0, "play_mask": play_mask, "periods": periods, "cands_alt": None, "picker_gt": ["/tmp/gt_full_t.json"]}
    logs = []
    summary, folder, z = analyse(ctx, S, log=lambda *a: logs.append(" ".join(map(str, a))), export_kw={"frame_stride": 3, "split_s": 30, "copy_video": False, "make_zip": False, "video_url": "https://r2/TEST/video.mp4"}, gt_path="/tmp/gt_full_t.json")
    print([l for l in logs if l.startswith("clean:") or "frame files" in l or "match_data" in l or l.startswith("ball check") or l.startswith("picker test")])
    md = json.load(open(f"{folder}/match_data.json")); lib = json.load(open(f"{root}/runs/library.json"))["matches"][-1]
    print("frame_chunks:", [(c["key"], c["t_start"], c["t_end"]) for c in md["frame_chunks"]])
    f1 = json.load(open(f"{folder}/frames_001.json"))["frames"]; fr = f1[0]
    g = int(round(fr["t"] * fps)); truth = {f"{t}{j}": m for (n_, t, m) in world(g) for j in [int(n_[1:])] if n_ == f"{t}{j}"}
    print(f"frames_001 first frame t={fr['t']} (global frame {g}); players {len(fr['players'])}; ball {fr['ball'] is not None}")
    print("video in library entry:", lib["files"]["video"], "| video_url:", lib.get("video_url"), "| frame file keys:", [k for k in lib["files"] if k.startswith("frames_")])
    ids_after = sorted({p["id"] for c in md["frame_chunks"] for p in [pp for f in json.load(open(f"{folder}/{c['key']}.json"))["frames"] for pp in f["players"]]})
    print(f"unique player ids after cleanup: {len(ids_after)} (ideal 20: every player carried across both cuts)")
    print("summary: duration", summary["duration_s"], "| ball_check", summary["ball_check"], "| players/frame", summary["players_per_frame_median"])
    assert len(md["periods"]) == 2 and md["periods"][1]["mirrored"] is True
    assert abs(summary["match_seconds"] - 80.0) < 0.2, summary["match_seconds"]
    assert summary["ball_reliable"] == summary["ball_grade"]["possession_ok"]
