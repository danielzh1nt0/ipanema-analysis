"""Line model round 2 in one call (Colab): camera base check/fix from the painted near touchline -> labels under the
(possibly) fixed base -> train -> grade on the 9 held-back clicked frames, with the painted-pixel snap only if the base
fix was accepted (with the old base the snap was measured to pull AWAY from the clicked pose). Saves as it goes."""
import os, json, time, zipfile, shutil, numpy as np, cv2
from . import lines as LN, basefix as BF, linetrain as LT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(LAB, OUT, epochs=80, max_minutes=40, n_extra=12, log=print, work="/content", weights="imagenet"):
    t0 = time.time(); os.makedirs(f"{OUT}/base_check", exist_ok=True); os.makedirs(f"{OUT}/eval", exist_ok=True)
    sol = json.load(open(f"{ROOT}/calibration/panorama/SFKBP1109_clicks_solution.json"))
    clicks = {s: json.load(open(f"{ROOT}/results/labels/SFKBP1109_points_{s}.json")) for s in ("s1", "s2")}
    zips = {s: f"{LAB}/SFKBP1109_frames_{s}.zip" for s in ("s1", "s2")}; Z = {s: zipfile.ZipFile(p) for s, p in zips.items()}
    imgs = {}
    for i, f in enumerate(sol["frames"]):
        im = cv2.imdecode(np.frombuffer(Z[f["session"]].read(f["frame"]), np.uint8), cv2.IMREAD_COLOR)
        imgs[i] = cv2.resize(im, (1280, 720)) if im.shape[1] != 1280 else im
    # 1) base
    new, rep, ev = BF.fix_base(sol, clicks, imgs, log=log); ok = rep["accepted"]
    json.dump(rep, open(f"{OUT}/base_check/report.json", "w"), indent=1, default=float)
    json.dump(new, open(f"{OUT}/base_check/solution_used.json", "w"))
    for i in sorted(ev)[:10]:
        cv2.imwrite(f"{OUT}/base_check/near_{sol['frames'][i]['frame'][:-4]}.jpg", cv2.resize(BF.picture(imgs[i], sol, new, i, ev[i]), (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    log(f"step 1 (base) done in {(time.time() - t0) / 60:.1f} min: {'NEW base used' if ok else 'old base kept'}")
    # 2) labels
    pose_fn = (lambda p: BF.transfer_pose(sol["camera"], new["camera"], p)) if ok else None
    ds = f"{work}/lines_ds2"; shutil.rmtree(ds, ignore_errors=True)
    LN.build_dataset(new, zips, ds, random_json=f"{LAB}/SFKBP1109_random.json", random_review=f"{LAB}/SFKBP1109_random_review.json",
                     random_zip=f"{LAB}/SFKBP1109_random.zip", pose_fn=pose_fn, log=log)
    open(f"{OUT}/label_preview.jpg", "wb").write(LN.preview(ds, n=8))
    # 3) train
    w = LT.train(ds, f"{work}/lines_model2", epochs=epochs, max_minutes=max_minutes, log=log, weights=weights)
    shutil.copy(w, f"{OUT}/last.pt"); shutil.copy(f"{work}/lines_model2/progress.json", f"{OUT}/progress.json")
    # 4) grade (+ frames the old method got wrong, pictures only)
    prop = json.load(open(f"{LAB}/SFKBP1109_random.json")); revw = json.load(open(f"{LAB}/SFKBP1109_random_review.json"))
    no = sorted(n for n in prop if revw.get(n) == "no"); pick = no[::max(1, len(no) // n_extra)][:n_extra]; zr = zipfile.ZipFile(f"{LAB}/SFKBP1109_random.zip")
    extra = [(n[:-4], cv2.resize(cv2.imdecode(np.frombuffer(zr.read(n), np.uint8), 1), (640, 360), interpolation=cv2.INTER_AREA)) for n in pick]
    s = LT.evaluate(w, ds, f"{OUT}/eval", extra=extra, snap=ok, log=log)
    s["base_fix_accepted"] = ok; s["minutes"] = round((time.time() - t0) / 60, 1); json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    log(f"ALL DONE in {s['minutes']:.0f} min - tell Claude \"done\""); return s
