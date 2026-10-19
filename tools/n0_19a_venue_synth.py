"""19 Oct (worker, N0): end-to-end check of ipanema/venue.solve_base on line masks drawn from a KNOWN camera (the SFK-BP
/ Edsberg solution): start from a base 2-3 m off, does the search + refine come back to the true base, and does the
held-out check accept it over the wrong one? Local CPU, ~10 min. Writes results/qa/n0-19a/venue_synth.json + strip."""
import json, os, sys, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import venue as V, lines as LN
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/results/qa/n0-19a"
s = json.load(open(f"{ROOT}/calibration/panorama/SFKBP1109_lines_solution.json")); cam = s["camera"]
fr = s["frames"][::max(1, len(s["frames"]) // 8)][:8]
masks = [LN.render_mask(cam, f["pose"], 640, 360) for f in fr]
wrong = {"C": list(np.array(cam["C"]) + [2.5, -2.0, 0.6]), "base_tilt": cam["base_tilt"]}
t0 = time.time(); new, poses, rep = V.solve_base(masks, wrong, log=print, coarse_frames=8)
ho = V.heldout_check(masks, new, wrong, log=print)
err = [LN.pose_error(cam, np.asarray(p), np.asarray(f["pose"]))["median_px"] if p is not None else None for p, f in zip(poses, fr)]
out = {"true_base": cam, "start_base": wrong, "solved": new, "report": rep, "heldout": ho, "minutes": round((time.time() - t0) / 60, 1),
       "base_error_m": round(float(np.linalg.norm(np.asarray(new["C"]) - np.asarray(cam["C"]))), 2),
       "start_error_m": round(float(np.linalg.norm(np.asarray(wrong["C"]) - np.asarray(cam["C"]))), 2),
       "pose_px_vs_truth": [None if e is None else round(float(e), 1) for e in err], "poses": poses}
json.dump(out, open(f"{OUT}/venue_synth.json", "w"), indent=1, default=float); print(json.dumps({k: out[k] for k in ("base_error_m", "start_error_m", "heldout", "pose_px_vs_truth", "minutes")}))
frames = [np.full((720, 1280, 3), 40, np.uint8) for _ in masks]
V.strip(frames, masks, new, poses, out_path=f"{OUT}/venue_synth_strip.jpg", n=8)
# truth (white) vs solved (colours) on the same tiles, to judge by eye
tiles = []
for f, p in zip(fr, poses):
    im = np.full((720, 1280, 3), 30, np.uint8); im = LN.draw_pose(im, cam, f["pose"], colour=(255, 255, 255), thick=5)
    if p is not None: im = LN.draw_pose(im, new, p, thick=2)
    tiles.append(cv2.resize(im, (640, 360)))
cv2.imwrite(f"{OUT}/venue_synth_truth_vs_solved.jpg", np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
