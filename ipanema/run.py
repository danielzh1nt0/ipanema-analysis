"""ipanema.run.run(video_path, match_id): the whole pipeline, cached per stage, one summary at the end."""
import os, json, pickle, time, numpy as np
from .config import Settings
from . import video as V, calibration as C, teams as T, tracking as TR, ball as BL, possession as P, analytics as AN, export as EX, metrics as M

def run(video_src, match_id=None, settings=None, log=print):
    S = settings or Settings(); t0 = time.time()
    try:
        from .segments import prepare_segments; prepare_segments(S.root, log=log)     # cut full games into segments before the first clip runs
    except Exception as e: log(f"segments: {e!r}")
    match_id = match_id or os.path.splitext(os.path.basename(video_src))[0]
    work = os.path.join(S.work, match_id); os.makedirs(work, exist_ok=True)
    cache = os.path.join(S.root, "cache", match_id); os.makedirs(cache, exist_ok=True)
    from . import __version__
    log(f"=== {match_id} === (ipanema {__version__}; ball: {os.path.basename(S.weights['ball'])}; pitch: {os.path.basename(S.weights['pitch'])})")
    video = V.normalise(video_src, os.path.join(work, "video.mp4")); vi = V.info(video)
    log(f"video: {vi['width']}x{vi['height']} @ {vi['fps']:.1f} fps, {vi['n']} frames ({vi['n']/vi['fps']:.0f} s)")
    from .mosaic import calibrate_via_mosaic
    Hm = None
    try: Hm = calibrate_via_mosaic(video, match_id, S.root, log=log)
    except Exception as e: log(f"mosaic calibration failed: {e!r}")
    if Hm:
        from .calibration import _pitch_config
        _, L, W = _pitch_config(S.sports_dir)
        valid = sorted(Hm)
        H = {k: (Hm[k] if k in Hm else Hm[min(valid, key=lambda v: abs(v - k))]) for k in range(vi["n"])}
        cal = {"coverage": len(Hm) / max(1, vi["n"]), "frozen": vi["n"] - len(Hm), "H": H, "L": L, "W": W}
        log(f"calibration: from panorama, {len(Hm)}/{vi['n']} frames registered")
    else:
        cal = C.calibrate(video, S.weights["pitch"], S.sports_dir, S.kp_conf, cache=f"{cache}/calibration.pkl", log=log)
    H, L, W = cal["H"], cal["L"], cal["W"]
    T.silence_progress(); tm = T.TeamModel(S.sports_dir).fit(video, S.weights["player"], S.conf_player, log=log)
    trk = f"{cache}/tracks.pkl"
    if os.path.exists(trk): per, fps = pickle.load(open(trk, "rb")); log("tracking: cached")
    else: per, fps = TR.track(video, S.weights["player"], H, tm, S.conf_player, log=log); pickle.dump((per, fps), open(trk, "wb"))
    per, cl = TR.clean(per, L, W, fps, log=log)
    cands = BL.candidates(video, S.weights["ball"], f"{cache}/ball_cands.pkl", S.conf_ball, tiles=S.ball_tiles, log=log)
    ball = BL.bridge(BL.pick(cands, H, L, W, log=log), fps)
    ball_check = BL.check(ball, cands, os.path.join(S.root, "reference", match_id, "ball_gt.json"), log=log)
    frames_, ballm = P.carriers(per, ball, H, S.carrier_r, S.near_r)
    state, bspeed = P.viterbi(per, ballm, fps, L, W)
    attack_right, conf = P.direction(state, ballm, log=log)
    import glob as _g
    ref = next(iter(_g.glob(os.path.join(S.root, "reference", f"events_gt_{match_id}.json")) + _g.glob(os.path.join(S.root, "reference", match_id, "events_gt*.json"))), None)
    ref_dir = json.load(open(ref)).get("attack_right_A") if ref else None
    if ref_dir is not None:
        attack_right = {"A": bool(ref_dir), "B": not ref_dir}; log(f"direction: from reference labels -> A {'→' if attack_right['A'] else '←'}")
    elif min(conf.values()) < 0.15:
        attack_right = P.direction_from_keepers(per, L, log=log) or P.direction_fallback(per, L, log=log)
    log("  step: sequences"); t_ = time.time()
    seqs = P.sequences(state, ballm, bspeed, fps, L, attack_right); rst = P.restarts(state, ballm, fps, L, W)
    log("  step: turnovers"); t_ = time.time()
    tvs = P.turnovers(per, frames_, state, ballm, fps, attack_right, S.press_r, S.near_r)
    log("  step: lanes"); t_ = time.time()
    ln = AN.lanes(per, frames_, attack_right, S.lane_half, S.max_lane)
    log("  step: passes"); t_ = time.time()
    ps, tracks = AN.passes(per, frames_, tvs, ln, attack_right, fps)
    log("  step: shapes"); t_ = time.time()
    sh = AN.shapes(per, L)
    log("  step: stats"); t_ = time.time()
    st = AN.stats(per, frames_, tvs, ps, tracks, state, fps, L, W, attack_right, S.press_r, S.near_r)
    log("  step: metrics"); t_ = time.time()
    mx = M.compute(state, ballm, bspeed, fps, L, W, attack_right, rst, ps, st['players'], tvs, sh, per=per, frames_=frames_); st['metrics'] = mx
    ctrl = int((state < 2).sum()); n = len(per)
    summary = {"match_id": match_id, "duration_s": round(n / fps, 1), "calibration_coverage": round(cal["coverage"], 2), "calibration_frozen": cal["frozen"],
               "team_dark_share": tm.dark_share, "players_per_frame_median": {t: float(np.median([sum(1 for r in per[k] if r[1] == t) for k in range(n)])) for t in ("A", "B")},
               "ball_frames_pct": round(100 * len(ball) / n), "ball_check": ball_check, "possession_pct": {t: round(100 * int((state == i).sum()) / max(1, ctrl)) for i, t in enumerate(("A", "B"))},
               "loose_pct": round(100 * int((state == 2).sum()) / n), "dead_pct": round(100 * int((state == 3).sum()) / n), "attack_right": attack_right, "direction_confidence": conf,
               "turnovers": len(tvs), "passes": len(ps), "restarts": len(rst), "sequences": len(seqs), "shots": {t: sum(1 for s in mx["shots"] if s["team"] == t) for t in ("A", "B")}, "goals": {t: sum(1 for s in mx["goals"] if s["team"] == t) for t in ("A", "B")}, "high_turnovers": mx["high_turnover_counts"], "field_tilt": {t: mx["field"][t]["field_tilt_pct"] for t in ("A", "B")}, "runtime_min": round((time.time() - t0) / 60, 1)}
    log("  step: export"); t_ = time.time()
    root, zpath = EX.write(os.path.join(S.root, "runs"), match_id, video, vi, per, frames_, ball, ballm, state, H, L, W, attack_right, conf, tvs, ps, rst, seqs, ln, sh, st, tm, summary, log=log)
    try:
        from .upload import upload_match; upload_match(S.root, match_id, log=log)
    except Exception as e: log(f"upload failed: {e!r}")
    log("\n--- SUMMARY ---"); [log(f"  {k:26s} {v}") for k, v in summary.items()]
    return summary, root, zpath
