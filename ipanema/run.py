"""ipanema.run.run(video_path, match_id): the whole pipeline, cached per stage, one summary at the end."""
import os, json, pickle, time, numpy as np
from .config import Settings
from . import video as V, calibration as C, teams as T, tracking as TR, ball as BL, possession as P, analytics as AN, export as EX, metrics as M

def prepare(video_src, match_id, S, log=print, train_ball=True, debug=True):
    """per clip: calibration, teams, tracking and ball candidates (everything that needs the video / a GPU)"""
    t0 = time.time()
    work = os.path.join(S.work, match_id); os.makedirs(work, exist_ok=True)
    cache = os.path.join(S.root, "cache", match_id); os.makedirs(cache, exist_ok=True)
    from . import __version__
    log(f"=== {match_id} === (ipanema {__version__}; ball: {os.path.basename(S.weights['ball'])}; pitch: {os.path.basename(S.weights['pitch'])})")
    video = V.normalise(video_src, os.path.join(work, "video.mp4")); vi = V.info(video)
    log(f"video: {vi['width']}x{vi['height']} @ {vi['fps']:.1f} fps, {vi['n']} frames ({vi['n']/vi['fps']:.0f} s)")
    from .mosaic import calibrate_via_mosaic
    Hm = None
    # Veo panorama clips (<match>_pano...): one curved-camera calibration for every frame (no per-frame registration)
    import re as _re, json as _json
    # looked up by the exact clip name (e.g. an app upload) or its "<name>_pano" prefix; a follow-cam match never matches one
    _mid = _re.sub(r"_c\d+$", "", match_id); _dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "panorama")
    _cands = [f"{_dir}/{_mid}.json"] + ([f"{_dir}/{_re.match(r'^(.*?_pano)', _mid).group(1)}.json"] if "_pano" in _mid else [])
    _spec = next((c for c in _cands if os.path.exists(c)), None)
    panorama = _spec is not None
    from .mosaic import CAL_VERSION as _CV
    pano_cache = f"{cache}/calibration_pano_{_CV}.pkl"
    try:
        if panorama: raise StopIteration
        if os.path.exists(pano_cache): Hm = pickle.load(open(pano_cache, "rb")); log(f"calibration: panorama registration cached ({len(Hm)} frames)")
        else:
            Hm = calibrate_via_mosaic(video, match_id, S.root, log=log)
            if Hm: pickle.dump(Hm, open(pano_cache, "wb"))
    except StopIteration: pass
    except Exception as e: log(f"mosaic calibration failed: {e!r}")
    if panorama:
        from .cylcam import CylCam, fit as _cylfit, NAMES as _CN
        spec = _json.load(open(_spec)); L, W = spec["pitch"]["length"], spec["pitch"]["width"]
        clip_cache = os.path.join(S.root, "cache", _mid); os.makedirs(clip_cache, exist_ok=True)
        cam_cache = f"{clip_cache}/calibration_cyl.json"                    # one camera for the whole clip: every piece uses it
        if not os.path.exists(cam_cache) and os.path.exists(f"{cache}/calibration_cyl.json"):
            import shutil as _sh; _sh.copy(f"{cache}/calibration_cyl.json", cam_cache)
        if os.path.exists(cam_cache):
            cam = CylCam(_json.load(open(cam_cache))["params"]); log("calibration: curved panorama camera (fitted once for this clip)")
        else:
            import cv2 as _cv
            from .cylcam import plausible as _plaus
            cap = _cv.VideoCapture(video); iw, ih = spec["image_size"]; init0 = [spec["params"][k] for k in _CN]
            tries = []
            for frac in (0.15, 0.35, 0.5, 0.7, 0.9):                        # one frame can fit badly: try five across the clip, keep the best possible one
                cap.set(_cv.CAP_PROP_POS_FRAMES, int(vi["n"] * frac)); okf, fr = cap.read()
                if not okf: continue
                hh, ww = fr.shape[:2]; init = list(init0); s = ww / iw
                for j in (4, 5, 6, 7): init[j] *= s
                cam_i, stats_i = _cylfit(fr, init, L, W, mask_top=int(spec["mask_rows"]["top"] / ih * hh), mask_bottom=int(spec["mask_rows"]["bottom_from"] / ih * hh), log=lambda *a: None)
                good, why = _plaus(cam_i.params, L, W)
                log(f"  calibration try at {frac:.0%} of the clip: {stats_i} {'ok' if good else 'rejected: ' + ', '.join(why)}")
                if good: tries.append((stats_i["median_px"], cam_i, stats_i))
            cap.release()
            if not tries: raise RuntimeError("panorama calibration: no physically possible camera fitted on any of five frames")
            _, cam, stats = min(tries, key=lambda z: z[0]); log(f"calibration: kept the best of {len(tries)} possible fits ({stats})")
            _json.dump({"params": cam.params.tolist(), "fit": stats}, open(cam_cache, "w"))
        H = {k: cam for k in range(vi["n"])}
        cal = {"coverage": 1.0, "frozen": 0, "H": H, "L": L, "W": W}
        log(f"calibration: curved panorama camera for all {vi['n']} frames, pitch {L:.0f} x {W:.0f} m")
    elif Hm:
        from .calibration import _pitch_config
        _, L, W = _pitch_config(S.sports_dir)
        valid = sorted(Hm)
        H = {k: (Hm[k] if k in Hm else Hm[min(valid, key=lambda v: abs(v - k))]) for k in range(vi["n"])}
        cal = {"coverage": len(Hm) / max(1, vi["n"]), "frozen": vi["n"] - len(Hm), "H": H, "L": L, "W": W}
        log(f"calibration: from panorama, {len(Hm)}/{vi['n']} frames registered")
    else:
        cal = C.calibrate(video, S.weights["pitch"], S.sports_dir, S.kp_conf, cache=f"{cache}/calibration.pkl", log=log)
    H, L, W = cal["H"], cal["L"], cal["W"]
    # debug overlays: the pitch model drawn on a few frames, published for inspection
    try:
        if not debug: raise StopIteration
        dbg = "/content/ipanema-analysis/results/debug"; os.makedirs(dbg, exist_ok=True)
        from .video import frame_at
        import cv2 as _cv
        for k in [int(vi["n"] * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]:
            fr = frame_at(video, k)
            if fr is not None: C.draw_model(fr, H[k], L, W); _cv.imwrite(f"{dbg}/{match_id}_v{__version__}_f{k:06d}.jpg", _cv.resize(fr, (1280, 720)), [_cv.IMWRITE_JPEG_QUALITY, 80])
        log(f"  debug overlays -> {dbg}")
    except StopIteration: pass
    except Exception as e: log(f"  debug overlays failed: {e!r}")
    T.silence_progress(); tm = T.TeamModel(S.sports_dir).fit(video, S.weights["player"], S.conf_player, log=log)
    from .mosaic import CAL_VERSION
    _trk_name = os.environ.get('IPANEMA_TRACKER', 'bytetrack')
    trk = f"{cache}/tracks_cyl.pkl" if panorama else f"{cache}/tracks_{('pano_' + CAL_VERSION) if Hm else 'kp'}" + ("" if _trk_name == "bytetrack" else f"_{_trk_name}") + ".pkl"   # bytetrack keeps the original cache name        # positions in metres depend on the calibration: cache per calibration version
    alt_trk = trk.replace(f"tracks_pano_{CAL_VERSION}", "tracks_kp") if (Hm and not panorama) else None
    if os.path.exists(trk): per, fps = pickle.load(open(trk, "rb")); log("tracking: cached")
    elif alt_trk and alt_trk != trk and os.path.exists(alt_trk):
        # detections are made in the picture; only metres depend on calibration -> re-position, don't re-detect
        per, fps = pickle.load(open(alt_trk, "rb")); per = TR.reposition(per, H); pickle.dump((per, fps), open(trk, "wb"))
        log("tracking: reused detections, re-positioned with the panorama calibration")
    else: per, fps = TR.track(video, S.weights["player"], H, tm, 0.1 if panorama else S.conf_player, log=log, imgsz=2560 if panorama else None)   # panorama: whole frame at full resolution (measured 12:29: 23 players/frame at 7.4 frames/s; 6 tiles found 24 at 1.8 frames/s); pickle.dump((per, fps), open(trk, "wb"))
    ball_backend = os.environ.get("IPANEMA_BALL", "wasb")
    cands = None
    if ball_backend == "wasb":
        try:
            from . import wasb
            cands = wasb.candidates(video, S.root, f"{cache}/ball_cands_wasb.pkl", log=log, train=train_ball)
            # union with the single-frame detector (off by default: measured ceiling gain is small, junk candidates cost picks)
            if os.environ.get("IPANEMA_BALL_MERGE", "0") != "1": raise StopIteration("wasb-only")
            try:
                yolo = BL.candidates(video, S.weights["ball"], f"{cache}/ball_cands.pkl", S.conf_ball, tiles=S.ball_tiles, log=log)
                merged = {}
                for k in set(cands) | set(yolo):
                    ws = [(x, y, min(0.99, 0.5 + 0.5 * c)) for x, y, c in cands.get(k, [])]        # WASB peaks: conf 0.5-1.0
                    ys = [(x, y, c * 0.6) for x, y, c in yolo.get(k, []) if all(np.hypot(x - wx, y - wy) > 12 for wx, wy, _ in ws)]
                    merged[k] = ws + ys
                cands = merged; log(f"ball: WASB + YOLO candidates merged, {sum(len(v) for v in cands.values())/max(1,len(cands)):.1f}/frame")
            except StopIteration: raise
            except Exception as e: log(f"ball: YOLO merge skipped ({e!r})")
        except StopIteration: log(f"ball: WASB-only candidates, {sum(len(v) for v in cands.values())/max(1,len(cands)):.1f}/frame")
        except Exception as e: log(f"wasb failed ({e!r}); falling back to YOLO ball"); cands = None
    if cands is None:
        cands = BL.candidates(video, S.weights["ball"], f"{cache}/ball_cands.pkl", S.conf_ball, tiles=S.ball_tiles, log=log)
        try:
            from . import ballcls
            if os.path.exists(ballcls.weights_path(S.root)): cands = ballcls.rescore(video, cands, S.root, cache=f"{cache}/ball_cands_cls.pkl", log=log)
        except Exception as e: log(f"ball classifier: {e!r}")
    cands_alt = None      # WASB + cached YOLO, scored side by side in analyse() when available (costs no GPU)
    try:
        yc = f"{cache}/ball_cands.pkl"
        if os.environ.get("IPANEMA_BALL_MERGE", "0") != "1" and os.path.exists(yc) and cands:
            yolo = pickle.load(open(yc, "rb")); cands_alt = {}
            for k in set(cands) | set(yolo):
                ws = [(x, y, min(0.99, 0.5 + 0.5 * c)) for x, y, c in cands.get(k, [])]
                ys = [(x, y, c * 0.6) for x, y, c in yolo.get(k, []) if all(np.hypot(x - wx, y - wy) > 12 for wx, wy, _ in ws)]
                cands_alt[k] = ws + ys
    except Exception as e: log(f"alt candidates skipped: {e!r}")
    return {"match_id": match_id, "video": video, "vi": vi, "H": H, "L": L, "W": W, "cal": cal, "tm": tm, "per": per, "fps": fps, "cands": cands, "t0": t0, "cands_alt": cands_alt}


def analyse(ctx, S, log=print, export_kw=None, gt_path=None):
    """from raw tracks + ball candidates to stats, events and the exported match (CPU)"""
    match_id, video, vi, H, L, W, cal, tm, per, fps, cands, t0 = (ctx[k] for k in ("match_id", "video", "vi", "H", "L", "W", "cal", "tm", "per", "fps", "cands", "t0"))
    per, cl = TR.clean(per, L, W, fps, log=log)
    def _v1():
        g = BL.pick_global(cands, H, L, W, per=per, fps=fps, log=log)
        if len(g) < 0.2 * len(cands): log("ball: global path too sparse, falling back to trajectory picker"); g = BL.pick(cands, H, L, W, per=per, log=log)
        return BL.bridge(g, fps)
    def _v2(): return BL.bridge(BL.pick_v2(cands, H, L, W, per=per, fps=fps, log=log), fps)
    picker = os.environ.get("IPANEMA_PICKER", "v2")
    ball = _v2() if picker == "v2" else _v1()
    other = (_v1 if picker == "v2" else _v2) if cands else None
    ball_check = BL.check(ball, cands, gt_path or os.path.join(S.root, "reference", match_id, "ball_gt.json"), log=log, video=video, debug_dir=f"/content/ipanema-analysis/results/debug/ballcheck_{match_id}")
    # the other picker on the same frames, logged side by side (CPU only)
    try:
        if other is not None:
            o = BL.check(other(), cands, gt_path or os.path.join(S.root, "reference", match_id, "ball_gt.json"), log=lambda *a: None)
            if o: log(f"ball check (other picker, {'v1' if picker == 'v2' else 'v2'}): {o['correct']}/{o['total']} correct, ceiling {o['ceiling']}/{o['total']}")
        for pg in ctx.get("picker_gt") or []:
            a = BL.check(ball, cands, pg, log=lambda *a: None); b = BL.check(other(), cands, pg, log=lambda *a: None) if other else None
            if a: log(f"picker test on detector-training clicks ({os.path.basename(os.path.dirname(pg))}, {a['total']} frames; detection is optimistic here, the pick is the fair part): {picker} {a['correct']}/{a['total']}" + (f", other {b['correct']}/{b['total']}" if b else "") + f", ceiling {a['ceiling']}")
    except Exception as e: log(f"picker comparison failed: {e!r}")
    if ctx.get("cands_alt"):
        try:
            alt = BL.pick_global(ctx["cands_alt"], H, L, W, per=per, fps=fps, log=lambda *a: None)
            alt_check = BL.check(BL.bridge(alt, fps), ctx["cands_alt"], gt_path or os.path.join(S.root, "reference", match_id, "ball_gt.json"), log=lambda *a: None)
            if alt_check: log(f"ball check (alternative: WASB + YOLO candidates): {alt_check['correct']}/{alt_check['total']} correct, ceiling {alt_check['ceiling']}/{alt_check['total']}")
        except Exception as e: log(f"alternative ball check failed: {e!r}")
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
    # Veo's own shots/goals (to the second), when we have them, replace our shot detector; ours keeps being scored against them
    try:
        from . import veo as VEO
        vp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", f"veo_highlights_{match_id}.txt")
        if os.path.exists(vp):
            vs, rej = VEO.build(VEO.load(vp), fps, ballm, per, L, W, attack_right, rst, periods=ctx.get("periods"), log=log)
            sc = VEO.score_detector(mx["shots"], vs)
            log(f"our shot detector vs Veo: found {sc['found']}/{sc['veo_shots']} of Veo's shots, {sc['real']}/{sc['ours']} of ours are real (within 10 s)")
            for g in rej: log(f"veo: rejected goal tag at {g['t']} s: {g['why']}")
            mx["shots_detected"], mx["shots"], mx["goals"], mx["veo_rejected"] = mx["shots"], vs, [s for s in vs if s["goal"]], rej
    except Exception as e: log(f"veo import failed: {e!r}")
    ctrl = int((state < 2).sum()); n = len(per)
    play = ctx.get("play_mask"); n_play = int(play.sum()) if play is not None else n; play_ks = [k for k in range(n) if play is None or play[k]]
    # ball reliability: a real ball is near a player most of the time and does not teleport
    near = [k for k in ballm if per[k] and min(np.linalg.norm(r[2] - ballm[k]) for r in per[k]) < 4.0]
    jumps = sum(1 for k in ballm if k - 1 in ballm and np.linalg.norm(ballm[k] - ballm[k - 1]) * fps > 35)
    ball_reliable = bool(len(ballm) > 0.4 * n_play and len(near) > 0.6 * max(1, len(ballm)) and jumps < 0.02 * max(1, len(ballm)))
    # graded reliability: different stats need different ball accuracy
    acc = (ball_check["correct"] / ball_check["total"]) if (ball_check and ball_check.get("total", 0) >= 10) else None
    if acc is not None: ball_reliable = acc >= 0.6
    near_frac = len(near) / max(1, len(ballm))
    ball_grade = {
        "accuracy": round(acc, 2) if acc is not None else None,
        "near_player_pct": round(near_frac, 2),
        "frames_pct": round(len(ballm) / max(1, n_play), 2),
        # possession and territory tolerate a loose ball: the nearest player is usually still right
        "possession_ok": bool((acc is None or acc >= 0.45) and near_frac >= 0.85 and len(ballm) > 0.4 * n_play),
        # events need the ball in the right place at the right moment
        "events_ok": bool(acc is not None and acc >= 0.6 and near_frac >= 0.85),
    }
    log(f"ball grade: accuracy {ball_grade['accuracy']}, near-player {ball_grade['near_player_pct']}, possession {'OK' if ball_grade['possession_ok'] else 'withheld'}, events {'OK' if ball_grade['events_ok'] else 'withheld'}")
    ball_reliable = bool(ball_grade["possession_ok"])      # one verdict everywhere: the old flag follows the stricter grade
    log(f"ball reliability: {len(ballm)/n:.0%} frames, {len(near)/max(1,len(ballm)):.0%} near a player, {jumps} jumps -> {'OK' if ball_reliable else 'UNRELIABLE (stats withheld in app)'}")
    summary = {"match_id": match_id, "ball_reliable": ball_reliable, "ball_grade": ball_grade, "duration_s": round(n / fps, 1), "calibration_coverage": round(cal["coverage"], 2), "calibration_frozen": cal["frozen"],
               "team_dark_share": tm.dark_share, "players_per_frame_median": {t: float(np.median([sum(1 for r in per[k] if r[1] == t) for k in play_ks] or [0])) for t in ("A", "B")},
               "ball_frames_pct": round(100 * len(ball) / max(1, n_play)), "match_seconds": round(n_play / fps, 1), "periods": ctx.get("periods"), "ball_check": ball_check, "possession_pct": {t: round(100 * int((state == i).sum()) / max(1, ctrl)) for i, t in enumerate(("A", "B"))},
               "loose_pct": round(100 * int((state == 2).sum()) / n), "dead_pct": round(100 * int((state == 3).sum()) / n), "attack_right": attack_right, "direction_confidence": conf,
               "turnovers": len(tvs), "passes": len(ps), "restarts": len(rst), "sequences": len(seqs), "shots": {t: sum(1 for s in mx["shots"] if s["team"] == t) for t in ("A", "B")}, "goals": {t: sum(1 for s in mx["goals"] if s["team"] == t) for t in ("A", "B")}, "high_turnovers": mx["high_turnover_counts"], "field_tilt": {t: mx["field"][t]["field_tilt_pct"] for t in ("A", "B")}, "runtime_min": round((time.time() - t0) / 60, 1)}
    log("  step: export"); t_ = time.time()
    root, zpath = EX.write(os.path.join(S.root, "runs"), match_id, video, vi, per, frames_, ball, ballm, state, H, L, W, attack_right, conf, tvs, ps, rst, seqs, ln, sh, st, tm, summary, log=log, periods=ctx.get("periods"), **(export_kw or {}))
    try:
        from .upload import upload_match; upload_match(S.root, match_id, log=log)
    except Exception as e: log(f"upload failed: {e!r}")
    log("\n--- SUMMARY ---"); [log(f"  {k:26s} {v}") for k, v in summary.items()]
    return summary, root, zpath


def run(video_src, match_id=None, settings=None, log=print):
    S = settings or Settings()
    try:
        from .segments import prepare_segments; prepare_segments(S.root, log=log)     # cut full games into segments before the first clip runs
    except Exception as e: log(f"segments: {e!r}")
    match_id = match_id or os.path.splitext(os.path.basename(video_src))[0]
    return analyse(prepare(video_src, match_id, S, log=log), S, log=log)
