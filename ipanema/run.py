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
    from . import linecal as LC
    _lines = LC.find_rows(S.root, match_id)                                   # the line model's full-match calibration, if this match has one
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
        if panorama or _lines: raise StopIteration
        if os.path.exists(pano_cache): Hm = pickle.load(open(pano_cache, "rb")); log(f"calibration: panorama registration cached ({len(Hm)} frames)")
        else:
            Hm = calibrate_via_mosaic(video, match_id, S.root, log=log)
            if Hm: pickle.dump(Hm, open(pano_cache, "wb"))
    except StopIteration: pass
    except Exception as e: log(f"mosaic calibration failed: {e!r}")
    from . import fixedcam as FC
    _fixed = FC.find(match_id)
    if _fixed:                                                                 # 3 Oct: a fixed camera (tactical/broadcast): one homography from hand-read pitch points, every frame
        cal = FC.calibration(_fixed, vi["n"], vi["width"], vi["height"], log=log); L, W = cal["L"], cal["W"]
    elif _lines:
        _base, _off, _rows = _lines
        cal = LC.calibration_for_clip(_rows, vi["n"], vi["fps"], vi["width"], vi["height"], offset_s=_off, log=log)
        L, W = cal["L"], cal["W"]
    elif panorama:
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
    _rf_trk, _rf_kit = f"{cache}/tracks_rfdetr_v1.pkl", f"{cache}/kits_rfdetr_v1.pkl"   # 29 Sep: new players made by modal_app.players_rf (RF-DETR image)
    _use_rf = os.environ.get("IPANEMA_PLAYERS", "rfdetr") == "rfdetr" and not panorama and os.path.exists(_rf_trk) and os.path.exists(_rf_kit)
    if _use_rf:
        tm = pickle.load(open(_rf_kit, "rb")); log(f"players: new pipeline (RF-DETR + per-match kits), from {os.path.basename(_rf_trk)}")
    else:
        if os.environ.get("IPANEMA_PLAYERS", "rfdetr") == "rfdetr" and not panorama: log("players: no RF-DETR tracks for this match yet (run players_rf first) -> old detector")
        T.silence_progress(); tm = T.TeamModel(S.sports_dir).fit(video, S.weights["player"], S.conf_player, log=log)
    from .mosaic import CAL_VERSION
    _trk_name = os.environ.get('IPANEMA_TRACKER', 'bytetrack')
    trk = f"{cache}/tracks_cyl.pkl" if panorama else f"{cache}/tracks_lines_far.pkl" if _lines else f"{cache}/tracks_{('pano_' + CAL_VERSION) if Hm else 'kp'}" + ("" if _trk_name == "bytetrack" else f"_{_trk_name}") + ".pkl"   # bytetrack keeps the original cache name        # positions in metres depend on the calibration: cache per calibration version
    alt_trk = trk.replace(f"tracks_pano_{CAL_VERSION}", "tracks_kp") if (Hm and not panorama) else None
    if _lines: alt_trk = None                                                  # far-band detection is new: detect again (old caches missed far players)
    if _use_rf: per, fps = pickle.load(open(_rf_trk, "rb")); per = TR.reposition(per, H); log("tracking: RF-DETR tracks re-positioned with this calibration")
    elif os.path.exists(trk): per, fps = pickle.load(open(trk, "rb")); log("tracking: cached")
    elif alt_trk and alt_trk != trk and os.path.exists(alt_trk):
        # detections are made in the picture; only metres depend on calibration -> re-position, don't re-detect
        per, fps = pickle.load(open(alt_trk, "rb")); per = TR.reposition(per, H); pickle.dump((per, fps), open(trk, "wb"))
        log(f"tracking: reused detections, re-positioned with the {'line' if _lines else 'panorama'} calibration")
    else: per, fps = TR.track(video, S.weights["player"], H, tm, 0.1 if panorama else S.conf_player, log=log, imgsz=2560 if panorama else 960, tiles=None if panorama else TR.FOLLOW_TILES, pano=panorama); pickle.dump((per, fps), open(trk, "wb"))   # panorama: whole frame at full resolution (measured 12:29: 23 players/frame at 7.4 frames/s; 6 tiles found 24 at 1.8 frames/s)
    ball_backend = os.environ.get("IPANEMA_BALL", "wasb")
    _clicks_w = os.path.join(S.root, "models", "ball", "clicks_latest.pt")     # the click-trained ball model (26 Sep: 52/71 vs 38/71 old)
    if "IPANEMA_BALL" not in os.environ and os.path.exists(_clicks_w): ball_backend = "clicks"
    from . import ballrf as BRF                                                # 29 Sep: RF-DETR ball finder (exam 84/108 vs 70), guesses made by ball_rf in the RF-DETR image
    _rf_ball = f"{cache}/ball_cands_{BRF.VERSION}.pkl"
    if os.environ.get("IPANEMA_BALL", "rfdetr") == "rfdetr" and os.path.exists(_rf_ball): ball_backend = "rfdetr"
    cands = None
    if ball_backend == "rfdetr":
        cands = pickle.load(open(_rf_ball, "rb")); log(f"ball: RF-DETR ball finder ({BRF.VERSION}), {sum(len(v) for v in cands.values())/max(1,len(cands)):.1f} candidates/frame")
        if os.environ.get("IPANEMA_BALL_FUSE", BRF.FUSE_WASB) == "1":
            try:
                from . import wasb
                wb = wasb.candidates(video, S.root, f"{cache}/ball_cands_wasb.pkl", log=log, train=train_ball, thr=float(os.environ.get("IPANEMA_WASB_THR", "0.05")))
                cands = BL.fuse_candidates(cands, wb); log(f"ball: RF-DETR + WASB guesses fused, {sum(len(v) for v in cands.values())/max(1,len(cands)):.1f}/frame")
            except Exception as e: log(f"ball: WASB merge skipped ({e!r})")
    if ball_backend == "clicks":
        from . import ballclicks as BC
        _bj = os.path.join(S.root, "models", "ball", "best.json")                # 28 Sep: cache per model version, or a promoted model reuses old guesses
        _ver = json.load(open(_bj)).get("weights", "v")[:-3] if os.path.exists(_bj) else f"{int(os.path.getmtime(_clicks_w))}"
        clicks = BC.candidates(video, _clicks_w, f"{cache}/ball_cands_clicks_{_ver}.pkl", S.conf_ball, log=log)
        log(f"ball: click-trained model ({os.path.basename(_clicks_w)}), {sum(len(v) for v in clicks.values())/max(1,len(clicks)):.1f} candidates/frame")
        try:                                                                   # 27 Sep: fused guesses (tested on the clip: 26/34 vs 22/34, ceiling 27)
            from . import wasb
            wb = wasb.candidates(video, S.root, f"{cache}/ball_cands_wasb.pkl", log=log, train=train_ball, thr=float(os.environ.get("IPANEMA_WASB_THR", "0.05")))
            cands = BL.fuse_candidates(clicks, wb)
            log(f"ball: WASB + click-model guesses fused, {sum(len(v) for v in cands.values())/max(1,len(cands)):.1f}/frame")
        except Exception as e: log(f"ball: WASB merge skipped ({e!r})"); cands = clicks
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


def _ball_gt(S, match_id):
    """ball answer key: the volume copy first, else the one in the repo (1 Oct: AIK key from the blind A/B check)"""
    p = os.path.join(S.root, "reference", match_id, "ball_gt.json")
    if os.path.exists(p): return p
    q = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", match_id, "ball_gt.json")
    return q if os.path.exists(q) else p


def periods_direction(root, match_id):
    """4 Oct: which way the dark team (A) attacks in the first period, checked by eye from a goal (who celebrates / who kicks
    off) and written in periods/<match>.json as "attack_right_A". The automatic guess is near a coin flip on our clips
    (confidence 0.01-0.10) and gave SFK-BP's first-half goal to BP. Later periods are mirrored, so one value covers the match."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(root, "periods", f"{match_id}.json"), os.path.join(here, "periods", f"{match_id}.json")):
        if os.path.exists(p):
            v = json.load(open(p)).get("attack_right_A")
            if v is not None: return bool(v)
    return None

def veo_in_window(t, duration_s, periods, slack_s=5.0):
    """3 Oct: a Veo shot/goal counts only inside the analysed video and (when periods are set) inside a period +- slack.
    SFK-BP: Veo's list had a 'goal' clip named 00:00:28 (before kick-off) and the second-half goals; a first-half cut must not show them."""
    if t > duration_s: return False
    return not periods or any(p["t_start"] - slack_s <= t <= p["t_end"] + slack_s for p in periods)

def analyse(ctx, S, log=print, export_kw=None, gt_path=None):
    """from raw tracks + ball candidates to stats, events and the exported match (CPU)"""
    match_id, video, vi, H, L, W, cal, tm, per, fps, cands, t0 = (ctx[k] for k in ("match_id", "video", "vi", "H", "L", "W", "cal", "tm", "per", "fps", "cands", "t0"))
    per, cl = TR.clean(per, L, W, fps, log=log)
    per, _nf = TR.fill_gaps(per, fps, 1.0, H=H)                              # 28 Sep: players the detector drops for < 1 s (dark kits blink: median track 0.8 s)
    log(f"players: {_nf} short gaps filled (marked 'filled' in the export)")
    try:                                                                       # 1 Oct: the picker's exact inputs, so it can be tuned for free offline ([fetch:cache/<match>/picker_inputs.pkl])
        pickle.dump({"cands": cands, "H": H, "L": L, "W": W, "fps": fps, "per": {k: [[r[0], r[1], r[2], r[3]] for r in v] for k, v in per.items()}},
                    open(os.path.join(S.root, "cache", match_id, "picker_inputs.pkl"), "wb"))
    except Exception as e: log(f"picker inputs not saved ({e!r})")
    def _v1():
        g = BL.pick_global(cands, H, L, W, per=per, fps=fps, log=log)
        if len(g) < 0.2 * len(cands): log("ball: global path too sparse, falling back to trajectory picker"); g = BL.pick(cands, H, L, W, per=per, log=log)
        return BL.bridge(g, fps)
    bridged = set()
    def _v2(): return BL.bridge(BL.pick_v2(cands, H, L, W, per=per, fps=fps, log=log), fps, bridged=bridged)
    picker = os.environ.get("IPANEMA_PICKER", "v2")
    ball = _v2() if picker == "v2" else _v1()
    other = (_v1 if picker == "v2" else _v2) if cands else None
    ball_check = BL.check(ball, cands, gt_path or _ball_gt(S, match_id), log=log, video=video, debug_dir=f"/content/ipanema-analysis/results/debug/ballcheck_{match_id}")
    # the other picker on the same frames, logged side by side (CPU only)
    try:
        if other is not None:
            o = BL.check(other(), cands, gt_path or _ball_gt(S, match_id), log=lambda *a: None)
            if o: log(f"ball check (other picker, {'v1' if picker == 'v2' else 'v2'}): {o['correct']}/{o['total']} correct, ceiling {o['ceiling']}/{o['total']}")
        for pg in ctx.get("picker_gt") or []:
            a = BL.check(ball, cands, pg, log=lambda *a: None); b = BL.check(other(), cands, pg, log=lambda *a: None) if other else None
            if a: log(f"picker test on detector-training clicks ({os.path.basename(os.path.dirname(pg))}, {a['total']} frames; detection is optimistic here, the pick is the fair part): {picker} {a['correct']}/{a['total']}" + (f", other {b['correct']}/{b['total']}" if b else "") + f", ceiling {a['ceiling']}")
    except Exception as e: log(f"picker comparison failed: {e!r}")
    if ctx.get("cands_alt"):
        try:
            alt = BL.pick_global(ctx["cands_alt"], H, L, W, per=per, fps=fps, log=lambda *a: None)
            alt_check = BL.check(BL.bridge(alt, fps), ctx["cands_alt"], gt_path or _ball_gt(S, match_id), log=lambda *a: None)
            if alt_check: log(f"ball check (alternative: WASB + YOLO candidates): {alt_check['correct']}/{alt_check['total']} correct, ceiling {alt_check['ceiling']}/{alt_check['total']}")
        except Exception as e: log(f"alternative ball check failed: {e!r}")
    frames_, ballm = P.carriers(per, ball, H, S.carrier_r, S.near_r)
    state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=log)   # E4 (29 Sep): possession_simple by default; IPANEMA_POSSESSION=viterbi = old model
    attack_right, conf = P.direction(state, ballm, log=log)
    import glob as _g
    ref = next(iter(_g.glob(os.path.join(S.root, "reference", f"events_gt_{match_id}.json")) + _g.glob(os.path.join(S.root, "reference", match_id, "events_gt*.json"))), None)
    ref_dir = json.load(open(ref)).get("attack_right_A") if ref else None
    if ref_dir is None: ref_dir = periods_direction(S.root, match_id)
    if ref_dir is not None:
        attack_right = {"A": bool(ref_dir), "B": not ref_dir}; log(f"direction: from reference labels -> A {'→' if attack_right['A'] else '←'}")
    elif min(conf.values()) < 0.15:
        attack_right = P.direction_from_keepers(per, L, log=log) or P.direction_fallback(per, L, log=log)
    _t = [time.time()]
    def step(name): log(f"  step: {name} ({time.time() - _t[0]:.0f} s since the last step)"); _t[0] = time.time()   # 3 Oct: step timers were started and never reported
    step("sequences")
    _take = float(os.environ.get("IPANEMA_SPELL_TAKE", P.SPELL_TAKE_S)); cstate = P.spell_state(state, fps, _take, P.SPELL_JOIN_S)   # S8: counted stats read the de-flickered state
    _fl = lambda st: int(sum(1 for i in range(1, len(st)) if st[i] < 2 and st[i - 1] < 2 and st[i] != st[i - 1]))
    log(f"  spell state (take {_take} s): possession flips {_fl(state)} -> {_fl(cstate)} ({_fl(cstate) / max(1e-6, len(per) / fps / 60):.1f} per minute)")
    seqs = P.sequences(cstate, ballm, bspeed, fps, L, attack_right, **pinfo["seq"]); rst = P.restarts(dstate, ballm, fps, L, W)
    step("turnovers")
    tvs = P.turnovers(per, frames_, cstate, ballm, fps, attack_right, S.press_r, S.near_r, min_before_s=pinfo["turnover_s"], min_after_s=pinfo["turnover_s"])
    step("lanes")
    ln = AN.lanes(per, frames_, attack_right, S.lane_half, S.max_lane)
    step("passes")
    # 4 Oct (S4b): passes only where the de-flickered possession (spell state) agrees. SFK-BP clip, 28 graded passes:
    # fake 8 -> 2 kept, real 20 -> 14; 87 -> 59 passes (11.8/min, a plausible U15 rate; ~62 real by the key), completion 71 -> 78%.
    ps, tracks = AN.passes(per, frames_, tvs, ln, attack_right, fps, state=(cstate if os.environ.get("IPANEMA_PASS_STATE", "spell") == "spell" else None))
    _kf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "overrides", f"{match_id}_kicks.json")
    if os.path.exists(_kf) and os.environ.get("IPANEMA_PASS_CONFIRM", "1") == "1":   # P-PASS (6 Oct): passes confirmed by the video ball-action model
        _n = len(ps); ps = AN.confirm_passes(ps, json.load(open(_kf))["kicks"], float(os.environ.get("IPANEMA_PASS_CONFIRM_TOL", "0.7")))
        log(f"passes: {_n} -> {len(ps)} confirmed by the video ball-action model ({os.path.basename(_kf)})")
    _keep = float(os.environ.get("IPANEMA_LOSS_KEEP_S", P.LOSS_KEEP_S)); _nt = len(tvs)   # S9 (8 Oct): a duel is a lost ball only if the winner keeps it / passes
    tvs, _duels = P.confirm_losses(tvs, cstate, ps, fps, _keep)
    log(f"lost balls: {_nt} -> {len(tvs)} (duels dropped: winner kept it < {_keep} s and played no pass; IPANEMA_LOSS_KEEP_S=0 = off)")
    step("shapes")
    sh = AN.shapes(per, L, fps)
    step("stats")
    st = AN.stats(per, frames_, tvs, ps, tracks, state, fps, L, W, attack_right, S.press_r, S.near_r, sequences_=seqs)
    step("metrics")
    mx = M.compute(state, ballm, bspeed, fps, L, W, attack_right, rst, ps, st['players'], tvs, sh, per=per, frames_=frames_); st['metrics'] = mx
    # Veo's own shots/goals (to the second), when we have them, replace our shot detector; ours keeps being scored against them
    try:
        from . import veo as VEO
        vp = next((p for p in (os.path.join(S.root, "reference", f"veo_highlights_{match_id}.txt"),
                               os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", f"veo_highlights_{match_id}.txt")) if os.path.exists(p)), "")
        if os.path.exists(vp):
            vs, rej = VEO.build(VEO.load(vp), fps, ballm, per, L, W, attack_right, rst, periods=ctx.get("periods"), log=log, known_teams=VEO.teams(vp), known_origins=VEO.origins(vp))
            # 3 Oct: only shots inside the analysed video and its match periods (a first-half cut must not carry second-half goals)
            _in = lambda t: veo_in_window(t, len(per) / fps, ctx.get("periods"))
            _drop = [x for x in vs if not _in(x["t"])]; vs = [x for x in vs if _in(x["t"])]
            if _drop: log(f"veo: {len(_drop)} shots/goals outside the analysed video or its periods left out ({sum(1 for x in _drop if x['goal'])} goals)")
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
        "possession_ok": bool((acc >= 0.6) if acc is not None else (near_frac >= 0.85 and len(ballm) > 0.4 * n_play)),   # 27 Sep: the checked accuracy decides when we have it (the app hides the ball layer otherwise)
        # events need the ball in the right place at the right moment
        "events_ok": bool(acc is not None and acc >= 0.6 and near_frac >= 0.85),
    }
    log(f"ball grade: accuracy {ball_grade['accuracy']}, near-player {ball_grade['near_player_pct']}, possession {'OK' if ball_grade['possession_ok'] else 'withheld'}, events {'OK' if ball_grade['events_ok'] else 'withheld'}")
    ball_reliable = bool(ball_grade["possession_ok"])      # one verdict everywhere: the old flag follows the stricter grade
    log(f"ball reliability: {len(ballm)/n:.0%} frames, {len(near)/max(1,len(ballm)):.0%} near a player, {jumps} jumps -> {'OK' if ball_reliable else 'UNRELIABLE (stats withheld in app)'}")
    summary = {"match_id": match_id, "ball_reliable": ball_reliable, "ball_grade": ball_grade, "duration_s": round(n / fps, 1), "calibration_coverage": round(cal["coverage"], 2), "calibration_frozen": cal["frozen"],
               "team_dark_share": tm.dark_share, "players_per_frame_median": {t: float(np.median([sum(1 for r in per[k] if r[1] == t) for k in play_ks] or [0])) for t in ("A", "B")},
               "ball_frames_pct": round(100 * len(ball) / max(1, n_play)), "match_seconds": round(n_play / fps, 1), "periods": ctx.get("periods"), "ball_check": ball_check, "possession_pct": {t: round(100 * int((state == i).sum()) / max(1, ctrl)) for i, t in enumerate(("A", "B"))},
               "loose_pct": round(100 * int((state == 2).sum()) / n), "dead_pct": round(100 * int((np.asarray(dstate) == 3).sum()) / n), "possession_model": pinfo["mode"], "attack_right": attack_right, "direction_confidence": conf,
               "turnovers": len(tvs), "passes": len(ps), "restarts": len(rst), "sequences": len(seqs), "shots": {t: sum(1 for s in mx["shots"] if s["team"] == t) for t in ("A", "B")}, "goals": {t: sum(1 for s in mx["goals"] if s["team"] == t) for t in ("A", "B")}, "high_turnovers": mx["high_turnover_counts"], "field_tilt": {t: mx["field"][t]["field_tilt_pct"] for t in ("A", "B")}, "runtime_min": round((time.time() - t0) / 60, 1)}
    step("export")
    try:                                                                       # 1 Oct: speed + distance per player for the app's speed layer
        from . import motion as MO
        if os.environ.get("IPANEMA_MOTION", "0") != "1": raise StopIteration("off until speeds pass the by-eye check (results/review/speedcheck_2026-10-01.md)")
        motion = MO.compute(per, fps, H=H); log(f"motion: speed layer {MO.summary(motion, fps)}")
    except (Exception, StopIteration) as e: motion = None; log(f"motion skipped ({e!r})")
    root, zpath = EX.write(os.path.join(S.root, "runs"), match_id, video, vi, per, frames_, ball, ballm, state, H, L, W, attack_right, conf, tvs, ps, rst, seqs, ln, sh, st, tm, summary, log=log, periods=ctx.get("periods"), unsure=cal.get("unsure", frozenset()), cands_conf=BL.pick_confidence(ball, cands), bridged=bridged, motion=motion, **(export_kw or {}))
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
