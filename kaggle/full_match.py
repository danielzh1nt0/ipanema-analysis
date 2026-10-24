# Kaggle (free GPU / CPU), 16 Oct 2026 (F1): a whole match end to end WITHOUT Modal - Daniel's rule from 16 Oct: no paid runs.
# Same code path as modal_app.rf_full + run_full, piece by piece:
#   STAGE=pieces  cut 5-min pieces of the match video (R2), one kit model for the match, per piece: RF-DETR players,
#                 RF-DETR ball, run.prepare (line calibration + WASB fused) -> cache/<pid>/piece_v5.pkl + calib_ok_v3.npy
#                 -> /kaggle/working/root/cache (kept on Kaggle; a later kernel mounts it as an input)
#   STAGE=join    join the pieces (from the mounted piece kernels), analyse, export -> /kaggle/working/export/<match>/
#                 (the free GitHub runner then uploads the export to the app's storage)
# Placeholders filled by tools/kaggle_run.py: {{R2}} {{MATCH}} {{STAGE}} {{PIECES}} (e.g. 0-11, all) {{WASB}} (repo path of
# the WASB weights to use, default results/volume/models/wasb_finetuned.pth = the app's) {{SOURCES_DIRS}} (join: the mounted
# piece outputs, comma separated, e.g. /kaggle/input/ipanema-full-match-a,/kaggle/input/ipanema-full-match-b)
import subprocess, os, sys, json, time, glob, shutil, pickle, traceback
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
t0 = time.time(); R2 = "{{R2}}".rstrip("/"); M = "{{MATCH}}"; STAGE = "{{STAGE}}" or "pieces"; PIECES = "{{PIECES}}" or "all"; WASB_W = "{{WASB}}" or "results/volume/models/wasb_finetuned.pth"
W = "/kaggle/working"; T = "/kaggle/temp"; IA = f"{T}/ia"; ROOT = f"{W}/root"; BUDGET_MIN = float("{{BUDGET}}" or "690")   # Kaggle stops a kernel at 12 h
log_lines = []
if "{{NEAR}}": os.environ["IPANEMA_NEAR_M"] = "{{NEAR}}"                            # T1: possession nearness in metres (default 1.5 in ipanema.possession)
def log(*a):
    s = " ".join(str(x) for x in a); log_lines.append(f"{(time.time() - t0) / 60:6.1f} {s}"); print(s, flush=True)
    try: open(f"{W}/log.txt", "a").write(log_lines[-1] + "\n")
    except Exception: pass
sh("nvidia-smi --query-gpu=name --format=csv || true; python -c 'import torch;print(torch.__version__, torch.cuda.is_available())'")
sh(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {IA}"); sh(f"cd {IA} && git log -1 --format='repo at %h %s'")
sh("pip install -q rfdetr==1.11.0 'supervision>=0.29,<0.31' omegaconf gdown 'numpy<2' scipy scikit-learn pandas ultralytics==8.3.40")   # ultralytics only as a fallback import (calibration.py) - not used with a line calibration
sh("which ffmpeg || (apt-get install -y -q ffmpeg > /dev/null 2>&1 && which ffmpeg)")
# the pipeline's Colab-era default path for the repo (mosaic calibration reads calibration/ and results/prepare/ from it)
if not os.path.exists("/content/ipanema-analysis"): os.makedirs("/content", exist_ok=True); os.symlink(IA, "/content/ipanema-analysis")
os.makedirs(f"{ROOT}/cache", exist_ok=True); os.makedirs(f"{ROOT}/models", exist_ok=True); os.makedirs(f"{ROOT}/videos", exist_ok=True); os.makedirs(f"{T}/work", exist_ok=True)
sys.path.insert(0, IA); os.environ["WASB_DIR"] = f"{T}/WASB-SBDT"; os.environ["IPANEMA_DETECTOR"] = "rfdetr"; os.environ["IPANEMA_RF_FP16"] = "1"; os.environ.setdefault("IPANEMA_DET_BATCH", "8"); os.environ.setdefault("IPANEMA_LOG_EVERY", "2000")
# the app's WASB weights (or the ones named): the pipeline looks for models/wasb_finetuned.pth under the root
if os.path.exists(f"{IA}/{WASB_W}"): shutil.copy(f"{IA}/{WASB_W}", f"{ROOT}/models/wasb_finetuned.pth"); json.dump({"pinned": True, "source": WASB_W}, open(f"{ROOT}/models/wasb_train_manifest.json", "w")); log(f"wasb weights: {WASB_W}")
else: log(f"wasb weights {WASB_W} not in the repo: the pretrained soccer weights will be used")
for d in ("reference", "periods", "overrides", "results/volume/reference"):     # answer keys / match settings kept in the repo
    if os.path.isdir(f"{IA}/{d}"): shutil.copytree(f"{IA}/{d}", f"{ROOT}/{os.path.basename(d)}", dirs_exist_ok=True)
from ipanema.config import Settings
from ipanema import fullmatch as FM, linecal as LC
S = Settings(root=ROOT, sports_dir=f"{T}/sports", work=f"{T}/work")
# a match without a line calibration in the repo (a fresh upload) falls back to the pitch-keypoint model from roboflow/sports,
# as the Modal run did; that needs the sports checkout and its weights (setup.sh downloads them)
if STAGE in ("pieces", "all") and not LC.find_rows(ROOT, M) and not os.path.isdir(f"{T}/sports"):
    sh(f"git clone -q https://github.com/roboflow/sports.git {T}/sports && pip install -q -e {T}/sports && cd {T}/sports/examples/soccer && bash setup.sh"); log("calibration: no line calibration for this match - pitch-keypoint model installed (roboflow/sports)")
full = f"{ROOT}/videos/{M}/full.mp4"
def get_video():
    if os.path.exists(full) and os.path.getsize(full) > 10_000_000: return
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if sh(f"wget -q -O {full} {R2}/{M}/video.mp4") or os.path.getsize(full) < 10_000_000: raise SystemExit(f"video {R2}/{M}/video.mp4 not reachable")
    log(f"video: {os.path.getsize(full) / 1e9:.2f} GB")
result = {"match": M, "stage": STAGE, "pieces_done": [], "pieces_failed": []}
def save(): result["minutes"] = round((time.time() - t0) / 60, 1); json.dump(result, open(f"{W}/result.json", "w"), indent=1, default=str)

def mount_sources():
    srcs = [x for x in "{{SOURCES_DIRS}}".split(",") if x]                        # outputs of earlier kernels (kits, pieces)
    if srcs: log("inputs: " + "; ".join(f"{d}: {sorted(os.listdir(d))[:8]}" for d in glob.glob("/kaggle/input/*") if os.path.isdir(d)))
    for slug in srcs:
        d = next((x for x in glob.glob(f"/kaggle/input/{slug}") + glob.glob(f"/kaggle/input/*/{slug}") + glob.glob(f"/kaggle/input/*/*/{slug}") if os.path.isdir(x)), f"/kaggle/input/{slug}")   # notebooks/<user>/<slug>
        found = glob.glob(f"{d}/root/cache/*") or glob.glob(f"{d}/*/root/cache/*") or glob.glob(f"{d}/cache/*")
        for c in found: shutil.copytree(c, f"{ROOT}/cache/{os.path.basename(c)}", dirs_exist_ok=True)
        log(f"mounted {d}: {len(found)} cache folders")
        if not found: raise SystemExit(f"input {d} has no cache folders (listing above): not running without it")

if STAGE in ("kits", "pieces", "all"):                                           # all = pieces + join in one kernel (a short clip)
    import numpy as np, cv2
    from ipanema import tracking as TR, video as V, ballrf as BR, kits as KT, calcheck as CC, linecal as LC
    mount_sources(); get_video(); n, fps = FM.video_info(full); plan_ = FM.plan(n, fps); result.update(n=n, fps=fps, n_pieces=len(plan_))
    # which pieces this kernel does: "all", "a-b", "j/N" (every N-th piece starting at j, only pieces with play - the kernels
    # of one match run in parallel without knowing the piece count), or "missing" (whatever the mounted kernels left undone)
    def _plays(i):
        pf0 = f"{ROOT}/periods/{M}.json"
        if not os.path.exists(pf0): return True
        a, b = plan_[i]["start_s"], plan_[i]["start_s"] + plan_[i]["dur_s"]
        return any(x < b and a < y for x, y in json.load(open(pf0))["periods_s"])
    done_ = lambda i: os.path.exists(f"{ROOT}/cache/{FM.piece_id(M, i)}/{FM.PIECE_FILE}")
    if STAGE == "kits": want = []
    elif PIECES in ("all", ""): want = list(range(len(plan_)))
    elif PIECES == "missing": want = [i for i in range(len(plan_)) if _plays(i) and not done_(i)]
    elif "/" in PIECES: j, N = (int(x) for x in PIECES.split("/")); want = [i for i in range(len(plan_)) if i % N == j and _plays(i) and not done_(i)]
    else: want = list(range(int(PIECES.split("-")[0]), int(PIECES.split("-")[1]) + 1))
    log(f"=== {M}: {n} frames ({n / fps:.0f} s) @ {fps:.3f} fps -> {len(plan_)} pieces; this kernel: {STAGE} {want}")
    # one kit model for the whole match (modal_app.rf_kits): 36 frames spread over the playing time. Made once (STAGE=kits)
    # and mounted by the piece kernels, so two kernels running in parallel can never number the teams differently.
    kit_path = f"{ROOT}/cache/{M}/kits_rfdetr_v1.pkl"; os.makedirs(os.path.dirname(kit_path), exist_ok=True)
    if os.path.exists(kit_path): tm = pickle.load(open(kit_path, "rb")); log("kits: mounted from the kits kernel")
    else:
        pf = f"{ROOT}/periods/{M}.json"; spans = json.load(open(pf))["periods_s"] if os.path.exists(pf) else [[0.15 * n / fps, 0.45 * n / fps], [0.55 * n / fps, 0.9 * n / fps]]
        cap = cv2.VideoCapture(full); ts = np.concatenate([np.linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), 18) for a, b in spans]); det = TR.RFDetrPerson("medium"); fb = []
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
            if ok: fb.append((f, np.array([b for b in det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0].xyxy])))
        cap.release(); tm = KT.KitTeamModel().fit_frames(fb, log=log); tm.offpitch = False; pickle.dump(tm, open(kit_path, "wb"))
        result["kits"] = {"frames": len(fb), "kit_groups": [int(x) for x in tm.model["sizes"]], "dark_share": {k: float(v) for k, v in tm.dark_share.items()}}; log(f"kits: {result['kits']}"); save()
    for i in want:
        if (time.time() - t0) / 60 > BUDGET_MIN - 40: log(f"time budget: stopping before piece {i}"); result["stopped_before"] = i; break
        p = plan_[i]; pid = FM.piece_id(M, i); cache = f"{ROOT}/cache/{pid}"; os.makedirs(cache, exist_ok=True); tp = time.time()
        try:
            src = FM.cut(full, f"{ROOT}/videos/{pid}.mp4", p["start_s"], p["dur_s"]); video = V.normalise(src, f"{T}/work/{pid}/video.mp4")
            shutil.copy(kit_path, f"{cache}/kits_rfdetr_v1.pkl")
            if not os.path.exists(f"{cache}/tracks_rfdetr_v1.pkl"):                 # modal_app.rf_piece: players in picture space, re-positioned by prepare()
                npc = V.info(video)["n"]; S_ = np.array([[1920 / 106, 0, 0], [0, 1080 / 64, 0], [0, 0, 1.0]]); Hs = {k: S_ for k in range(npc)}
                per, pfps = TR.track(video, None, Hs, tm, 0.3, log=log, tiles=TR.FOLLOW_TILES, imgsz=960, pano=False); pickle.dump((per, pfps), open(f"{cache}/tracks_rfdetr_v1.pkl", "wb"))
                log(f"  {pid}: {sum(len(v) for v in per.values()) / max(1, len(per)):.1f} players/frame")
            BR.candidates(video, f"{IA}/{BR.WEIGHTS}", f"{cache}/ball_cands_{BR.VERSION}.pkl", log=log, batch=4)
            path = FM.process_piece(full, M, p, S, log=log)                         # prepare(): line calibration, WASB fused with RF-DETR ball -> piece_v5.pkl
            pk = pickle.load(open(path, "rb"))
            if "unsure" in pk and LC.find_rows(ROOT, pid): ok = np.ones(int(pk["n"]), bool); ok[[k for k in pk["unsure"] if k < len(ok)]] = False
            else: ok, _ = CC.confidence_mask(f"{ROOT}/videos/{pid}.mp4", pk["H"], int(pk["n"]), pk["L"], pk["W"])
            np.save(f"{cache}/calib_ok_v3.npy", ok)
            result["pieces_done"].append({"i": i, "minutes": round((time.time() - tp) / 60, 1), "trusted_pct": round(100 * float(ok.mean()), 1)}); log(f"piece {i} done in {(time.time() - tp) / 60:.1f} min, {100 * ok.mean():.0f}% frames trusted")
        except Exception:
            result["pieces_failed"].append({"i": i, "error": traceback.format_exc()[-1500:]}); log(f"PIECE {i} FAILED: " + traceback.format_exc()[-800:])
        for f in (f"{ROOT}/videos/{pid}.mp4", f"{T}/work/{pid}"): shutil.rmtree(f, ignore_errors=True) if os.path.isdir(f) else (os.remove(f) if os.path.exists(f) else None)
        for f in glob.glob(f"{cache}/ball_cands_wasb*.pkl"): os.remove(f)         # the fused candidates are inside piece_v5.pkl
        save()
    if STAGE != "all": os.remove(full)                                            # the output must stay small: the video is on R2

if STAGE == "relabel":
    # K3c on the new tracks (17 Oct): the live track vote files a sunlit black player as "K" (neither team) when more than
    # 60% of his reads say "neither" -> 3-5 K rows per frame on Vallentuna, removed at the join, SFK possession 34% vs 56% by
    # eye. Here every track is re-read from the video with the match kit model and decided by the majority of its A/B
    # reads (at least MIN_AB), as tools/vall_relabel.py did for the Modal run. No detection: the boxes are in the pieces.
    import numpy as np, cv2, collections
    from ipanema import kits as KT
    mount_sources(); get_video(); n, fps = FM.video_info(full); plan_ = FM.plan(n, fps)
    STEP = int("{{STEP}}" or "6"); MIN_AB = int("{{MIN_AB}}" or "3")
    tm = pickle.load(open(f"{ROOT}/cache/{M}/kits_rfdetr_v1.pkl", "rb")); votes = collections.defaultdict(collections.Counter); old = {}; series = collections.defaultdict(list)
    cap = cv2.VideoCapture(full); n_fr = 0
    for p in plan_:
        pid = FM.piece_id(M, p["i"]); pf_ = f"{ROOT}/cache/{pid}/{FM.PIECE_FILE}"
        if not os.path.exists(pf_): continue
        per = pickle.load(open(pf_, "rb"))["per"]; off = p["offset"]
        for k in sorted(per):
            if k % STEP or not per[k]: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, off + k); ok, f = cap.read()
            if not ok: continue
            rows = [r for r in per[k] if r[4] is not None]; labs = tm.predict_batch(f, [list(map(float, r[4])) for r in rows]); n_fr += 1
            for r, lab in zip(rows, labs):
                tid = int(r[0]) + p["i"] * FM.ID_STRIDE if r[0] >= 0 else int(r[0])   # the id this track has after FM.join
                votes[tid][lab] += 1; old[tid] = r[1]
                if lab in ("A", "B"): series[tid].append((off + k, lab))
        log(f"relabel: {pid} read ({n_fr} frames so far, {len(votes)} tracks)")
    cap.release(); teams = {}; changed = collections.Counter(); n_split = 0
    for tid, v in votes.items():
        a, b = v.get("A", 0), v.get("B", 0)
        if a + b >= MIN_AB: new = "A" if a >= b else "B"
        else: continue
        # a track that switches between two people (tracker id swap, black <-> red): segments of >= SEG consecutive
        # smoothed reads of the other team get their own team, as [t_start, t_end, team] pieces (fullmatch.apply_team_override)
        parts = FM.team_segments(series[tid], fps, seg=int("{{SEG}}" or "4"))   # same rule, moved to ipanema/fullmatch.py (tests/test_t1_segments.py)
        if parts:
            teams[str(tid)] = parts; n_split += 1; changed[f"{old.get(tid)}->split"] += 1; continue
        if new != old.get(tid): changed[f"{old.get(tid)}->{new}"] += 1
        teams[str(tid)] = new
    os.makedirs(f"{ROOT}/overrides", exist_ok=True); json.dump(teams, open(f"{ROOT}/overrides/{M}_teams.json", "w"))
    result["relabel"] = {"tracks_read": len(votes), "decided": len(teams), "changed": dict(changed), "split": n_split, "frames": n_fr, "step": STEP}; log(f"relabel: {result['relabel']}")
    os.remove(full)

if STAGE == "kitprobe":
    # T1 (17 Oct): what the kit reader sees per track - 4 crops of each sampled track + its colour features + reads,
    # to judge by eye (Claude) and fit a better rule for sun + shade. CPU. Output: sheets/*.jpg + probe.json
    import numpy as np, cv2, collections, random
    from ipanema import kits as KT
    mount_sources(); get_video(); n, fps = FM.video_info(full); plan_ = FM.plan(n, fps)
    tm = pickle.load(open(f"{ROOT}/cache/{M}/kits_rfdetr_v1.pkl", "rb")); NT = int("{{NTRACKS}}" or "96")
    occ = collections.defaultdict(list)                                           # tid -> [(global frame, box, team)]
    for p in plan_:
        pf_ = f"{ROOT}/cache/{FM.piece_id(M, p['i'])}/{FM.PIECE_FILE}"
        if not os.path.exists(pf_): continue
        for k, rows in pickle.load(open(pf_, "rb"))["per"].items():
            for r in rows:
                if r[4] is not None and r[0] >= 0: occ[int(r[0]) + p["i"] * FM.ID_STRIDE].append((p["offset"] + k, [float(v) for v in r[4]], r[1]))
    long = [t for t, v in occ.items() if len(v) >= 60]; random.Random(3).shuffle(long); picks = long[:NT]; log(f"kitprobe: {len(occ)} tracks, {len(long)} with >= 2 s, probing {len(picks)}")
    cap = cv2.VideoCapture(full); probe = []; strips = []
    for tid in picks:
        v = sorted(occ[tid]); idx = [v[int(i)] for i in np.linspace(0, len(v) - 1, 4)]; crops = []; feats = []
        for k, box, team in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
            if not ok: continue
            g = KT.grass_lab(f); fe = KT.torso_feature(f, box, g, light=getattr(tm, "light", "frame")); lab = tm.predict_batch(f, [box])[0]
            _, hue = KT._kit_hue(tm.model); r, d, wh = KT.kit_share(f, box, hue, white=True)
            feats.append({"frame": k, "L": None if fe is None else round(float(fe[0]), 1), "a": None if fe is None else round(float(fe[1]), 1), "b": None if fe is None else round(float(fe[2]), 1), "red": None if r is None else round(float(r), 3), "dark": None if d is None else round(float(d), 3), "white": None if wh is None else round(float(wh), 3), "read": lab, "track_team": team})
            x1, y1, x2, y2 = [int(z) for z in box]; h = y2 - y1; cx = (x1 + x2) // 2; w = max(24, int(h * 0.6))
            c = f[max(0, y1 - 4):min(f.shape[0], y2 + 4), max(0, cx - w // 2):min(f.shape[1], cx + w // 2)]
            crops.append(cv2.resize(c, (72, 144)) if c.size else np.zeros((144, 72, 3), np.uint8))
        while len(crops) < 4: crops.append(np.zeros((144, 72, 3), np.uint8))
        strip = np.hstack(crops); cv2.putText(strip, str(tid)[-5:], (2, 12), 0, 0.4, (255, 255, 255), 1)
        strips.append(strip); probe.append({"tid": tid, "n": len(v), "feats": feats, "votes": dict(collections.Counter(fx["read"] for fx in feats))})
    cap.release(); os.makedirs(f"{W}/sheets", exist_ok=True)
    for si in range(0, len(strips), 24):                                           # 24 tracks per sheet, 4 columns
        rows_ = [np.hstack(strips[si + r * 4: si + r * 4 + 4] + [np.zeros((144, 288, 3), np.uint8)] * (4 - len(strips[si + r * 4: si + r * 4 + 4]))) for r in range(6) if si + r * 4 < len(strips)]
        cv2.imwrite(f"{W}/sheets/sheet_{si // 24:02d}.jpg", np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(probe, open(f"{W}/probe.json", "w")); result["kitprobe"] = {"tracks": len(probe), "sheets": len(range(0, len(strips), 24)), "light": getattr(tm, "light", None)}; os.remove(full)

if STAGE in ("join", "all"):
    import numpy as np, types
    from ipanema.run import analyse
    mount_sources(); get_video(); n, fps = FM.video_info(full); plan_ = FM.plan(n, fps)
    cdir = lambda p: f"{ROOT}/cache/{FM.piece_id(M, p['i'])}"
    done = [p for p in plan_ if os.path.exists(f"{cdir(p)}/{FM.PIECE_FILE}")]
    log(f"=== {M} join: {len(done)}/{len(plan_)} pieces present"); result["pieces_present"] = [p["i"] for p in done]
    pf_ = f"{ROOT}/periods/{M}.json"
    def _plays_(p):
        if not os.path.exists(pf_): return True
        a, b = p["start_s"], p["start_s"] + p["dur_s"]; return any(x < b and a < y for x, y in json.load(open(pf_))["periods_s"])
    play_pieces = [p for p in plan_ if _plays_(p)]; done_play = [p for p in done if _plays_(p)]
    log(f"join: {len(done_play)}/{len(play_pieces)} pieces with play present")
    if len(done_play) < 0.8 * len(play_pieces): save(); raise SystemExit("too many pieces missing")
    pieces = [pickle.load(open(f"{cdir(p)}/{FM.PIECE_FILE}", "rb")) for p in done]
    per, H, cands, meta = FM.join(pieces, done, n, fps); del pieces
    # overrides/<match>_teams.json and _extras.json (K3c / K4) are keyed by the track ids of the Modal run they were made from;
    # a Kaggle re-track makes new ids, so they would hit the wrong tracks. Off unless USE_OVERRIDE=1 (pieces copied from that run).
    _ov = f"{ROOT}/overrides/{M}_teams.json"; _ex = f"{ROOT}/overrides/{M}_extras.json"; team_override = None
    for slug in [x for x in "{{SOURCES_DIRS}}".split(",") if x]:                  # a relabel kernel's override: keyed by the joined (pre-clean) ids, applied here
        for d in glob.glob(f"/kaggle/input/*/*/{slug}") + glob.glob(f"/kaggle/input/*/{slug}") + glob.glob(f"/kaggle/input/{slug}"):
            rj = f"{d}/result.json"
            if os.path.exists(f"{d}/root/overrides/{M}_teams.json") and os.path.exists(rj) and json.load(open(rj)).get("stage") == "relabel":   # only a relabel kernel's file (a piece kernel carries the repo's stale, old-id override)
                rl = json.load(open(f"{d}/root/overrides/{M}_teams.json")); nch = FM.apply_team_override(per, rl, fps, from_k=True)
                log(f"teams: relabel from {slug} ({len(rl)} tracks) changed {nch} rows before track cleaning"); _ov = "/nonexistent"; _ex = "/nonexistent"
    if "{{USE_OVERRIDE}}" == "1":
        team_override = json.load(open(_ov)) if os.path.exists(_ov) else None
        if os.path.exists(_ex): log(f"players: {FM.add_extra_rows(per, json.load(open(_ex)), fps)} extra rows (K4)")
        if team_override: log(f"teams: offline override ({len(team_override)} tracks) applied after track cleaning")
    elif os.path.exists(_ov) or os.path.exists(_ex): log("teams: the offline override / extras of the Modal run are NOT applied (new track ids); the live K3c kit rule decides")
    play_mask, periods = None, None; pf = f"{ROOT}/periods/{M}.json"
    if os.path.exists(pf):
        spec = json.load(open(pf)); per, H, cands, play_mask, periods = FM.apply_periods(per, H, cands, spec["periods_s"], fps, meta["L"], meta["W"])
        log(f"periods: {[(p['t_start'], p['t_end']) for p in periods]} s")
    # a clip (<match>_s<start>) inherits its match's trust_all_calibration (AIK: the per-frame line check rejects ~90% of frames)
    import re as _re
    base_m = _re.sub(r"_s\d+$", "", M); pf_base = f"{ROOT}/periods/{base_m}.json"
    trust_all = (os.path.exists(pf) and json.load(open(pf)).get("trust_all_calibration", False)) or (os.path.exists(pf_base) and json.load(open(pf_base)).get("trust_all_calibration", False))
    if trust_all: log("calibration: per-frame check skipped for this match (periods file: trust_all_calibration)")
    from ipanema import linecal as LC
    _mf = lambda p: f"{cdir(p)}/calib_ok_v3.npy"
    entries = [] if trust_all else [(p["i"], p["offset"], np.load(_mf(p)) if os.path.exists(_mf(p)) else None, bool(LC.find_rows(ROOT, FM.piece_id(M, p["i"])))) for p in done]
    trusted, missing = FM.trusted_frames(n, entries)
    base = play_mask if play_mask is not None else np.ones(n, bool)
    log(f"calibration: {100 * (trusted & base).sum() / max(1, base.sum()):.1f}% of match frames placed confidently")
    per, cands = FM.apply_unknown(per, cands, trusted); play_mask = base & trusted
    ctx = {"match_id": M, "video": full, "vi": {"n": n, "fps": fps, "width": meta["width"], "height": meta["height"]}, "H": H, "L": meta["L"], "W": meta["W"],
           "cal": {"coverage": meta["coverage"], "frozen": meta["frozen"], "unsure": meta.get("unsure", set())}, "tm": types.SimpleNamespace(dark_share=meta["dark_share"], strips=meta["strips"]),
           "per": per, "fps": fps, "cands": cands, "t0": t0, "play_mask": play_mask, "periods": periods, "team_override": team_override,
           "picker_gt": [p for p in (f"{ROOT}/reference/{M}/ball_gt.json",) if os.path.exists(p)]}
    gt = {}
    for d in glob.glob(f"{ROOT}/reference/{M}_s*/ball_gt.json"):
        start = int(d.split("_s")[-1].split("/")[0]); tmp = f"/tmp/gt_{start}.json"; FM.remap_gt(d, start, fps, tmp); gt.update(json.load(open(tmp)))
    # the match's own by-eye key grades the ball (Modal never did: there the key could be WASB training labels; on Kaggle the
    # weights are pinned and never trained on it, see the manifest)
    gt_path = FM.ball_key_path(ROOT, M)
    if gt: gt_path = "/tmp/gt_full.json"; json.dump(gt, open(gt_path, "w")); log(f"ball check: {len(gt)} held-out test frames mapped into the full match")
    summary, folder, _ = analyse(ctx, S, log=log, export_kw={"frame_stride": 3, "split_s": 300, "copy_video": False, "make_zip": False, "video_url": f"{R2}/{M}/video.mp4"}, gt_path=gt_path or "/nonexistent")
    safe = json.loads(json.dumps(summary, default=lambda o: o.item() if hasattr(o, "item") else str(o))); result["summary"] = safe
    os.makedirs(f"{W}/export", exist_ok=True); shutil.copytree(f"{ROOT}/runs/matches/{M}", f"{W}/export/{M}", dirs_exist_ok=True)
    if os.path.exists(f"{ROOT}/runs/library.json"): shutil.copy(f"{ROOT}/runs/library.json", f"{W}/export/library.json")
    for p in glob.glob(f"/content/ipanema-analysis/results/debug/ballcheck_{M}/*.jpg"): shutil.copy(p, W)
    log("export: " + ", ".join(sorted(os.listdir(f"{W}/export/{M}")))); os.remove(full)
    if STAGE != "all": shutil.rmtree(f"{ROOT}/cache", ignore_errors=True)         # a one-kernel clip run keeps its pieces so a join can be re-run on them
save(); log(f"done: {result.get('stage')} in {(time.time() - t0) / 60:.1f} min")
