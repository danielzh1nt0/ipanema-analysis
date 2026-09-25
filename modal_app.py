"""Ipanema on Modal: one match per call on an NVIDIA L4. Video from R2, code from GitHub, models/labels/caches on a Volume,
results to Supabase + R2 like the Colab pipeline. Submit with:  modal run modal_app.py --match-id SFKBP1109_s1200"""
import modal, os

APP = "ipanema"; REPO = "https://github.com/danielzh1nt0/ipanema-analysis.git"; ROOT = "/data/match_analysis"
vol = modal.Volume.from_name("ipanema-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "git", "wget", "libgl1", "libglib2.0-0")
         .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
         .pip_install("ultralytics==8.3.40", "supervision==0.25.1", "opencv-python-headless", "numpy<2", "pandas", "scipy", "scikit-learn", "umap-learn", "transformers==4.46.3", "timm==1.0.11", "huggingface_hub<1.0", "pillow", "tqdm", "boto3", "supabase", "requests", "boxmot")
         .pip_install("gdown", "pyyaml", "omegaconf")
         .run_commands("git clone -q --depth 1 https://github.com/nttcom/WASB-SBDT.git /content/WASB-SBDT",
                       "git clone -q --depth 1 https://github.com/mguti97/PnLCalib.git /content/PnLCalib")
         .pip_install("lsq-ellipse==2.2.1", "shapely")
         .run_commands("git clone -q https://github.com/roboflow/sports.git /content/sports && pip install -q -e /content/sports",
                       "cd /content/sports/examples/soccer && bash setup.sh"))
app = modal.App(APP, image=image)

@app.function(gpu="L4", timeout=150 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def run_match(match_id: str, video_url: str, start_s: int = 0, dur_s: int = 0, log_tail: int = 400):
    import subprocess, sys, requests, importlib
    subprocess.run(f"rm -rf /content/ipanema-analysis && git clone -q {REPO} /content/ipanema-analysis", shell=True, check=True)
    sys.path.insert(0, "/content/ipanema-analysis")
    os.makedirs(f"{ROOT}/videos", exist_ok=True)
    src = f"{ROOT}/videos/{match_id}.mp4"
    if not os.path.exists(src):
        with requests.get(video_url, stream=True, timeout=600) as r:
            r.raise_for_status(); open(src, "wb").write(b"".join(r.iter_content(1 << 20)))
        if dur_s:   # cut a segment from a full match
            seg = f"{ROOT}/videos/{match_id}_s{start_s}.mp4"
            subprocess.run(["ffmpeg", "-y", "-ss", str(start_s), "-i", src, "-t", str(dur_s), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", seg], check=True, capture_output=True)
            src = seg; match_id = f"{match_id}_s{start_s}"
    from ipanema.config import Settings
    from ipanema.run import run
    S = Settings(root=ROOT, sports_dir="/content/sports", work="/tmp/work")
    from ipanema.progress import Progress
    prog = Progress(match_id)
    _log, lines = _logger(f"{match_id}/run_match.log")
    def log(*a):
        _log(*a)
        try: prog.feed(" ".join(str(x) for x in a))
        except Exception: pass
    prog.push()
    try: summary, folder, z = run(src, match_id=match_id, settings=S, log=log)
    except Exception as e:
        import traceback; log("RUN FAILED: " + traceback.format_exc()); summary = {"error": str(e)}
    try: prog.finished(ok="error" not in summary)
    except Exception: pass
    # a test segment of an uploaded match inherits that match's team names, so its library card reads properly
    try:
        import re as _re
        base = _re.sub(r"_s(\d+)$", "", match_id)
        if base != match_id and "error" not in summary and prog.db:
            lab = prog.db.table("match_labels").select("*").eq("match_id", base).maybe_single().execute()
            if lab and lab.data:
                row = {k: v for k, v in lab.data.items() if k not in ("match_id", "id", "created_at", "updated_at")}
                start = int(_re.search(r"_s(\d+)$", match_id).group(1)); mm = lambda s: f"{s // 60}:{s % 60:02d}"
                row["competition"] = f"5-minute test · {mm(start)}–{mm(start + 300)}" + (f" · {row['competition']}" if row.get("competition") else "")
                prog.db.table("match_labels").upsert({"match_id": match_id, **row}).execute()
    except Exception as e: log(f"label copy skipped: {e!r}")
    if os.environ.get("IPANEMA_PNL_VALIDATE", "0") == "1":
        try:
            from ipanema import pnlcalib
            pnlcalib.validate(ROOT, log=log)
        except Exception as e:
            import traceback; log("pnlcalib validation failed: " + traceback.format_exc()[-600:])
    vol.commit()
    # publish log + summary to the repo (results/modal/<match>.txt) so results can be read without the dashboard
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok: log("publish skipped: GITHUB_TOKEN not in the ipanema-storage secret (re-run colab_modal_setup.py)")
    else:
        try:
            import json, datetime
            d = "/content/ipanema-analysis/results/modal"; os.makedirs(d, exist_ok=True)
            open(f"{d}/{match_id}.txt", "w").write("\n".join(lines[-2000:]) + "\n\nSUMMARY " + json.dumps(summary, default=str))
            url = f"https://x-access-token:{tok}@github.com/danielzh1nt0/ipanema-analysis.git"
            r = subprocess.run(f"cd /content/ipanema-analysis && git config user.email modal@ipanema && git config user.name modal && git add results/modal && git commit -qm 'modal results for {match_id}' && git pull -q --rebase -X theirs {url} main && git push -q {url} HEAD:main", shell=True, capture_output=True, text=True)
            log("published to results/modal" if r.returncode == 0 else "publish failed: " + (r.stderr or r.stdout)[-300:])
        except Exception as e: log(f"publish failed: {e!r}")
    files = {}
    import glob as _g, base64
    for p in _g.glob(f"/content/ipanema-analysis/results/debug/ballcheck_{match_id}/ballcheck.jpg") + sorted(_g.glob("/content/ipanema-analysis/results/debug/pnlcalib_*.jpg"))[:5] + sorted(_g.glob(f"/content/ipanema-analysis/results/debug/{match_id}_v{__import__('ipanema').__version__}_f*.jpg"))[:4]:
        try: files[os.path.basename(os.path.dirname(p)) + "_" + os.path.basename(p)] = base64.b64encode(open(p, "rb").read()).decode()
        except Exception: pass
    import json as _json
    safe = _json.loads(_json.dumps(summary, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return {"summary": safe, "log_tail": [str(l) for l in lines[-log_tail:]], "files": files}


def _logger(relpath):
    """log to stdout, keep the lines, and stream them to logs/<relpath> on the volume (committed every 30 s)"""
    import time as _t
    path = f"{ROOT}/logs/{relpath}"; os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close(); state = {"t": 0.0}; lines = []
    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
        try:
            with open(path, "a") as fh: fh.write(s + "\n")         # never left open: Modal refuses to reload/commit a volume with open files
            if _t.time() - state["t"] > 30: state["t"] = _t.time(); vol.commit()
        except Exception: pass
    return log, lines

# ---------------- Full match: pieces in parallel on GPUs, then one analysis on CPU ----------------
def _setup():
    import subprocess, sys
    subprocess.run(f"rm -rf /content/ipanema-analysis && git clone -q {REPO} /content/ipanema-analysis", shell=True, check=True)
    for m in [k for k in sys.modules if k == "ipanema" or k.startswith("ipanema.")]: del sys.modules[m]   # a warm container keeps old code loaded (25 Sep: two runs died on it)
    if "/content/ipanema-analysis" not in sys.path: sys.path.insert(0, "/content/ipanema-analysis")
    from ipanema.config import Settings
    return Settings(root=ROOT, sports_dir="/content/sports", work="/tmp/work")

@app.function(gpu="L4", timeout=75 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def train_ball(match_id: str = "_train"):
    """make sure the fine-tuned ball model exists (trains once, before the pieces start)"""
    S = _setup()
    from ipanema import wasb
    from ipanema.wasb_train import ensure_finetuned
    log, lines = _logger(f"{match_id}/train_ball.log")
    wasb.ensure(S.root, log=log); ensure_finetuned(S.root, f"{S.root}/videos", log=log); vol.commit()
    return lines[-40:]

def _piece_body(match_id, piece, full_path):
    S = _setup()
    from ipanema.fullmatch import process_piece
    log, lines = _logger(f"{match_id}/piece_{piece['i']:03d}.log")
    vol.reload()
    try: path = process_piece(full_path, match_id, piece, S, log=log); ok = True
    except Exception:
        import traceback; log("PIECE FAILED: " + traceback.format_exc()[-1500:]); path = None; ok = False
    vol.commit()
    import glob as _g, base64
    imgs = {}
    for p in sorted(_g.glob(f"/content/ipanema-analysis/results/debug/{match_id}_c{piece['i']:03d}_v*_f*.jpg"))[:2]:
        try: imgs[os.path.basename(p)] = base64.b64encode(open(p, "rb").read()).decode()
        except Exception: pass
    return {"i": piece["i"], "ok": ok, "path": path, "log": lines[-12:], "images": imgs}

@app.function(gpu="L4", timeout=75 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], max_containers=10)   # plan limit: 10 GPUs at once
def run_piece(match_id: str, piece: dict, full_path: str):
    return _piece_body(match_id, piece, full_path)

@app.function(timeout=75 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=8.0, memory=16384, max_containers=21)
def run_piece_cpu(match_id: str, piece: dict, full_path: str):
    """pieces whose players and ball are already detected only need calibration + positions + team split: no GPU"""
    return _piece_body(match_id, piece, full_path)

@app.function(timeout=5 * 60, volumes={"/data": vol}, cpu=1.0)
def piece_inventory(match_id: str):
    """which pieces of a full match are done / can finish on CPU / need a GPU (reads the volume only; costs ~nothing)"""
    import glob
    _setup()
    from ipanema import fullmatch as FM
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    if full is None: return {"error": "full video not on the volume"}
    n, fps = FM.video_info(full); rows = []
    for p in FM.plan(n, fps):
        c = f"{ROOT}/cache/{FM.piece_id(match_id, p['i'])}"
        done = os.path.exists(f"{c}/{FM.PIECE_FILE}")
        det = (os.path.exists(f"{c}/tracks_kp.pkl") or bool(glob.glob(f"{c}/tracks_pano_*.pkl"))) and bool(glob.glob(f"{c}/ball_cands_wasb_*_t2x2.pkl"))
        tracked = False
        if os.path.exists(f"{c}/calibration_ptz.pkl"):
            try:
                import pickle as _pk; tracked = bool(_pk.load(open(f"{c}/calibration_ptz.pkl", "rb")).get("complete"))
            except Exception: tracked = False
        rows.append({"i": p["i"], "status": "done" if done else ("cpu" if det else "gpu"), "tracked": tracked})
    return {"pieces": rows, "done": sum(r["status"] == "done" for r in rows), "cpu": sum(r["status"] == "cpu" for r in rows), "gpu": sum(r["status"] == "gpu" for r in rows)}

@app.function(timeout=40 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=8.0, memory=16384)
def check_piece(match_id: str):
    """ONE piece of a full match on CPU only, as a check before the whole match. Uses a piece whose players and ball are
    already detected; if there is none it stops instead of using a GPU."""
    import glob
    _setup()
    from ipanema import fullmatch as FM
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    if full is None: return {"error": "full video not on the volume; not downloading for a check"}
    n, fps = FM.video_info(full); plan_ = FM.plan(n, fps)
    pick = None
    for i in (4, 10, 16, 0):
        if i >= len(plan_): continue
        c = f"{ROOT}/cache/{FM.piece_id(match_id, i)}"
        if (os.path.exists(f"{c}/tracks_kp.pkl") or glob.glob(f"{c}/tracks_pano_*.pkl")) and glob.glob(f"{c}/ball_cands_wasb_*_t2x2.pkl"):
            pick = plan_[i]; break
    if pick is None: return {"error": "no piece with saved detections among 4/10/16/0; stopped rather than use a GPU"}
    out = _piece_body(match_id, pick, full)
    out["piece"] = pick; out["frames_total"] = n; out["fps"] = fps
    return out

@app.function(timeout=30 * 60, volumes={"/data": vol}, cpu=2.0, memory=4096, max_containers=21)
def calib_mask_piece(match_id: str, i: int):
    """'Don't guess' check for one piece: which frames' calibration is trusted (CPU, reads saved data); saved as calib_ok.npy"""
    import pickle, numpy as np
    _setup()
    from ipanema import fullmatch as FM, calcheck as CC
    pid = FM.piece_id(match_id, i); c = f"{ROOT}/cache/{pid}"; p = pickle.load(open(f"{c}/{FM.PIECE_FILE}", "rb"))
    ok, summary = CC.confidence_mask(f"{ROOT}/videos/{pid}.mp4", p["H"], int(p["n"]), p["L"], p["W"])
    np.save(f"{c}/calib_ok_v2.npy", ok); vol.commit()
    summary["i"] = i; return summary

@app.function(timeout=150 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=8.0, memory=32768)
def run_full(match_id: str, video_url: str, log_tail: int = 500):
    import pickle, glob, types, json, time, requests
    import numpy as np
    S = _setup()
    from ipanema import fullmatch as FM
    from ipanema.run import analyse
    from ipanema.progress import Progress
    import shutil
    shutil.rmtree(f"{ROOT}/logs/{match_id}", ignore_errors=True)          # fresh live log for this run
    _log, lines = _logger(f"{match_id}/run_full.log")
    prog = Progress(match_id)                                              # live stage/percent in the app's processing screen
    def log(*a):
        _log(*a)
        try: prog.feed(" ".join(str(x) for x in a))
        except Exception: pass
    t0 = time.time()
    full = next((p for p in (f"{ROOT}/videos/{match_id}/full_cropped.mp4", f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    if full and full.endswith("full_cropped.mp4"): video_url = video_url.replace("/video.mp4", "/video_cropped.mp4")   # the app plays what we analysed
    if full is None:
        full = f"{ROOT}/videos/{match_id}/full.mp4"; os.makedirs(os.path.dirname(full), exist_ok=True)
        with requests.get(video_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(full, "wb") as f:
                for chunk in r.iter_content(8 << 20): f.write(chunk)
        vol.commit()
    n, fps = FM.video_info(full); plan_ = FM.plan(n, fps)
    log(f"=== {match_id} full match === video: {n} frames ({n / fps:.0f} s) @ {fps:.3f} fps -> {len(plan_)} pieces")
    log("calibration: from the clip's own camera")
    for line in train_ball.remote(match_id): log("  " + line)
    from ipanema.fullmatch import piece_id, PIECE_FILE
    cdir = lambda p: f"{ROOT}/cache/{piece_id(match_id, p['i'])}"
    cached = [p for p in plan_ if os.path.exists(f"{cdir(p)}/{PIECE_FILE}")]
    todo = [p for p in plan_ if p not in cached]
    detected = lambda p: (os.path.exists(f"{cdir(p)}/tracks_kp.pkl") or bool(glob.glob(f"{cdir(p)}/tracks_pano_*.pkl"))) and bool(glob.glob(f"{cdir(p)}/ball_cands_wasb_*_t2x2.pkl"))
    cpu_todo = [p for p in todo if detected(p)]; gpu_todo = [p for p in todo if not detected(p)]
    log(f"pieces: {len(cached)} already processed, {len(cpu_todo)} to finish on CPU (detections saved), {len(gpu_todo)} need a GPU")
    res = [{"i": p["i"], "ok": True, "path": f"{cdir(p)}/{PIECE_FILE}", "log": ["cached"], "images": {}} for p in cached]
    # canary: one piece first; the rest only start if it looks right
    ci = FM.canary_index(plan_, [p["i"] for p in todo])
    if ci is not None:
        cp = next(p for p in todo if p["i"] == ci); fn = run_piece_cpu if cp in cpu_todo else run_piece
        log(f"canary: piece {ci} first ({'CPU' if fn is run_piece_cpu else 'GPU'})")
        r = fn.remote(match_id, cp, full)
        ok, why = FM.canary_ok(r.get("log") or [], os.path.exists(f"/content/ipanema-analysis/calibration/{match_id}.json"))
        if not (r.get("ok") and ok):
            log(f"CANARY FAILED on piece {ci}: {why}; the other {len(todo) - 1} pieces were not started")
            for line in (r.get("log") or [])[-12:]: log("  " + line)
            return {"summary": {"error": "canary failed: " + why}, "log_tail": lines[-log_tail:], "files": r.get("images") or {}}
        log(f"canary passed: {why}"); res.append(r)
        cpu_todo = [p for p in cpu_todo if p["i"] != ci]; gpu_todo = [p for p in gpu_todo if p["i"] != ci]
    calls = []
    if cpu_todo: calls.append(run_piece_cpu.map([match_id] * len(cpu_todo), cpu_todo, [full] * len(cpu_todo), return_exceptions=True))
    if gpu_todo: calls.append(run_piece.map([match_id] * len(gpu_todo), gpu_todo, [full] * len(gpu_todo), return_exceptions=True))
    for c in calls: res += list(c)
    ok = [r for r in res if isinstance(r, dict) and r.get("ok")]; bad = [r for r in res if not (isinstance(r, dict) and r.get("ok"))]
    log(f"pieces: {len(ok)}/{len(plan_)} done in {(time.time() - t0) / 60:.1f} min")
    log(f"tracking frame {int(n * len(ok) / max(1, len(plan_)))}")          # moves the app's progress bar to match the pieces done
    for r in bad: log(f"  piece failed: {str(r)[-600:]}")
    for r in sorted(ok, key=lambda r: r["i"]): log(f"  piece {r['i']:02d}: " + (r["log"][-1] if r["log"] else ""))
    if len(ok) < len(plan_) * 0.8: log("too many pieces failed; not analysing"); return {"summary": {"error": "pieces failed"}, "log_tail": lines[-log_tail:], "files": {}}
    # "don't guess": which frames' calibration is trusted, per piece (computed once, cached)
    vol.reload()
    need = [r["i"] for r in ok if not os.path.exists(f"{ROOT}/cache/{piece_id(match_id, r['i'])}/calib_ok_v2.npy")]
    if need:
        for s in calib_mask_piece.map([match_id] * len(need), need, return_exceptions=True):
            log(f"  calibration check piece {s.get('i') if isinstance(s, dict) else '?'}: {s}")
    vol.reload()
    done = sorted(ok, key=lambda r: r["i"]); pieces = [pickle.load(open(r["path"], "rb")) for r in done]; pl = [plan_[r["i"]] for r in done]
    per, H, cands, meta = FM.join(pieces, pl, n, fps); del pieces
    play_mask, periods = None, None
    pf = next((p for p in (f"{ROOT}/periods/{match_id}.json", f"/content/ipanema-analysis/periods/{match_id}.json") if os.path.exists(p)), f"/content/ipanema-analysis/periods/{match_id}.json")
    if os.path.exists(pf):
        spec = json.load(open(pf))
        per, H, cands, play_mask, periods = FM.apply_periods(per, H, cands, spec["periods_s"], fps, meta["L"], meta["W"])
        log(f"periods: {[(p['t_start'], p['t_end']) for p in periods]} s; {int(play_mask.sum() / fps / 60)} min of match time kept, later halves mirrored so each team attacks the same way")
    else: log("periods: none set, the whole recording counts as match time")
    trusted = np.ones(n, bool); missing = []
    for r, p in zip(done, pl):
        f = f"{ROOT}/cache/{piece_id(match_id, r['i'])}/calib_ok_v2.npy"
        if not os.path.exists(f): missing.append(r["i"]); continue
        m_ = np.load(f); a = p["offset"]; b = min(n, a + len(m_)); trusted[a:b] = m_[:b - a]
    if missing: log(f"calibration check missing for pieces {missing}: their frames are trusted as before")
    base = play_mask if play_mask is not None else np.ones(n, bool)
    log(f"calibration: {100 * (trusted & base).sum() / max(1, base.sum()):.1f}% of match frames placed confidently; the rest are marked unknown and excluded")
    per, cands = FM.apply_unknown(per, cands, trusted); play_mask = base & trusted
    ctx = {"match_id": match_id, "video": full, "vi": {"n": n, "fps": fps, "width": meta["width"], "height": meta["height"]}, "H": H, "L": meta["L"], "W": meta["W"],
           "cal": {"coverage": meta["coverage"], "frozen": meta["frozen"]}, "tm": types.SimpleNamespace(dark_share=meta["dark_share"], strips=meta["strips"]),
           "per": per, "fps": fps, "cands": cands, "t0": t0, "play_mask": play_mask, "periods": periods,
           "picker_gt": [p for p in (f"{ROOT}/reference/{match_id}/ball_gt.json", f"{ROOT}/reference/{match_id}b/ball_gt.json") if os.path.exists(p)]}
    # honest ball score: the held-out test frames of this match's segments, mapped into the full timeline (never the training labels)
    gt = {}
    for d in glob.glob(f"{ROOT}/reference/{match_id}_s*/ball_gt.json"):
        start = int(d.split("_s")[-1].split("/")[0]); tmp = f"/tmp/gt_{start}.json"; FM.remap_gt(d, start, fps, tmp); gt.update(json.load(open(tmp)))
    gt_path = None
    if gt: gt_path = "/tmp/gt_full.json"; json.dump(gt, open(gt_path, "w")); log(f"ball check: {len(gt)} held-out test frames mapped into the full match")
    summary, folder, _ = analyse(ctx, S, log=log, export_kw={"frame_stride": 3, "split_s": 300, "copy_video": False, "make_zip": False, "video_url": video_url}, gt_path=gt_path or "/nonexistent")
    # the full match inherits team names from one of its labelled segments
    try:
        from supabase import create_client
        db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        if not (db.table("match_labels").select("match_id").eq("match_id", match_id).execute().data or []):
            src = db.table("match_labels").select("*").like("match_id", f"{match_id}_%").limit(1).execute().data or []
            if src:
                row = {k: v for k, v in src[0].items() if k not in ("match_id", "id", "created_at", "updated_at")}; row["competition"] = "Full match"
                db.table("match_labels").upsert({"match_id": match_id, **row}).execute(); log("labels copied from " + src[0]["match_id"])
    except Exception as e: log(f"label copy skipped: {e!r}")
    qa = []; verdicts = {}
    def check(name, ok_, detail, stat=None):
        qa.append(f"QA {'PASS' if ok_ else 'FAIL'}: {name} ({detail})")
        if stat: verdicts[stat] = {"ok": bool(ok_), "detail": detail}
    ppf = summary.get("players_per_frame_median") or {}
    a_, b_ = float(ppf.get("A", 0)), float(ppf.get("B", 0))
    check("pieces processed", len(ok) == len(plan_), f"{len(ok)}/{len(plan_)}", "pieces")
    check("calibration", (summary.get("calibration_coverage") or 0) > 0.95, f"coverage {summary.get('calibration_coverage')}", "positions")
    check("players per frame", a_ + b_ >= 14, f"A {a_} + B {b_}", "players")
    check("team balance", min(a_, b_) >= 0.5 * max(a_, b_) if max(a_, b_) else False, f"A {a_} vs B {b_}", "teams")
    check("shots and goals from Veo", (summary.get("shots") or {}) != {} and sum((summary.get("goals") or {}).values()) > 0, f"shots {summary.get('shots')}, goals {summary.get('goals')}", "shots")
    check("match time", (summary.get("match_seconds") or 0) > 0.5 * n / fps, f"{summary.get('match_seconds')} s of {n / fps:.0f} s", "match_time")
    grade = (summary.get("ball_grade") or {})
    verdicts["ball"] = {"ok": bool(grade.get("possession_ok")), "detail": f"accuracy {grade.get('accuracy')}, near a player {grade.get('near_player_pct')}"}
    verdicts["possession"] = {"ok": bool(grade.get("possession_ok")), "detail": "unlocked" if grade.get("possession_ok") else "withheld: ball not reliable enough"}
    summary["quality"] = verdicts                                           # travels with the match so the app can show what to trust
    for q in qa: log(q)
    vol.commit()
    import json as _json
    safe = _json.loads(_json.dumps(summary, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    safe["qa"] = qa
    log("SUMMARY " + _json.dumps(safe))                                   # also in the volume log, so a result can be collected later
    files = {}
    import glob as _g, base64
    for r in ok: files.update(r.get("images") or {})
    for p in _g.glob(f"/content/ipanema-analysis/results/debug/ballcheck_{match_id}/ballcheck.jpg"):
        try: files[os.path.basename(p)] = base64.b64encode(open(p, "rb").read()).decode()
        except Exception: pass
    return {"summary": safe, "log_tail": [str(l) for l in lines[-log_tail:]], "files": files}


@app.function(timeout=20 * 60, volumes={"/data": vol}, cpu=2.0, memory=8192)
def export_picker_data(match_id: str):
    """ball candidates + player positions + calibration + labels of one clip, packed small, for offline picker work (no GPU)"""
    import pickle, glob, json, io, base64, numpy as np
    _setup()
    cache = f"{ROOT}/cache/{match_id}"
    cf = sorted(glob.glob(f"{cache}/ball_cands_wasb_*_t2x2.pkl"), key=os.path.getmtime)
    if not cf: return {"error": f"no tiled candidates in {cache}: {sorted(os.listdir(cache)) if os.path.isdir(cache) else 'missing'}"}
    cands = pickle.load(open(cf[-1], "rb"))
    per, fps = pickle.load(open(f"{cache}/tracks_pano_direct-v3.pkl", "rb"))
    Hm = pickle.load(open(sorted(glob.glob(f"{cache}/calibration_pano_*.pkl"))[-1], "rb"))
    cr = [(k, x, y, c) for k, v in cands.items() for x, y, c in v]
    pr = [(k, r[0], 0 if r[1] == "A" else 1, r[2][0], r[2][1], r[3][0], r[3][1], int(bool(r[5]))) for k, rows in per.items() for r in rows]
    hk = sorted(Hm)
    gt_path = f"{ROOT}/reference/{match_id}/ball_gt.json"; gt = json.load(open(gt_path)) if os.path.exists(gt_path) else {}
    buf = io.BytesIO()
    np.savez_compressed(buf, cands=np.array(cr, np.float32).reshape(-1, 4), players=np.array(pr, np.float32).reshape(-1, 8),
                        h_frames=np.array(hk, np.int32), h=np.stack([np.asarray(Hm[k], np.float32) for k in hk]), fps=np.float32(fps), n=np.int32(len(per)),
                        gt=np.frombuffer(json.dumps(gt).encode(), np.uint8), source=np.frombuffer(os.path.basename(cf[-1]).encode(), np.uint8))
    return {"npz": base64.b64encode(buf.getvalue()).decode(), "frames": len(per), "candidates": len(cr), "file": os.path.basename(cf[-1])}


@app.function(timeout=20 * 60, volumes={"/data": vol}, cpu=2.0, memory=4096, max_containers=21)
def calib_report_piece(match_id: str, i: int):
    """one piece: line-fit score over ~60 frames, off-pitch rate, players per frame, one drawn frame (CPU, reads saved data)"""
    import pickle, base64
    _setup()
    from ipanema import fullmatch as FM, calcheck as CC
    pid = FM.piece_id(match_id, i); p = pickle.load(open(f"{ROOT}/cache/{pid}/{FM.PIECE_FILE}", "rb"))
    out, jpg = CC.report_piece(f"{ROOT}/videos/{pid}.mp4", p["H"], p["per"], p["L"], p["W"])
    out["i"] = i
    return out, (base64.b64encode(jpg).decode() if jpg else None)

@app.function(timeout=25 * 60, volumes={"/data": vol}, cpu=1.0)
def calib_report(match_id: str):
    _setup()
    from ipanema import fullmatch as FM
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    n, fps = FM.video_info(full); plan_ = FM.plan(n, fps)
    res = list(calib_report_piece.map([match_id] * len(plan_), [p["i"] for p in plan_], return_exceptions=True))
    rows, imgs = [], {}
    for p, r in zip(plan_, res):
        if isinstance(r, tuple):
            rows.append(r[0])
            if r[1]: imgs[f"piece_{p['i']:02d}.jpg"] = r[1]
        else: rows.append({"i": p["i"], "error": str(r)[:300]})
    return {"rows": rows, "images": imgs}

@app.function(timeout=20 * 60, volumes={"/data": vol}, cpu=2.0, memory=4096, max_containers=21)
def snap_preview_piece(match_id: str, i: int, L: float, W: float):
    import pickle, base64, numpy as np
    _setup()
    from ipanema import fullmatch as FM, calcheck as CC
    pid = FM.piece_id(match_id, i); p = pickle.load(open(f"{ROOT}/cache/{pid}/{FM.PIECE_FILE}", "rb"))
    to_model = np.array([[1, 0, (p["L"] - L) / 2], [0, 1, (p["W"] - W) / 2], [0, 0, 1.0]])     # real-pitch coords -> the calibration's coords (centres aligned)
    res = CC.snap_preview(f"{ROOT}/videos/{pid}.mp4", p["H"], L, W, to_model=to_model)
    return [{"i": i, "k": r["k"], "kind": r["kind"], "info": r["info"], "jpg": base64.b64encode(r["jpg"]).decode()} for r in res]

@app.function(timeout=25 * 60, volumes={"/data": vol}, cpu=1.0)
def snap_preview_match(match_id: str, pieces: list, L: float, W: float):
    _setup()
    out = []
    for r in snap_preview_piece.map([match_id] * len(pieces), pieces, [L] * len(pieces), [W] * len(pieces), return_exceptions=True):
        if isinstance(r, list): out += r
        else: out.append({"error": str(r)[:300]})
    return out

@app.function(timeout=45 * 60, volumes={"/data": vol}, cpu=8.0, memory=16384)
def build_full_panorama(match_id: str, every_s: float = 5.0):
    """Grow the verified panorama with frames from the whole match (CPU). Returns the extended map at full resolution,
    a copy with the current calibration drawn on it, and the transform from the old map to the new one."""
    import json, base64, numpy as np, cv2
    _setup()
    from ipanema import fullmatch as FM, panorama as PX, calcheck as CC
    log, lines = _logger(f"{match_id}/panorama.log")
    spec = json.load(open(f"/content/ipanema-analysis/calibration/{match_id}.json"))
    seed = cv2.imread(f"/content/ipanema-analysis/{spec['mosaic']}")
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    n, fps = FM.video_info(full); plan_ = FM.plan(n, fps); keys = []
    for p in plan_:
        pv = f"{ROOT}/videos/{FM.piece_id(match_id, p['i'])}.mp4"
        m = int(p["dur_s"] * fps); keys += [(pv, k) for k in range(0, m, int(every_s * fps))]
    caps = {}
    def get_frame(key):
        pv, k = key
        cap = caps.get(pv) or caps.setdefault(pv, cv2.VideoCapture(pv))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); return f if ok else None
    log(f"panorama: {len(keys)} candidate frames from {len(plan_)} pieces (every {every_s:.0f} s)")
    canvas, covered, T, placed = PX.extend(seed, keys, get_frame, pad=(3000, 300, 3000, 1800), log=log)
    ys, xs = np.where(covered); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1        # crop to what's covered
    canvas = canvas[y0:y1, x0:x1]; T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ T
    Hpm = T @ np.array(spec["H_pitch_to_mosaic"], float)                                                   # current calibration on the new map
    drawn = CC.draw_model(canvas.copy(), Hpm, 120.0, 70.0, (0, 0, 255), 3)
    enc = lambda im, q: base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()
    log(f"panorama: {len(placed)} frames placed; map {canvas.shape[1]}x{canvas.shape[0]} (old {seed.shape[1]}x{seed.shape[0]})")
    return {"log": lines[-40:], "T_old_to_new": T.tolist(), "H_pitch_to_mosaic_new": Hpm.tolist(), "size": [int(canvas.shape[1]), int(canvas.shape[0])],
            "placed": [[str(k[0]).split("/")[-1], int(k[1]), g] for k, H, g in placed],
            "panorama_jpg": enc(canvas, 88), "drawn_jpg": enc(cv2.resize(drawn, (2400, int(2400 * drawn.shape[0] / drawn.shape[1]))), 85)}

@app.function(timeout=30 * 60, volumes={"/data": vol}, cpu=8.0, memory=16384)
def recheck_frames(match_id: str, items: list):
    """Gate 2: place specific frames on the new full-coverage map and compare with the old calibration (CPU)."""
    import json, pickle, base64, numpy as np, cv2
    _setup()
    from ipanema import fullmatch as FM, panorama as PX, calcheck as CC
    d = json.load(open(f"/content/ipanema-analysis/results/panorama/{match_id}/panorama_full.json"))
    pano = cv2.imread(f"/content/ipanema-analysis/results/panorama/{match_id}/panorama_full.jpg"); Hpm = np.array(d["H_pitch_to_mosaic_new"])
    cf = PX.canvas_features(cv2.SIFT_create(nfeatures=40000), pano, pano.max(2) > 8); matcher = PX.canvas_matcher(cf); sift = cv2.SIFT_create(nfeatures=4000)
    L, W = 120.0, 70.0; pts = CC.model_points(L, W); out = []; pieces = {}
    for i, k in items:
        pid = FM.piece_id(match_id, i)
        if pid not in pieces: pieces[pid] = pickle.load(open(f"{ROOT}/cache/{pid}/{FM.PIECE_FILE}", "rb"))["H"]
        cap = cv2.VideoCapture(f"{ROOT}/videos/{pid}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); cap.release()
        if not ok: out.append({"i": i, "k": k, "error": "frame not readable"}); continue
        H_old = np.asarray(pieces[pid][k], float); s_old = CC.score_frame(f, H_old, L, W, pts)
        Hfc, why = PX.register(PX.frame_features(sift, f), cf, f.shape, matcher=matcher)
        row = {"i": i, "k": k, "old_p80": s_old and round(s_old["p80_px"], 1), "placed": Hfc is not None, "why": why}
        img = CC.draw_model(f.copy(), H_old, L, W, (0, 220, 255), 3)
        if Hfc is not None:
            H_new = np.linalg.inv(Hfc) @ Hpm; s_new = CC.score_frame(f, H_new, L, W, pts); row["new_p80"] = s_new and round(s_new["p80_px"], 1)
            img = CC.draw_model(img, H_new, L, W, (0, 0, 255), 2)
        txt = f"piece {i} frame {k}: old p80 {row['old_p80']} -> new {row.get('new_p80', 'NOT PLACED: ' + why)}"
        cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5); cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        row["jpg"] = base64.b64encode(cv2.imencode(".jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()).decode(); out.append(row)
    return out

@app.function(timeout=15 * 60, volumes={"/data": vol}, cpu=2.0, memory=4096)
def export_frames(match_id: str, items: list):
    """clean full-resolution frames + their current calibration, for offline calibration work (CPU)"""
    import pickle, base64, numpy as np, cv2
    _setup()
    from ipanema import fullmatch as FM
    out, cache = [], {}
    for i, k in items:
        pid = FM.piece_id(match_id, i)
        if pid not in cache: cache[pid] = pickle.load(open(f"{ROOT}/cache/{pid}/{FM.PIECE_FILE}", "rb"))["H"]
        cap = cv2.VideoCapture(f"{ROOT}/videos/{pid}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); cap.release()
        if not ok: continue
        out.append({"i": i, "k": k, "H_old": np.asarray(cache[pid][k], float).tolist(),
                    "jpg": base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()).decode()})
    return out

@app.function(timeout=5 * 60, volumes={"/data": vol}, cpu=1.0)
def export_events(match_id: str, types: list):
    """events of the given types from a match's exported data (reads the volume only)"""
    import json
    md = json.load(open(f"{ROOT}/runs/matches/{match_id}/match_data.json"))
    return [{"t": e.get("t"), "type": e.get("type"), "team": e.get("team"), "title": e.get("title"), "payload": {k: v for k, v in (e.get("payload") or {}).items() if isinstance(v, (int, float, str, bool))}} for e in md.get("events", []) if e.get("type") in types]

@app.function(timeout=20 * 60, volumes={"/data": vol}, cpu=4.0, memory=16384)
def detect_periods(match_id: str):
    """match periods from the saved pieces (CPU), compared with the coach's periods and Veo's goal times"""
    import pickle, json, re, numpy as np
    _setup()
    from ipanema import fullmatch as FM, periods as PD
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    n, fps = FM.video_info(full); per = {}; trusted = np.ones(n, bool); L = W = None
    for p in FM.plan(n, fps):
        c = f"{ROOT}/cache/{FM.piece_id(match_id, p['i'])}"; pk = pickle.load(open(f"{c}/{FM.PIECE_FILE}", "rb")); L, W = pk["L"], pk["W"]
        for k, rows in pk["per"].items():
            g = p["offset"] + k
            if g < n: per[g] = [[r[0], r[1], np.asarray(r[2], float)] for r in rows]
        if os.path.exists(f"{c}/calib_ok_v2.npy"):
            m = np.load(f"{c}/calib_ok_v2.npy"); a = p["offset"]; b = min(n, a + len(m)); trusted[a:b] = m[:b - a]
        del pk
    ps = PD.per_second(per, fps, L, W, trusted); r = PD.detect(ps)
    out = {"detected": r, "per_second": {k: [round(float(v), 2) for v in a] for k, a in ps.items()}}   # for offline work on the detector
    pf = f"/content/ipanema-analysis/periods/{match_id}.json"
    if os.path.exists(pf): out["coach"] = json.load(open(pf))["periods_s"]
    hf = next((p for p in (f"{ROOT}/reference/veo_highlights_{match_id}.txt", f"/content/ipanema-analysis/reference/veo_highlights_{match_id}.txt") if os.path.exists(p)), ""); ef = f"/content/ipanema-analysis/reference/veo_events_{match_id}.txt"
    if os.path.exists(hf) and os.path.exists(ef) and r:
        goals = [int(l.split()[0]) for l in open(hf) if l.strip() and not l.startswith("#") and l.split()[1] == "goal"]
        mins = [int(l.split()[0]) for l in open(ef) if re.match(r"^\d+ \S+ Goal \d", l)]
        out["veo_second_half_kickoff_s"] = PD.veo_second_half_kickoff(goals, mins, r["periods"][0][1])
    return out

@app.function(timeout=3 * 60, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=0.5)
def recent_matches(limit: int = 8):
    """newest match rows (to find an upload without asking the coach for its id)"""
    from supabase import create_client
    db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    rows = db.table("matches").select("*").order("created_at", desc=True).limit(limit).execute().data or []
    keep = ("id", "status", "created_at", "updated_at", "title", "opponent", "match_date", "duration_s")
    out = []
    for r in rows:
        o = {k: r.get(k) for k in keep if k in r}
        s = r.get("summary") or {}
        if isinstance(s, dict): o["summary_keys"] = sorted(s)[:12]; o["progress"] = s.get("progress")
        o["columns"] = sorted(r)[:30]; out.append(o)
    return out

@app.function(timeout=60 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=4.0, memory=16384)
def test_piece(match_id: str, i: int, video_url: str):
    """One 5-minute piece end to end (GPU via run_piece), then pictures: pitch lines + every detected player on 3 frames,
    and the numbers that decide whether the rest is worth running: players per frame, team split, off-pitch rate."""
    import pickle, base64, requests, numpy as np, cv2
    S = _setup()
    from ipanema import fullmatch as FM
    from ipanema.calibration import draw_model
    log, lines = _logger(f"{match_id}/test_piece.log")
    full = f"{ROOT}/videos/{match_id}/full.mp4"
    if os.path.exists(f"{ROOT}/videos/{match_id}/full_cropped.mp4"): full = f"{ROOT}/videos/{match_id}/full_cropped.mp4"   # panorama: Veo player area only
    elif not os.path.exists(full):
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with requests.get(video_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(full, "wb") as f:
                for chunk in r.iter_content(8 << 20): f.write(chunk)
        vol.commit()
    n, fps = FM.video_info(full); plan_ = FM.plan(n, fps); log(f"clip: {n} frames @ {fps:.2f} fps ({n / fps / 60:.1f} min) -> {len(plan_)} pieces; testing piece {i} ({os.path.basename(full)})")
    pc = f"{ROOT}/cache/{FM.piece_id(match_id, i)}"
    for stale in ("tracks_cyl.pkl", FM.PIECE_FILE):                         # a test always re-tracks (keeps the verified calibration)
        if os.path.exists(f"{pc}/{stale}"): os.remove(f"{pc}/{stale}"); log(f"cleared {stale}")
    vol.commit()
    res = run_piece.remote(match_id, plan_[i], full)
    for l in res.get("log", []): log("  " + l)
    if not res.get("ok"): return {"ok": False, "log": lines[-60:]}
    vol.reload(); p = pickle.load(open(res["path"], "rb")); per, H, L, W = p["per"], p["H"], p["L"], p["W"]
    counts = [len(per[k]) for k in sorted(per)]; dets = [r for k in per for r in per[k]]
    off = sum(1 for r in dets if not (0 <= r[2][0] <= L and 0 <= r[2][1] <= W))
    teamA = [sum(1 for r in per[k] if r[1] == "A") for k in sorted(per)]; teamB = [sum(1 for r in per[k] if r[1] == "B") for k in sorted(per)]
    stats = {"frames": len(per), "players_per_frame_median": float(np.median(counts)), "frames_with_18_plus_pct": round(100 * float(np.mean(np.array(counts) >= 18)), 1),
             "team_A_median": float(np.median(teamA)), "team_B_median": float(np.median(teamB)), "off_pitch_pct": round(100 * off / max(1, len(dets)), 2),
             "pitch": [L, W], "camera": H[0].as_dict() if hasattr(H[0], "as_dict") else None}
    log(f"TEST RESULT: {stats}")
    imgs = {}; pv = f"{ROOT}/videos/{FM.piece_id(match_id, i)}.mp4"; cap = cv2.VideoCapture(pv)
    for frac in (0.2, 0.5, 0.8):
        k = int(len(per) * frac); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
        if not ok: continue
        draw_model(fr, H[k], L, W, (0, 0, 255))
        for r in per.get(k, []):
            fx, fy = map(int, r[3]); col = (40, 40, 40) if r[1] == "A" else (255, 255, 255)
            cv2.circle(fr, (fx, fy), 9, (0, 255, 255), 3); cv2.circle(fr, (fx, fy), 5, col, -1)
        cv2.putText(fr, f"frame {k}: {len(per.get(k, []))} players", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
        imgs[f"test_piece{i}_f{k:05d}.jpg"] = base64.b64encode(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
    cap.release()
    return {"ok": True, "stats": stats, "log": lines[-60:], "images": imgs}

@app.function(gpu="L4", timeout=25 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def profile_tracking(match_id: str, i: int, n_frames: int = 300):
    """detector size vs speed vs players ON THE PITCH (via the verified calibration), then 300 tracked frames timed per part"""
    import json, time, numpy as np, cv2, torch
    import supervision as sv
    S = _setup()
    from ipanema import fullmatch as FM, tracking as TR, teams as T
    from ipanema.cylcam import CylCam
    from ultralytics import YOLO
    log, lines = _logger(f"{match_id}/profile.log")
    pid = FM.piece_id(match_id, i); cache = f"{ROOT}/cache/{pid}"; video = f"{ROOT}/videos/{pid}.mp4"
    cam = CylCam(json.load(open(f"{cache}/calibration_cyl.json"))["params"]); L, W = 106.0, 64.0
    cap = cv2.VideoCapture(video); frames = []
    for k in range(0, 9000, 300):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok: frames.append(f)
    cap.release(); log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}; video {frames[0].shape[1]}x{frames[0].shape[0]}; {len(frames)} test frames")
    m = YOLO(S.weights["player"])
    for sz in (1920, 2560):
        def det(f): return sv.Detections.from_ultralytics(m(f, conf=S.conf_player, verbose=False, imgsz=sz)[0]).with_nms(0.5, class_agnostic=True)
        det(frames[0]); t0 = time.time(); on = []
        for f in frames:
            d = det(f); feet = np.c_[(d.xyxy[:, 0] + d.xyxy[:, 2]) / 2, d.xyxy[:, 3]] if len(d) else np.zeros((0, 2))
            mm = cam.to_m(feet) if len(feet) else np.zeros((0, 2))
            on.append(int((np.isfinite(mm).all(1) & (mm[:, 0] >= 0) & (mm[:, 0] <= L) & (mm[:, 1] >= 0) & (mm[:, 1] <= W)).sum()))
        ms = (time.time() - t0) / len(frames) * 1000
        log(f"SIZE {sz}: {ms:.0f} ms/frame ({1000 / ms:.1f} frames/s) | players ON the pitch per frame: median {np.median(on):.0f}, 10th pct {np.percentile(on, 10):.0f}, max {max(on)}")
    os.environ["IPANEMA_MAX_FRAMES"] = str(n_frames)
    t0 = time.time(); per, fps = TR.track(video, S.weights["player"], {k: cam for k in range(9000)}, None, S.conf_player, log=log, imgsz=1920)
    log(f"PROFILE 1920: tracked {len(per)} frames in {time.time() - t0:.1f} s")
    return lines[-40:]

@app.function(gpu="L4", timeout=15 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def compare_detectors(match_id: str, i: int):
    """speed and players found for several detector settings on the same 60 frames of a panorama piece"""
    import time, numpy as np, cv2
    import supervision as sv
    S = _setup()
    from ipanema import fullmatch as FM, tracking as TR
    from ultralytics import YOLO
    log, lines = _logger(f"{match_id}/compare_detectors.log")
    m = YOLO(S.weights["player"]); log(f"model's own image size: {m.overrides.get('imgsz')}")
    cap = cv2.VideoCapture(f"{ROOT}/videos/{FM.piece_id(match_id, i)}.mp4"); frames = []
    for k in range(0, 9000, 150):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    def whole(f, sz): return sv.Detections.from_ultralytics(m(f, conf=S.conf_player, verbose=False, **({"imgsz": sz} if sz else {}))[0]).with_nms(0.5, class_agnostic=True)
    configs = [("whole frame, default size", lambda f: whole(f, None)), ("whole frame, 1920", lambda f: whole(f, 1920)),
               ("tiles, default size", lambda f: TR.detect_tiled(m, f, S.conf_player, TR.PANO_TILES)[0]),
               ("tiles, 640", lambda f: TR.detect_tiled(m, f, S.conf_player, TR.PANO_TILES, imgsz=640)[0]),
               ("tiles, 960", lambda f: TR.detect_tiled(m, f, S.conf_player, TR.PANO_TILES, imgsz=960)[0])]
    for name, fn in configs:
        fn(frames[0]); t0 = time.time(); counts = [len(fn(f)) for f in frames]; ms = (time.time() - t0) / len(frames) * 1000
        log(f"COMPARE {name:26s}: {ms:5.0f} ms/frame ({1000 / ms:4.1f} frames/s) | detections per frame median {np.median(counts):4.1f}, 10th pct {np.percentile(counts, 10):4.1f}")
    return lines[-20:]

@app.function(timeout=60 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=8.0, memory=16384)
def prepare_panorama_clip(match_id: str, video_url: str):
    """Screen recordings include the browser around Veo's player: find the player area, crop the whole clip to it once,
    upload the cropped video for the app, and clear caches made from the uncropped video (CPU)."""
    import json, glob, shutil, subprocess, base64, requests, numpy as np, cv2
    _setup()
    from ipanema.cylcam import find_video_rect
    from ipanema import fullmatch as FM
    log, lines = _logger(f"{match_id}/prepare_panorama.log")
    d = f"{ROOT}/videos/{match_id}"; src = f"{d}/full.mp4"; dst = f"{d}/full_cropped.mp4"; os.makedirs(d, exist_ok=True)
    if not os.path.exists(src):
        with requests.get(video_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for chunk in r.iter_content(8 << 20): f.write(chunk)
    n, fps = FM.video_info(src); cap = cv2.VideoCapture(src); frames = []
    for k in np.linspace(0.05, 0.95, 9) * n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    x0, y0, x1, y1 = find_video_rect(frames); W_, H_ = x1 - x0, y1 - y0
    log(f"recording {frames[0].shape[1]}x{frames[0].shape[0]}: Veo player area x {x0}-{x1}, y {y0}-{y1} ({W_}x{H_}, aspect {W_ / H_:.3f})")
    if not (1.6 < W_ / H_ < 2.0 and W_ > 0.4 * frames[0].shape[1]):
        raise RuntimeError(f"player area looks wrong ({W_}x{H_}); not cropping")
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vf", f"crop={W_}:{H_}:{x0}:{y0}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", dst], check=True, capture_output=True)
    m, fps2 = FM.video_info(dst); log(f"cropped video: {m} frames @ {fps2:.2f} fps (original {n})")
    for p in glob.glob(f"{ROOT}/cache/{match_id}_c*"): shutil.rmtree(p, ignore_errors=True)
    for p in glob.glob(f"{ROOT}/videos/{match_id}_c*.mp4"): os.remove(p)
    log("cleared caches and pieces made from the uncropped recording")
    json.dump({"rect": [x0, y0, x1, y1], "source_size": [frames[0].shape[1], frames[0].shape[0]]}, open(f"{d}/crop.json", "w"))
    import boto3
    c = {k: os.environ[k] for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET", "R2_PUBLIC_URL")}
    r2 = boto3.client("s3", endpoint_url=f"https://{c['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", aws_access_key_id=c["R2_ACCESS_KEY"], aws_secret_access_key=c["R2_SECRET_KEY"], region_name="auto")
    r2.upload_file(dst, c["R2_BUCKET"], f"{match_id}/video_cropped.mp4", ExtraArgs={"ContentType": "video/mp4"})
    url = f"{c['R2_PUBLIC_URL'].rstrip('/')}/{match_id}/video_cropped.mp4"; log(f"cropped video uploaded: {url}")
    vol.commit()
    cap = cv2.VideoCapture(dst); cap.set(cv2.CAP_PROP_POS_FRAMES, m // 2); ok, f = cap.read(); cap.release()
    img = base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode() if ok else None
    return {"rect": [x0, y0, x1, y1], "url": url, "log": lines[-20:], "image": img}

@app.function(timeout=30 * 60, volumes={"/data": vol}, cpu=4.0, memory=16384)
def link_proof(pano_mid: str, fc_mid: str, rec_times: list, offset_s: float):
    """Proof of the follow-cam <-> panorama link at a few moments: exact same-moment frame, follow-cam calibration from the
    panorama (pitch lines drawn), and the follow-cam's ball drawn onto the panorama (CPU, reads saved data)."""
    import json, glob, pickle, base64, numpy as np, cv2
    _setup()
    from ipanema import fullmatch as FM, link as LK
    from ipanema.cylcam import CylCam
    from ipanema.calibration import draw_model
    log, lines = _logger(f"{pano_mid}/link_proof.log")
    cam = CylCam(json.load(open(f"{ROOT}/cache/{FM.piece_id(pano_mid, 2)}/calibration_cyl.json"))["params"]); L, W = 106.0, 64.0
    pv = cv2.VideoCapture(f"{ROOT}/videos/{pano_mid}/full_cropped.mp4") if pano_mid else None; pfps = pv.get(cv2.CAP_PROP_FPS) if pv else 0.0
    fcv = f"{ROOT}/videos/{fc_mid}.mp4"; fn, ffps = FM.video_info(fcv); fcap = cv2.VideoCapture(fcv); plan_ = FM.plan(fn, ffps)
    def read(cap, k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read(); return f if ok else None
    out, imgs = [], {}
    for t in rec_times:
        g = int(round((t + offset_s) * ffps)); fc = read(fcap, g)
        if fc is None: out.append({"t": t, "error": "follow-cam frame not readable"}); continue
        k0 = int(round(t * pfps)); ks = list(range(k0 - 45, k0 + 46, 3))
        n1, i1, _ = LK.best_frame(fc, [read(pv, k) for k in ks], cam, L, W)                       # coarse: +-1.5 s in steps of 0.1 s
        if i1 is None: out.append({"t": t, "error": "no panorama frame matched"}); continue
        ks2 = list(range(ks[i1] - 3, ks[i1] + 4)); frames2 = [read(pv, k) for k in ks2]
        n2, i2, H = LK.best_frame(fc, frames2, cam, L, W); kbest = ks2[i2]; pano = frames2[i2]    # fine: frame by frame
        H, info = LK.fc_to_metres(fc, pano, cam, L, W)
        row = {"t": t, "fc_frame": g, "pano_frame": kbest, "offset_s": round(g / ffps - kbest / pfps, 3), **info}
        # the ball the follow-cam detector found in this frame (top candidate), into metres and onto the panorama
        p = next(pp for pp in reversed(plan_) if pp["offset"] <= g); cf = sorted(glob.glob(f"{ROOT}/cache/{FM.piece_id(fc_mid, p['i'])}/ball_cands_wasb_*_t2x2.pkl"), key=os.path.getmtime)
        cands = pickle.load(open(cf[-1], "rb")).get(g - p["offset"], []) if cf else []
        fcd = fc.copy(); pad = pano.copy()
        draw_model(fcd, np.linalg.inv(H), L, W, (0, 0, 255)); draw_model(pad, cam, L, W, (0, 0, 255))
        if cands:
            bx, by, bc = max(cands, key=lambda c: c[2]); bm = cv2.perspectiveTransform(np.float32([[[bx, by]]]), H).reshape(2); bp = cam.project(bm)[0]
            row.update(ball_conf=round(float(bc), 2), ball_m=[round(float(v), 1) for v in bm])
            cv2.circle(fcd, (int(bx), int(by)), 22, (0, 255, 255), 4)
            if np.isfinite(bp).all(): cv2.circle(pad, (int(bp[0]), int(bp[1])), 14, (0, 255, 255), 3)
        a = cv2.resize(fcd, (960, 540)); b = cv2.resize(pad, (960, int(960 * pad.shape[0] / pad.shape[1])))
        both = np.vstack([a, b]); cv2.putText(both, f"t {t}s: follow-cam (top) / panorama (bottom); yellow = follow-cam ball", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        imgs[f"link_t{int(t):04d}.jpg"] = base64.b64encode(cv2.imencode(".jpg", both, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
        log(f"LINK t={t}: {row}"); out.append(row)
    return {"rows": out, "images": imgs, "log": lines[-30:]}

@app.function(gpu="L4", timeout=20 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def compare_detectors2(match_id: str, i: int):
    """which detector setting finds the small dark (SFK) players on the far side: players ON the pitch and how many in black,
    on 30 frames of the cropped panorama; plus a picture of the same frame for each setting"""
    import json, time, base64, numpy as np, cv2
    import supervision as sv
    S = _setup()
    from ipanema import fullmatch as FM, tracking as TR
    from ipanema.cylcam import CylCam
    from ultralytics import YOLO
    log, lines = _logger(f"{match_id}/compare_detectors2.log")
    pid = FM.piece_id(match_id, i); cam = CylCam(json.load(open(f"{ROOT}/cache/{pid}/calibration_cyl.json"))["params"]); L, W = 106.0, 64.0
    cap = cv2.VideoCapture(f"{ROOT}/videos/{pid}.mp4"); frames = []
    for k in range(0, 9000, 300):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    foot = YOLO(S.weights["player"]); coco = YOLO("yolov8x.pt")
    FAR = [(0.0, 0.22, 0.4, 0.50), (0.3, 0.22, 0.7, 0.50), (0.6, 0.22, 1.0, 0.50)]          # far side of the pitch, zoomed
    def fb(model, f, conf, sz, classes=None):
        r = model(f, conf=conf, verbose=False, imgsz=sz, **({"classes": classes} if classes is not None else {}))[0]; d = sv.Detections.from_ultralytics(r)
        if classes is None:                                                                   # football model: drop the ball class
            keep = [j for j in range(len(d)) if "ball" not in r.names[int(d.class_id[j])].lower()]; d = d[keep] if len(d) else d
        return d
    def far_tiles(model, f, conf, classes=None):
        h, w = f.shape[:2]; parts = []
        for x0, y0, x1, y1 in FAR:
            a, b, c, dd = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h); d = fb(model, f[b:dd, a:c], conf, 1280, classes)
            if len(d): xy = d.xyxy.copy(); xy[:, [0, 2]] += a; xy[:, [1, 3]] += b; parts.append(sv.Detections(xyxy=xy, confidence=d.confidence, class_id=np.zeros(len(d), int)))
        return parts
    def merge(ds):
        ds = [d for d in ds if len(d)]
        if not ds: return sv.Detections.empty()
        return sv.Detections(xyxy=np.vstack([d.xyxy for d in ds]), confidence=np.concatenate([d.confidence for d in ds]), class_id=np.zeros(sum(len(d) for d in ds), int)).with_nms(0.4, class_agnostic=True)
    configs = [("football model, 2560, conf 0.3 (now)", lambda f: fb(foot, f, 0.3, 2560)),
               ("football model, 2560, conf 0.1", lambda f: fb(foot, f, 0.1, 2560)),
               ("football 2560 conf 0.1 + far side zoomed", lambda f: merge([fb(foot, f, 0.1, 2560)] + far_tiles(foot, f, 0.1))),
               ("general person model, 2560, conf 0.2", lambda f: fb(coco, f, 0.2, 2560, classes=[0])),
               ("general person 2560 + far side zoomed", lambda f: merge([fb(coco, f, 0.2, 2560, [0])] + far_tiles(coco, f, 0.2, [0])))]
    imgs = {}
    for name, fn in configs:
        fn(frames[0]); t0 = time.time(); on_n, dark_n = [], []
        for idx, f in enumerate(frames):
            d = fn(f)
            if not len(d): on_n.append(0); dark_n.append(0); continue
            feet = np.c_[(d.xyxy[:, 0] + d.xyxy[:, 2]) / 2, d.xyxy[:, 3]]; mm = cam.to_m(feet)
            on = np.isfinite(mm).all(1) & (mm[:, 0] >= 0) & (mm[:, 0] <= L) & (mm[:, 1] >= 0) & (mm[:, 1] <= W)
            labs = TR.kit_labels(f, d.xyxy[on]); on_n.append(int(on.sum())); dark_n.append(sum(1 for l in labs if l == "A"))
            if idx == 15:
                v = f.copy()
                for (x0, y0, x1, y1), o, lab in zip(d.xyxy, on, TR.kit_labels(f, d.xyxy) if len(d) else []):
                    col = (0, 0, 0) if lab == "A" else ((0, 165, 255) if lab == "K" else (255, 255, 255)); cv2.rectangle(v, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255) if o else (128, 128, 128), 2)
                    cv2.circle(v, (int((x0 + x1) / 2), int(y1)), 5, col, -1)
                cv2.putText(v, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2); imgs[f"det_{len(imgs)}.jpg"] = base64.b64encode(cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
        ms = (time.time() - t0) / len(frames) * 1000
        log(f"DET {name:44s}: {ms:5.0f} ms/frame | on the pitch median {np.median(on_n):4.1f} (10th pct {np.percentile(on_n, 10):4.1f}) | in black median {np.median(dark_n):4.1f}")
    return {"log": lines[-20:], "images": imgs}

@app.function(gpu="L4", timeout=25 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def compare_player_detectors(match_id: str, i: int):
    """Which detector finds the players in black on the far side? Same 30 frames: players ON the pitch, how many in dark
    and white shirts, speed; plus one picture per option."""
    import json, time, base64, numpy as np, cv2
    import supervision as sv
    S = _setup()
    from ipanema import fullmatch as FM, tracking as TR
    from ipanema.cylcam import CylCam
    from ultralytics import YOLO
    log, lines = _logger(f"{match_id}/compare_player_detectors.log")
    pid = FM.piece_id(match_id, i); cam = CylCam(json.load(open(f"{ROOT}/cache/{pid}/calibration_cyl.json"))["params"]); L, W = 106.0, 64.0
    cap = cv2.VideoCapture(f"{ROOT}/videos/{pid}.mp4"); frames = []
    for k in list(range(0, 9000, 300)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    football = YOLO(S.weights["player"]); coco = YOLO("yolo11x.pt")
    def run(model, f, sz, conf, classes=None):
        kw = {"imgsz": sz, "conf": conf, "verbose": False}
        if classes is not None: kw["classes"] = classes
        return sv.Detections.from_ultralytics(model(f, **kw)[0]).with_nms(0.5, class_agnostic=True)
    def far_band(model, f, conf, classes=None):
        h, w = f.shape[:2]; y0, y1 = int(0.22 * h), int(0.58 * h); boxes, confs = [], []
        for a, b in ((0, 0.4), (0.3, 0.7), (0.6, 1.0)):
            x0, x1 = int(a * w), int(b * w); kw = {"imgsz": 1280, "conf": conf, "verbose": False}
            if classes is not None: kw["classes"] = classes
            d = sv.Detections.from_ultralytics(model(f[y0:y1, x0:x1], **kw)[0])
            if len(d): xy = d.xyxy.copy(); xy[:, [0, 2]] += x0; xy[:, [1, 3]] += y0; boxes.append(xy); confs.append(d.confidence)
        full = run(model, f, 1920, conf, classes)
        if len(full): boxes.append(full.xyxy); confs.append(full.confidence)
        if not boxes: return sv.Detections.empty()
        return sv.Detections(xyxy=np.vstack(boxes), confidence=np.concatenate(confs), class_id=np.zeros(sum(len(c) for c in confs), int)).with_nms(0.5, class_agnostic=True)
    options = [("football 2560, conf 0.30 (current)", lambda f: run(football, f, 2560, 0.30)),
               ("football 2560, conf 0.10", lambda f: run(football, f, 2560, 0.10)),
               ("football, far band zoomed, conf 0.15", lambda f: far_band(football, f, 0.15)),
               ("general person 2560, conf 0.15", lambda f: run(coco, f, 2560, 0.15, [0])),
               ("general person, far band zoomed, conf 0.15", lambda f: far_band(coco, f, 0.15, [0]))]
    out, imgs = [], {}
    for name, fn in options:
        fn(frames[0]); t0 = time.time(); on, dark, white = [], [], []
        for idx, f in enumerate(frames):
            d = fn(f)
            if not len(d): on.append(0); dark.append(0); white.append(0); continue
            feet = np.c_[(d.xyxy[:, 0] + d.xyxy[:, 2]) / 2, d.xyxy[:, 3]]; mm = cam.to_m(feet)
            inside = np.isfinite(mm).all(1) & (mm[:, 0] >= 0) & (mm[:, 0] <= L) & (mm[:, 1] >= 0) & (mm[:, 1] <= W)
            labs = TR.kit_labels(f, d.xyxy[inside]); on.append(int(inside.sum())); dark.append(labs.count("A")); white.append(labs.count("B"))
            if idx == 15:
                v = f.copy()
                for (x0, y0, x1, y1), lab in zip(d.xyxy[inside], labs):
                    col = (30, 30, 30) if lab == "A" else ((255, 255, 255) if lab == "B" else (0, 200, 255))
                    cv2.rectangle(v, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2); cv2.circle(v, (int((x0 + x1) / 2), int(y1)), 5, col, -1)
                cv2.putText(v, f"{name}: {int(inside.sum())} on pitch", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                imgs[f"opt{len(imgs)}.jpg"] = base64.b64encode(cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
        ms = (time.time() - t0) / len(frames) * 1000
        row = {"option": name, "ms_per_frame": round(ms), "on_pitch_median": float(np.median(on)), "dark_median": float(np.median(dark)), "white_median": float(np.median(white))}
        log(f"OPTION {row}"); out.append(row)
    return {"rows": out, "images": imgs, "log": lines[-20:]}

@app.function(gpu="L4", timeout=25 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def compare_detectors_v2(match_id: str, i: int):
    """Find the dark far-side players: detector options on 30 frames, counting players ON the pitch and how many wear
    dark vs white shirts (target: 20+ on the pitch, about half dark). Picture of the best option."""
    import json, time, base64, numpy as np, cv2
    import supervision as sv
    S = _setup()
    from ipanema import fullmatch as FM, tracking as TR
    from ipanema.cylcam import CylCam
    from ultralytics import YOLO
    log, lines = _logger(f"{match_id}/compare_detectors_v2.log")
    pid = FM.piece_id(match_id, i); cam = CylCam(json.load(open(f"{ROOT}/cache/{pid}/calibration_cyl.json"))["params"]); L, W = 106.0, 64.0
    cap = cv2.VideoCapture(f"{ROOT}/videos/{pid}.mp4"); frames = []
    for k in range(0, 9000, 300):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    foot = YOLO(S.weights["player"]); coco = YOLO("yolov8x.pt")
    FAR = [(0.00, 0.15, 0.40, 0.55), (0.30, 0.15, 0.70, 0.55), (0.60, 0.15, 1.00, 0.55)]            # the far side of the pitch
    def dets(model, f, sz, conf, person_only=False):
        r = model(f, conf=conf, verbose=False, imgsz=sz)[0]; d = sv.Detections.from_ultralytics(r)
        if person_only and len(d): d = d[d.class_id == 0]
        return d
    def far_tiles(model, f, conf, person_only=False):
        h, w = f.shape[:2]; boxes, confs = [], []
        for x0, y0, x1, y1 in FAR:
            a, b, c, e = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
            d = dets(model, f[b:e, a:c], 1280, conf, person_only)
            if len(d): xy = d.xyxy.copy(); xy[:, [0, 2]] += a; xy[:, [1, 3]] += b; boxes.append(xy); confs.append(d.confidence)
        return (np.vstack(boxes), np.concatenate(confs)) if boxes else (np.zeros((0, 4)), np.zeros(0))
    def merged(f, model, sz, conf, person_only=False):
        d = dets(model, f, sz, conf, person_only); fb, fc = far_tiles(model, f, conf, person_only)
        xy = np.vstack([d.xyxy, fb]) if len(fb) else d.xyxy; cf = np.concatenate([d.confidence, fc]) if len(fc) else d.confidence
        if not len(xy): return xy
        return sv.Detections(xyxy=xy, confidence=cf, class_id=np.zeros(len(xy), int)).with_nms(0.4, class_agnostic=True).xyxy
    options = {
        "A football 2560 conf .30 (current)": lambda f: dets(foot, f, 2560, 0.30).xyxy,
        "B football 2560 conf .12": lambda f: dets(foot, f, 2560, 0.12).xyxy,
        "C football 1920 + far-side zoom, conf .20": lambda f: merged(f, foot, 1920, 0.20),
        "D general detector 2560 conf .15": lambda f: dets(coco, f, 2560, 0.15, True).xyxy,
        "E general detector 1920 + far-side zoom, conf .15": lambda f: merged(f, coco, 1920, 0.15, True)}
    best = None
    for name, fn in options.items():
        fn(frames[0]); t0 = time.time(); on, dark, white = [], [], []
        for f in frames:
            xy = np.asarray(fn(f)).reshape(-1, 4)
            if not len(xy): on.append(0); dark.append(0); white.append(0); continue
            feet = np.c_[(xy[:, 0] + xy[:, 2]) / 2, xy[:, 3]]; mm = cam.to_m(feet)
            inside = np.isfinite(mm).all(1) & (mm[:, 0] >= 0) & (mm[:, 0] <= L) & (mm[:, 1] >= 0) & (mm[:, 1] <= W)
            labs = TR.kit_labels(f, xy[inside]) if inside.any() else []
            on.append(int(inside.sum())); dark.append(sum(1 for l in labs if l == "A")); white.append(sum(1 for l in labs if l == "B"))
        ms = (time.time() - t0) / len(frames) * 1000
        log(f"OPTION {name}: {ms:.0f} ms/frame | on the pitch median {np.median(on):.0f} (10th pct {np.percentile(on, 10):.0f}) | dark shirts median {np.median(dark):.0f}, white {np.median(white):.0f}")
        score = np.median(on) + np.median(dark)                          # more players, and the dark ones found
        if best is None or score > best[0]: best = (score, name, fn)
    f = frames[15].copy(); xy = np.asarray(best[2](f)).reshape(-1, 4); labs = TR.kit_labels(f, xy) if len(xy) else []
    for (x0, y0, x1, y1), lab in zip(xy, labs):
        c = (int((x0 + x1) / 2), int(y1)); col = {"A": (20, 20, 20), "B": (255, 255, 255), "K": (0, 140, 255)}.get(lab, (0, 0, 255))
        cv2.circle(f, c, 8, (0, 255, 255), 2); cv2.circle(f, c, 5, col, -1)
    cv2.putText(f, f"best: {best[1]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    return {"log": lines[-20:], "best": best[1], "image": base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()).decode()}

@app.function(gpu="L4", timeout=25 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def detector_options(match_id: str, i: int):
    """which detector setting finds the small, dark far-side players: players ON the pitch and how many in black, per option"""
    import json, time, base64, numpy as np, cv2
    import supervision as sv
    S = _setup()
    from ipanema import fullmatch as FM, tracking as TR
    from ipanema.cylcam import CylCam
    from ultralytics import YOLO
    log, lines = _logger(f"{match_id}/detector_options.log")
    pid = FM.piece_id(match_id, i); cam = CylCam(json.load(open(f"{ROOT}/cache/{pid}/calibration_cyl.json"))["params"]); L, W = 106.0, 64.0
    cap = cv2.VideoCapture(f"{ROOT}/videos/{pid}.mp4"); frames = []
    for k in range(0, 9000, 300):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    foot = YOLO(S.weights["player"]); coco = YOLO("yolo11x.pt")
    def run(model, f, sz, conf, classes=None):
        kw = {"imgsz": sz, "conf": conf, "verbose": False}
        if classes is not None: kw["classes"] = classes
        return sv.Detections.from_ultralytics(model(f, **kw)[0])
    def far_band(f):
        h, w = f.shape[:2]; y0, y1 = int(0.22 * h), int(0.52 * h); parts = []
        for x0, x1 in ((0, int(0.4 * w)), (int(0.3 * w), int(0.7 * w)), (int(0.6 * w), w)):
            d = run(foot, f[y0:y1, x0:x1], 1280, 0.2)
            if len(d): xy = d.xyxy.copy(); xy[:, [0, 2]] += x0; xy[:, [1, 3]] += y0; parts.append(sv.Detections(xyxy=xy, confidence=d.confidence, class_id=d.class_id))
        parts.append(run(foot, f, 1920, 0.2))
        return sv.Detections.merge(parts).with_nms(0.5, class_agnostic=True)
    options = [("football model, 2560, conf 0.30 (current)", lambda f: run(foot, f, 2560, 0.30).with_nms(0.5, class_agnostic=True)),
               ("football model, 2560, conf 0.12", lambda f: run(foot, f, 2560, 0.12).with_nms(0.5, class_agnostic=True)),
               ("football model, far side zoomed + whole frame", far_band),
               ("general person detector (yolo11x), 2560, conf 0.15", lambda f: run(coco, f, 2560, 0.15, classes=[0]).with_nms(0.5, class_agnostic=True))]
    out, imgs = [], {}
    for name, fn in options:
        fn(frames[0]); t0 = time.time(); on, dark, white = [], [], []; sample = None
        for j, f in enumerate(frames):
            d = fn(f)
            if not len(d): on.append(0); dark.append(0); white.append(0); continue
            feet = np.c_[(d.xyxy[:, 0] + d.xyxy[:, 2]) / 2, d.xyxy[:, 3]]; mm = cam.to_m(feet)
            inside = np.isfinite(mm).all(1) & (mm[:, 0] >= 0) & (mm[:, 0] <= L) & (mm[:, 1] >= 0) & (mm[:, 1] <= W)
            labs = TR.kit_labels(f, d.xyxy[inside]); on.append(int(inside.sum())); dark.append(labs.count("A")); white.append(labs.count("B"))
            if j == 15: sample = (f.copy(), d.xyxy[inside], labs)
        ms = (time.time() - t0) / len(frames) * 1000
        row = {"option": name, "ms_per_frame": round(ms), "on_pitch_median": float(np.median(on)), "on_pitch_p10": float(np.percentile(on, 10)), "black_median": float(np.median(dark)), "white_median": float(np.median(white))}
        log(f"OPTION {row}"); out.append(row)
        if sample:
            img, boxes, labs = sample
            for (x0, y0, x1, y1), lab in zip(boxes, labs):
                col = {"A": (0, 0, 0), "B": (255, 255, 255), "K": (0, 165, 255)}.get(lab, (128, 128, 128)); cx, cy = int((x0 + x1) / 2), int(y1)
                cv2.circle(img, (cx, cy), 9, (0, 255, 255), 3); cv2.circle(img, (cx, cy), 5, col, -1)
            cv2.putText(img, f"{name}: {len(boxes)} on pitch", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            imgs[f"option{len(out)}.jpg"] = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
    return {"rows": out, "images": imgs}

@app.function(gpu="L4", timeout=25 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")])
def detector_variants(match_id: str, i: int):
    """which detector setting finds the faint dark far-side players: players ON the pitch, how many dark (SFK) vs white (BP),
    speed, and one picture per setting (same frame)"""
    import json, time, base64, numpy as np, cv2
    import supervision as sv
    S = _setup()
    from ipanema import fullmatch as FM, tracking as TR
    from ipanema.cylcam import CylCam
    from ultralytics import YOLO
    log, lines = _logger(f"{match_id}/detector_variants.log")
    pid = FM.piece_id(match_id, i); cam = CylCam(json.load(open(f"{ROOT}/cache/{pid}/calibration_cyl.json"))["params"]); L, W = 106.0, 64.0
    cap = cv2.VideoCapture(f"{ROOT}/videos/{pid}.mp4"); frames = []
    for k in range(0, 9000, 300):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok: frames.append(f)
    cap.release()
    foot = YOLO(S.weights["player"]); coco = YOLO("yolo11x.pt")
    def run(model, f, sz, conf, classes=None):
        kw = {"imgsz": sz, "conf": conf, "verbose": False}
        if classes is not None: kw["classes"] = classes
        return sv.Detections.from_ultralytics(model(f, **kw)[0])
    def farband(f):
        h, w = f.shape[:2]; y0, y1 = int(0.18 * h), int(0.52 * h); out = [run(foot, f, 1920, 0.2)]
        for a, b in ((0.0, 0.38), (0.31, 0.69), (0.62, 1.0)):
            x0, x1 = int(a * w), int(b * w); d = run(foot, f[y0:y1, x0:x1], 1280, 0.2)
            if len(d): d.xyxy[:, [0, 2]] += x0; d.xyxy[:, [1, 3]] += y0; out.append(d)
        return sv.Detections.merge(out).with_nms(0.5, class_agnostic=True)
    variants = [("football model 2560, conf 0.30 (current)", lambda f: run(foot, f, 2560, 0.30)),
                ("football model 2560, conf 0.10", lambda f: run(foot, f, 2560, 0.10)),
                ("football model 1920 + zoomed far band", farband),
                ("general person model (COCO) 2560, conf 0.20", lambda f: run(coco, f, 2560, 0.20, classes=[0]))]
    out, imgs = [], {}
    for name, fn in variants:
        fn(frames[0]); t0 = time.time(); on, dark, white = [], [], []; show = None
        for j, f in enumerate(frames):
            d = fn(f).with_nms(0.5, class_agnostic=True)
            feet = np.c_[(d.xyxy[:, 0] + d.xyxy[:, 2]) / 2, d.xyxy[:, 3]] if len(d) else np.zeros((0, 2))
            mm = cam.to_m(feet) if len(feet) else np.zeros((0, 2))
            inside = np.isfinite(mm).all(1) & (mm[:, 0] >= 0) & (mm[:, 0] <= L) & (mm[:, 1] >= 0) & (mm[:, 1] <= W) if len(mm) else np.zeros(0, bool)
            labs = TR.kit_labels(f, d.xyxy[inside]) if inside.any() else []
            on.append(int(inside.sum())); dark.append(sum(1 for x in labs if x == "A")); white.append(sum(1 for x in labs if x == "B"))
            if j == 15:
                show = f.copy()
                for (fx, fy), lab in zip(feet[inside], labs):
                    col = (30, 30, 30) if lab == "A" else ((255, 255, 255) if lab == "B" else (0, 165, 255))
                    cv2.circle(show, (int(fx), int(fy)), 8, (0, 255, 255), 2); cv2.circle(show, (int(fx), int(fy)), 5, col, -1)
        ms = (time.time() - t0) / len(frames) * 1000
        row = {"variant": name, "ms_per_frame": round(ms), "on_pitch_median": float(np.median(on)), "dark_median": float(np.median(dark)), "white_median": float(np.median(white))}
        log(f"VARIANT {row}"); out.append(row)
        if show is not None:
            cv2.putText(show, f"{name}: {on[15]} on pitch ({dark[15]} dark, {white[15]} white)", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            imgs[f"variant_{len(imgs)}.jpg"] = base64.b64encode(cv2.imencode(".jpg", show, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
    return {"rows": out, "images": imgs}

@app.function(timeout=5 * 60, volumes={"/data": vol}, cpu=1.0)
def seed_clip_calibration(match_id: str, from_piece: int):
    """use a piece's verified panorama calibration as the clip's calibration (every piece then shares it)"""
    import json, shutil
    _setup()
    from ipanema import fullmatch as FM
    src = f"{ROOT}/cache/{FM.piece_id(match_id, from_piece)}/calibration_cyl.json"; dst_dir = f"{ROOT}/cache/{match_id}"
    if not os.path.exists(src): return {"error": f"no calibration at {src}"}
    os.makedirs(dst_dir, exist_ok=True); shutil.copy(src, f"{dst_dir}/calibration_cyl.json"); vol.commit()
    return {"seeded_from": from_piece, "params": json.load(open(src))["params"], "fit": json.load(open(src)).get("fit")}

@app.function(timeout=10 * 60, volumes={"/data": vol}, cpu=1.0)
def prepare_match_files(match_id: str, source_match: str, offset_s: float, periods_s: list = None):
    """put a clip's Veo shots (shifted to the clip's own clock) and its match periods on the volume, so a run needs no commit"""
    import json
    _setup()
    from ipanema import fullmatch as FM
    full = next((p for p in (f"{ROOT}/videos/{match_id}/full_cropped.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    n, fps = FM.video_info(full); dur = n / fps
    os.makedirs(f"{ROOT}/reference", exist_ok=True); os.makedirs(f"{ROOT}/periods", exist_ok=True)
    src = f"/content/ipanema-analysis/reference/veo_highlights_{source_match}.txt"; kept = []
    if os.path.exists(src):
        for line in open(src):
            if line.strip() and not line.startswith("#"):
                t, kind = line.split()[:2]
                if offset_s <= float(t) <= offset_s + dur: kept.append((round(float(t) - offset_s, 2), kind))
        open(f"{ROOT}/reference/veo_highlights_{match_id}.txt", "w").write(
            f"# Veo shots/goals of {source_match} inside this clip; the clip starts at {offset_s:.0f} s of that recording.\n" + "".join(f"{t} {k}\n" for t, k in kept))
    per = periods_s or [[0, round(dur, 1)]]
    json.dump({"periods_s": per, "note": f"clip of {source_match} from {offset_s:.0f} s", "veo_offset_s": offset_s}, open(f"{ROOT}/periods/{match_id}.json", "w"))
    vol.commit()
    return {"duration_s": round(dur, 1), "veo_clips_inside": len(kept), "periods": per}

@app.function(timeout=20 * 60, volumes={"/data": vol}, cpu=2.0, memory=4096)
def export_sample(fc_mid: str, pano_mid: str, windows: list, pano_offset_s: float, fps_out: int = 15, width: int = 960):
    """Consecutive follow-cam frames plus the same-moment panorama frames, reduced, zipped (CPU only). For offline development."""
    import io, zipfile, base64, cv2, numpy as np
    _setup()
    fc = cv2.VideoCapture(f"{ROOT}/videos/{fc_mid}.mp4"); ffps = fc.get(cv2.CAP_PROP_FPS)
    pv = cv2.VideoCapture(f"{ROOT}/videos/{pano_mid}/full_cropped.mp4"); pfps = pv.get(cv2.CAP_PROP_FPS)
    buf = io.BytesIO(); z = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED); n = 0; manifest = []
    def grab(cap, k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(k))); ok, f = cap.read()
        if not ok: return None
        return cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])))
    for (t0, t1) in windows:
        for t in np.arange(t0, t1, 1.0 / fps_out):
            f = grab(fc, t * ffps); p = grab(pv, (t - pano_offset_s) * pfps) if pv else None
            for tag, img in (("fc", f), ("pano", p)):
                if img is None: continue
                name = f"{tag}_{t:08.3f}.jpg"; z.writestr(name, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()); n += 1
            manifest.append(round(float(t), 3))
    z.writestr("manifest.json", __import__("json").dumps({"fc": fc_mid, "pano": pano_mid, "windows": windows, "fps_out": fps_out, "width": width, "pano_offset_s": pano_offset_s, "fc_fps": ffps, "pano_fps": pfps, "times": manifest}))
    z.close()
    return {"frames": n, "zip": base64.b64encode(buf.getvalue()).decode(), "mb": round(len(buf.getvalue()) / 1e6, 1)}

@app.function(timeout=90 * 60, volumes={"/data": vol}, cpu=1.0, memory=6144, max_containers=21)
def track_piece(match_id: str, i: int, step: int = 2, base_json: str = ""):
    """Follow-cam calibration for one 5-minute piece by fixed-base pan/tilt/zoom tracking (ONE CPU core; measured
    ~9.5 frames/s per core offline). Poses every `step` frames, re-localised from a single-frame line fit whenever the
    judge says the track is lost. Progress is SAVED EVERY 500 sampled frames, so a time limit never loses the work."""
    import json, pickle, base64, numpy as np, cv2, time
    _setup()
    from ipanema import fullmatch as FM, ptz, fccam as FC
    from ipanema.calcheck import judge_frame
    from ipanema.calibration import draw_model
    log, lines = _logger(f"{match_id}/track_{i:03d}.log")
    base = json.load(open(base_json or f"/content/ipanema-analysis/calibration/panorama/{match_id}_base.json")); C = base["C"]; L, W = 106.0, 64.0
    pid = FM.piece_id(match_id, i); video = f"{ROOT}/videos/{pid}.mp4"; out_path = f"{ROOT}/cache/{pid}/calibration_ptz.pkl"; os.makedirs(f"{ROOT}/cache/{pid}", exist_ok=True)
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []; k = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if k % step == 0: frames.append(cv2.resize(f, (960, 540)))
        k += 1
    cap.release(); h, w = 540, 960; t0 = time.time(); log(f"{pid}: {n} frames, tracking every {step} at 960x540 on one core")
    good_f = lambda fr, pose: judge_frame(fr, ptz.homography(pose, C, w, h), L, W)["verdict"] == "good"
    def single(fr):
        H, info = FC.fit(fr, C, L, W, tilt_range=(0.5, 30), coarse=(3.0, 1.5, 8))
        return np.array([np.radians(info["pan_deg"]), np.radians(info["tilt_deg"]), np.radians(info["roll_deg"]), info["zoom"]]) if judge_frame(fr, H, L, W)["verdict"] == "good" else None
    poses = [None] * len(frames); okv = [False] * len(frames); j = 0; relocs = 0; last_save = 0
    def save(final=False):
        S = np.diag([2.0, 2.0, 1.0]); Hs = {}; ok_mask = np.zeros(n, bool)
        for t in range(len(frames)):
            if poses[t] is None: continue
            Hf = (S @ ptz.homography(poses[t], C, w, h)).astype(np.float32)
            for d in range(step):
                if t * step + d < n: Hs[t * step + d] = Hf; ok_mask[t * step + d] = okv[t]
        pickle.dump({"H": Hs, "ok": ok_mask, "step": step, "complete": final, "sampled_done": sum(p is not None for p in poses)}, open(out_path, "wb")); vol.commit()
    while j < len(frames):
        p = single(frames[j])
        if p is None: j += 1; continue                                   # can't start here: try the next sampled frame
        relocs += 1; seg, _ = ptz.track_sequence(frames[j:], C, p, w, h, refine_lines=lambda f_, p_: FC.refine(f_, C, L, W, p_), judge=good_f, max_lost=15)
        for t, pose in enumerate(seg): poses[j + t] = pose; okv[j + t] = good_f(frames[j + t], pose)
        j += len(seg)
        if j < len(frames): j = max(j - 14, 0) + 15                       # skip the lost stretch, restart just after it
        if sum(p is not None for p in poses) - last_save >= 500:
            save(); last_save = sum(p is not None for p in poses); log(f"  {pid}: {last_save}/{len(frames)} sampled frames done, {sum(okv)} good, {time.time() - t0:.0f} s")
    save(final=True); good = sum(okv)
    log(f"{pid}: judged good {good}/{len(frames)} sampled frames ({100 * good / max(1, len(frames)):.1f}%), {relocs} (re)starts, {time.time() - t0:.0f} s")
    imgs = {}
    for frac in (0.15, 0.5, 0.85):
        t = int(len(frames) * frac)
        if poses[t] is None: continue
        vis = frames[t].copy(); draw_model(vis, ptz.homography(poses[t], C, w, h), L, W, (0, 0, 255)); cv2.putText(vis, f"frame {t * step}: {'good' if okv[t] else 'bad'}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        imgs[f"track_{pid}_f{t * step:05d}.jpg"] = base64.b64encode(cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
    return {"piece": i, "frames": n, "sampled": len(frames), "good": good, "good_pct": round(100 * good / max(1, len(frames)), 1), "restarts": relocs, "seconds": round(time.time() - t0), "images": imgs, "log": lines[-10:]}

# ---------------- Upload API (runs only when someone uploads; no GPU) ----------------
AUTH_URL = "https://savbsnvusqbogdzvkjaf.supabase.co"   # Lovable Cloud project: who is signed in
AUTH_KEY = "sb_publishable_KIrOxTM-qNYnJCfuJlcR_g_TE8is32f"                                 # its publishable key (public by design)
api_image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("fastapi[standard]", "boto3", "requests", "supabase")
PART_MB = 32

@app.function(image=api_image, secrets=[modal.Secret.from_name("ipanema-storage")], timeout=300)
@modal.asgi_app(label="ipanema-api")
def api():
    import re, time, secrets as _s, json, subprocess, requests, boto3
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from supabase import create_client
    web = FastAPI()
    web.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["ETag"])
    E = os.environ
    r2 = boto3.client("s3", endpoint_url=f"https://{E['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", aws_access_key_id=E["R2_ACCESS_KEY"], aws_secret_access_key=E["R2_SECRET_KEY"], region_name="auto")
    bucket, pub = E["R2_BUCKET"], E["R2_PUBLIC_URL"].rstrip("/")
    db = create_client(E["SUPABASE_URL"], E["SUPABASE_SERVICE_KEY"])

    def user_of(req: Request):
        auth = req.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "): raise HTTPException(401, "Sign in to upload")
        r = requests.get(f"{AUTH_URL}/auth/v1/user", headers={"Authorization": auth, "apikey": AUTH_KEY}, timeout=10)
        if r.status_code != 200: raise HTTPException(401, "Session expired — sign in again")
        return r.json()

    def slug(s):
        return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:24] or "team"

    @web.get("/health")
    def health(): return {"ok": True}

    @web.post("/upload/start")
    async def start(req: Request):
        user_of(req); body = await req.json()
        size = int(body.get("size") or 0)
        if size <= 0 or size > 12 * 1024 ** 3: raise HTTPException(400, "Video must be under 12 GB")
        date = re.sub(r"[^0-9-]", "", str(body.get("date") or time.strftime("%Y-%m-%d")))[:10]
        match_id = f"{slug(body.get('team'))}-vs-{slug(body.get('opponent'))}-{date}-{_s.token_hex(2)}"
        key = f"{match_id}/video.mp4"
        up = r2.create_multipart_upload(Bucket=bucket, Key=key, ContentType="video/mp4")
        part = PART_MB * 1024 * 1024; n = (size + part - 1) // part
        urls = [r2.generate_presigned_url("upload_part", Params={"Bucket": bucket, "Key": key, "UploadId": up["UploadId"], "PartNumber": i}, ExpiresIn=6 * 3600) for i in range(1, n + 1)]
        return {"match_id": match_id, "key": key, "upload_id": up["UploadId"], "part_size": part, "urls": urls}

    @web.post("/upload/complete")
    async def complete(req: Request):
        user = user_of(req); body = await req.json()
        key, upload_id, match_id = body["key"], body["upload_id"], body["match_id"]
        if not key.startswith(match_id + "/"): raise HTTPException(400, "Key does not belong to this match")
        parts = sorted(({"PartNumber": int(p["n"]), "ETag": p["etag"]} for p in body["parts"]), key=lambda p: p["PartNumber"])
        r2.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts})
        video_url = f"{pub}/{key}"
        duration = 0.0
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_url], capture_output=True, text=True, timeout=90)
            duration = float(out.stdout.strip() or 0)
        except Exception: pass
        m = body.get("meta") or {}
        db.table("matches").upsert({"id": match_id, "status": "processing", "duration_s": round(duration, 1), "files": {"video": video_url}}).execute()
        try:
            db.table("match_labels").upsert({"match_id": match_id, "club_team": "A", "name_a": m.get("team"), "name_b": m.get("opponent"), "opponent": m.get("opponent"),
                                             "colour_a": m.get("kit_colour"), "date": m.get("date"), "competition": m.get("competition"),
                                             "tags": [t for t in [m.get("age_group"), m.get("venue")] if t]}).execute()
        except Exception as e: print("labels:", e)
        print(json.dumps({"uploaded": match_id, "by": user.get("email"), "parts": len(parts), "duration_s": duration}))
        return {"match_id": match_id, "status": "processing", "duration_s": duration}

    @web.post("/upload/abort")
    async def abort(req: Request):
        user_of(req); body = await req.json()
        try: r2.abort_multipart_upload(Bucket=bucket, Key=body["key"], UploadId=body["upload_id"])
        except Exception: pass
        return {"ok": True}

    @web.post("/admin/r2-cors")
    async def r2_cors(req: Request):
        user_of(req)
        rules = {"CORSRules": [{"AllowedOrigins": ["*"], "AllowedMethods": ["PUT", "GET", "HEAD"], "AllowedHeaders": ["*"], "ExposeHeaders": ["ETag"], "MaxAgeSeconds": 3600}]}
        try: r2.put_bucket_cors(Bucket=bucket, CORSConfiguration=rules); return {"ok": True}
        except Exception as e: return {"ok": False, "error": str(e)[:300]}

    return web


@app.function(timeout=60 * 60, volumes={"/data": vol}, cpu=8.0, memory=16384, secrets=[modal.Secret.from_name("ipanema-storage")])
def prepare_match(match_id: str, video_url: str, start_s: int = 1200, dur_s: int = 300):
    """New match on a new ground: fetch it, cut a 5-minute test segment, stitch that segment's panorama so the ground
    can be calibrated once. Returns the panorama and three sample frames (base64 JPEG)."""
    import subprocess, sys, requests, cv2, base64
    subprocess.run(f"rm -rf /content/ipanema-analysis && git clone -q {REPO} /content/ipanema-analysis", shell=True, check=True)
    sys.path.insert(0, "/content/ipanema-analysis")
    from ipanema.mosaic import build
    if match_id == "latest":      # newest uploaded match that is still waiting for its ground setup
        from supabase import create_client
        db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        rows = db.table("matches").select("id,status,created_at,files").eq("status", "processing").order("created_at", desc=True).limit(1).execute().data
        if not rows: raise RuntimeError("no uploaded match is waiting")
        match_id = rows[0]["id"]; video_url = (rows[0].get("files") or {}).get("video") or video_url.replace("/latest/", f"/{match_id}/")
        print("latest upload:", match_id)
    vd = f"{ROOT}/videos/{match_id}"; os.makedirs(vd, exist_ok=True); full = f"{vd}/full.mp4"
    if not os.path.exists(full):
        with requests.get(video_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(full, "wb") as f:
                for chunk in r.iter_content(8 << 20): f.write(chunk)
    seg = f"{ROOT}/videos/{match_id}_s{start_s}.mp4"
    if not os.path.exists(seg):
        subprocess.run(["ffmpeg", "-y", "-ss", str(start_s), "-i", full, "-t", str(dur_s), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", seg], check=True, capture_output=True)
    cdir = f"{ROOT}/cache/{match_id}_s{start_s}"; os.makedirs(cdir, exist_ok=True)
    mos = build(seg, f"{cdir}/mosaic_seg_v3.pkl", stride=25, canvas=(4200, 1500), log=print)
    out = {"panorama.jpg": base64.b64encode(cv2.imencode(".jpg", mos["mosaic"], [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode(), "segment": os.path.basename(seg), "match_id": match_id}
    cap = cv2.VideoCapture(seg); n = int(cap.get(7))
    for q in (0.2, 0.5, 0.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * q)); ok, f = cap.read()
        if ok: out[f"frame_{int(n * q):06d}.jpg"] = base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    cap.release(); vol.commit()
    return out


@app.local_entrypoint()
def main(match_id: str, video_url: str = "", start_s: int = 0, dur_s: int = 0):
    out = run_match.remote(match_id, video_url, start_s, dur_s)
    for l in out["log_tail"]: print(l)

@app.function(timeout=25 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=2.0, memory=4096)
def export_line_frames(match_id: str = "SFKBP1109"):
    """The ~230 frames the line-model rounds need (clicked + reviewed moments), cut from the full match on the volume and
    put on R2 under {match_id}/lines/ so GitHub's free runner can fetch them. CPU only, a few minutes, cents. One-off."""
    import boto3, glob
    _setup()
    from ipanema import linerun
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    if full is None: return {"error": "full video not on the volume"}
    lab = "/tmp/lab"; log = []; linerun.cut_frames(full, lab, log=lambda m: log.append(m))
    c = {k: os.environ[k] for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET", "R2_PUBLIC_URL")}
    r2 = boto3.client("s3", endpoint_url=f"https://{c['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", aws_access_key_id=c["R2_ACCESS_KEY"], aws_secret_access_key=c["R2_SECRET_KEY"], region_name="auto")
    keys = []
    for p in sorted(glob.glob(f"{lab}/*")):
        k = f"{match_id}/lines/{os.path.basename(p)}"; r2.upload_file(p, c["R2_BUCKET"], k); keys.append(f"{c['R2_PUBLIC_URL'].rstrip('/')}/{k}")
    return {"log": log, "files": keys}

@app.function(gpu="L4", timeout=45 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=4.0, memory=16384)
def lines_round(epochs: int = 80, max_minutes: int = 25, match_id: str = "SFKBP1109"):
    """One line-model round on a GPU (~15-20 min, ~$0.30): frames from R2 (put there once by export_line_frames),
    base check, labels, training, grading against Daniel's clicks. Returns the results folder as a zip (base64);
    the weights stay on the volume. Hard stop at 45 min."""
    import subprocess, base64, io, zipfile, time, shutil
    subprocess.run("pip install -q --no-deps segmentation-models-pytorch", shell=True, check=True)
    _setup()
    from ipanema import linerun
    lab, out = "/tmp/lab", "/tmp/out"; os.makedirs(lab, exist_ok=True); pub = os.environ["R2_PUBLIC_URL"].rstrip("/")
    import requests
    for n in ("SFKBP1109_frames_s1.zip", "SFKBP1109_frames_s2.zip", "SFKBP1109_random.zip", "SFKBP1109_random.json", "SFKBP1109_random_review.json"):
        r = requests.get(f"{pub}/{match_id}/lines/{n}", stream=True, timeout=600); r.raise_for_status()
        with open(f"{lab}/{n}", "wb") as fh:
            for chunk in r.iter_content(1 << 20): fh.write(chunk)
    log_lines = []
    def log(m): log_lines.append(f"{time.strftime('%H:%M:%S')} {m}"); print(m, flush=True)
    s = linerun.run(lab, out, epochs=epochs, max_minutes=max_minutes, log=log, work="/tmp")
    open(f"{out}/log.txt", "w").write("\n".join(log_lines) + "\n")
    os.makedirs(f"{ROOT}/models/lines", exist_ok=True); shutil.copy(f"{out}/last.pt", f"{ROOT}/models/lines/last.pt"); os.remove(f"{out}/last.pt"); vol.commit()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out):
            for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), out))
    return {"summary": s, "zip_b64": base64.b64encode(buf.getvalue()).decode()}

@app.function(timeout=70 * 60, volumes={"/data": vol}, cpu=4.0, memory=8192, max_containers=12)
def calib_stretch(match_id: str, t0: float, t1: float, fps: float = 1.0):
    """one stretch of the match: line network on CPU, track + fresh placements, self-grading at Daniel's clicked moments"""
    import subprocess
    subprocess.run("pip install -q --no-deps segmentation-models-pytorch", shell=True, check=True)
    _setup()
    from ipanema import matchcal as MC
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    sol = __import__("json").load(open("/content/ipanema-analysis/calibration/panorama/SFKBP1109_lines_solution.json"))
    cps = MC.checkpoints_from_clicks("/content/ipanema-analysis"); log = []; pics = {}
    rows, grades = MC.calibrate_video(full, f"{ROOT}/models/lines/last.pt", sol["camera"], t0, t1, fps=fps, checkpoints=cps, log=lambda m: log.append(m), pictures=pics)
    import base64
    return {"rows": rows, "grades": grades, "log": log, "pictures": {str(t): base64.b64encode(b).decode() for t, b in pics.items()}}

@app.function(timeout=90 * 60, volumes={"/data": vol}, cpu=2.0, memory=4096)
def match_calibration(match_id: str = "SFKBP1109", stretches: int = 12, fps: float = 1.0):
    """the whole match, in parallel stretches (~$3, ~35 min). Returns rows, grades, summary and the 20-picture strip."""
    import base64, json as _j, cv2
    _setup()
    from ipanema import matchcal as MC
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    if full is None: return {"error": "full video not on the volume"}
    cap = cv2.VideoCapture(full); dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 29.97); cap.release()
    edges = [dur * i / stretches for i in range(stretches + 1)]
    parts = list(calib_stretch.map([match_id] * stretches, edges[:-1], edges[1:], [fps] * stretches))
    rows = sorted((r for p in parts for r in p["rows"]), key=lambda r: r["t"]); grades = sorted((g for p in parts for g in p["grades"]), key=lambda g: g["t"])
    sol = _j.load(open("/content/ipanema-analysis/calibration/panorama/SFKBP1109_lines_solution.json"))
    sheet = MC.strip(full, rows, sol["camera"], n=20); ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"rows": rows, "grades": grades, "summary": MC.summarize(rows, grades), "duration_s": dur, "strip_b64": base64.b64encode(buf.tobytes()).decode(),
            "log": [l for p in parts for l in p["log"]], "pictures": {k: v for p in parts for k, v in p.get("pictures", {}).items()}}

@app.function(gpu="L4", timeout=40 * 60, volumes={"/data": vol}, cpu=4.0, memory=16384)
def ball_hard_frames(match_id: str = "SFKBP1109", n: int = 200, fps: float = 1.0, weights: str = "", exclude_json: str = ""):
    """The frames worth clicking: run the ball detector once a second over the match and keep the moments where it is
    absent, unsure or the ball is far/small - spread over the match. Returns 960-wide JPEGs (base64) + the detector's
    own guesses so the click page can show them. One GPU pass (~5 min)."""
    import base64, cv2, numpy as np
    from ultralytics import YOLO
    S = _setup()
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    if full is None: return {"error": "full video not on the volume"}
    import json as _j
    model = YOLO(weights if weights and os.path.exists(weights) else S.weights["ball"]); cap = cv2.VideoCapture(full)
    already = {round(r["t"]) for r in _j.load(open(exclude_json))["frames"]} if exclude_json and os.path.exists(exclude_json) else set(); vfps = cap.get(cv2.CAP_PROP_FPS) or 29.97; n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(round(vfps / fps)); recs = []; k = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if k % step == 0 and round(k / vfps) not in already:
            r = model(f, conf=0.05, imgsz=1920, verbose=False)[0]; det = [(float((a + c) / 2), float((b + d) / 2), float(cf)) for (a, b, c, d), cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            det.sort(key=lambda z: -z[2]); top = det[0][2] if det else 0.0
            group = "absent" if top < 0.15 else ("unsure" if top < 0.5 else ("far" if det[0][1] < f.shape[0] * 0.38 else "easy"))
            recs.append({"t": round(k / vfps, 3), "k": k, "group": group, "top": round(top, 3), "det": [[round(x, 1), round(y, 1), round(c, 3)] for x, y, c in det[:5]]})
        k += 1
    cap.release()
    quota = {"absent": int(n * 0.35), "unsure": int(n * 0.35), "far": int(n * 0.2), "easy": n - int(n * 0.35) * 2 - int(n * 0.2)}; chosen = []
    for g_, q in quota.items():
        pool = [r for r in recs if r["group"] == g_]
        if pool: chosen += [pool[i] for i in sorted(set(np.linspace(0, len(pool) - 1, min(q, len(pool))).astype(int)))]
    chosen.sort(key=lambda r: r["t"]); cap = cv2.VideoCapture(full); out = []
    for r in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["k"]); ok, f = cap.read()
        if not ok: continue
        h, w = f.shape[:2]; s = 960 / w; img = cv2.resize(f, (960, int(round(h * s))))
        out.append(dict(r, w=w, h=h, jpg=base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()).decode()))
    cap.release()
    return {"frames": out, "counts": {g_: sum(1 for r in recs if r["group"] == g_) for g_ in quota}, "sampled": len(recs)}

@app.function(gpu="L4", timeout=60 * 60, volumes={"/data": vol}, cpu=4.0, memory=16384)
def ball_round(match_id: str = "SFKBP1109", epochs: int = 60):
    """fine-tune the ball detector on Daniel's clicks, grade old vs new on the 40 exam frames, pictures for each (~$0.50)"""
    import base64, shutil, time
    from ultralytics import YOLO
    S = _setup(); from ipanema import ballclicks as BC
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    if full is None: return {"error": "full video not on the volume"}
    log = []; L = lambda m: (log.append(f"{time.strftime('%H:%M:%S')} {m}"), print(m, flush=True))
    ds = "/tmp/ball_ds"; shutil.rmtree(ds, ignore_errors=True); BC.build_dataset(f"/content/ipanema-analysis/results/labels/{match_id}_ball_clicks.json", full, ds, log=L)
    base = S.weights["ball"]; L(f"training from {os.path.basename(base)} for {epochs} epochs")
    model = YOLO(base); model.train(data=f"{ds}/data.yaml", epochs=epochs, imgsz=1920, batch=4, lr0=0.001, freeze=4, mosaic=0.5, scale=0.3, fliplr=0.5, project="/tmp/ballft", name="run", exist_ok=True, verbose=False, patience=100)
    new_w = "/tmp/ballft/run/weights/last.pt"; pics = {}
    s = BC.grade(new_w, f"/content/ipanema-analysis/results/labels/{match_id}_ball_clicks.json", ds, log=L, pictures=pics, old_weights=base)
    os.makedirs(f"{ROOT}/models/ball", exist_ok=True); shutil.copy(new_w, f"{ROOT}/models/ball/clicks_v1.pt"); vol.commit()
    return {"summary": s, "log": log, "pictures": {k: base64.b64encode(v).decode() for k, v in pics.items()}}
