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
        rows.append({"i": p["i"], "status": "done" if done else ("cpu" if det else "gpu")})
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

@app.function(timeout=150 * 60, volumes={"/data": vol}, secrets=[modal.Secret.from_name("ipanema-storage")], cpu=8.0, memory=32768)
def run_full(match_id: str, video_url: str, log_tail: int = 500):
    import pickle, glob, types, json, time, requests
    S = _setup()
    from ipanema import fullmatch as FM
    from ipanema.run import analyse
    import shutil
    shutil.rmtree(f"{ROOT}/logs/{match_id}", ignore_errors=True)          # fresh live log for this run
    log, lines = _logger(f"{match_id}/run_full.log")
    t0 = time.time()
    full = next((p for p in (f"{ROOT}/videos/{match_id}.mp4", f"{ROOT}/videos/{match_id}/full.mp4") if os.path.exists(p)), None)
    if full is None:
        full = f"{ROOT}/videos/{match_id}/full.mp4"; os.makedirs(os.path.dirname(full), exist_ok=True)
        with requests.get(video_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(full, "wb") as f:
                for chunk in r.iter_content(8 << 20): f.write(chunk)
        vol.commit()
    n, fps = FM.video_info(full); plan_ = FM.plan(n, fps)
    log(f"=== {match_id} full match === {n} frames @ {fps:.3f} fps ({n / fps / 60:.1f} min) -> {len(plan_)} pieces")
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
    for r in bad: log(f"  piece failed: {str(r)[-600:]}")
    for r in sorted(ok, key=lambda r: r["i"]): log(f"  piece {r['i']:02d}: " + (r["log"][-1] if r["log"] else ""))
    if len(ok) < len(plan_) * 0.8: log("too many pieces failed; not analysing"); return {"summary": {"error": "pieces failed"}, "log_tail": lines[-log_tail:], "files": {}}
    vol.reload()
    done = sorted(ok, key=lambda r: r["i"]); pieces = [pickle.load(open(r["path"], "rb")) for r in done]; pl = [plan_[r["i"]] for r in done]
    per, H, cands, meta = FM.join(pieces, pl, n, fps); del pieces
    play_mask, periods = None, None
    pf = f"/content/ipanema-analysis/periods/{match_id}.json"
    if os.path.exists(pf):
        spec = json.load(open(pf))
        per, H, cands, play_mask, periods = FM.apply_periods(per, H, cands, spec["periods_s"], fps, meta["L"], meta["W"])
        log(f"periods: {[(p['t_start'], p['t_end']) for p in periods]} s; {int(play_mask.sum() / fps / 60)} min of match time kept, later halves mirrored so each team attacks the same way")
    else: log("periods: none set, the whole recording counts as match time")
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
    vol.commit()
    import json as _json
    safe = _json.loads(_json.dumps(summary, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
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
