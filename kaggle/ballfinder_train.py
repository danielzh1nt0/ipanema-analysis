# Kaggle (free T4), K1/T0 follow-up 29 Sep: fine-tune the click ball finder on thousands of labels from 4 matches.
# Inputs from R2 (copied off the Modal volume by volume_to_r2): models/ball/clicks_latest.pt (current best, 70/108),
# labels/<match>_trainset_clicks.json for Vasalund, Solheim, Spånga, Djursholm (NOT Reymersholm/Solberga), the match
# videos and the full SFK-BP video (the 108 exam frames come from it).
# Dataset exactly like modal_app.ball_dataset + ball_round (ipanema/ballclicks.py): SFK-BP click crops (640 px, no
# at-feet copies, as in the 70/108 round) + exam frames kept full-frame, + crops around other matches' labels
# (add_auto_crops, up to PER_MATCH per match). Training like ball_round: from clicks_latest.pt, 640 px, batch 16,
# lr0 0.001, freeze 4, mosaic 0.5, scale 0.3, fliplr 0.5, val off, capped by time. Grade: BC.grade on the 108 exam
# frames ('new' score, hit within 30 px, conf 0.25), same as ball_round. Variants: 'agree' = only labels where WASB
# agreed with the click finder; 'all' = any label (round 11's recipe, which had only 25 min on Modal).
# A variant is a candidate ONLY if it beats 70/108 and the re-graded clicks_latest -> /kaggle/working/candidate_<date>.pt.
# Nothing on Modal is replaced.
# Offline dry run: BF_R2=<local dir> BF_REPO=<repo> BF_OUT=<dir> BF_TMP=<dir> BF_DRY=1 BF_CLICKS=<fake clicks json>
import os, sys, json, time, shutil, subprocess, urllib.request
t0 = time.time()
R2 = os.environ.get("BF_R2") or "{{R2}}"
W = os.environ.get("BF_OUT", "/kaggle/working"); TMP = os.environ.get("BF_TMP", "/kaggle/temp"); DRY = os.environ.get("BF_DRY") == "1"
TRAIN_MIN = float(os.environ.get("BF_TRAIN_MIN", "0.3" if DRY else "95")); EPOCHS = int(os.environ.get("BF_EPOCHS", "1" if DRY else "40"))
PER_MATCH = int(os.environ.get("BF_PER_MATCH", "20" if DRY else "4000")); VARIANTS = os.environ.get("BF_VARIANTS", "agree,all").split(",")
MATCHES = os.environ.get("BF_MATCHES", "p15u-vs-vasalund-2026-09-20,p09-norrviken-vs-solheim-2026-08-30,p15u-vs-spanga-2026-09-25,p15u-vs-djursholm-2026-09-26").split(",")
assert not any(b in m for m in MATCHES for b in ("reymersholm", "solberga")), "Reymersholm/Solberga are not approved for training"
BAR = 70; os.makedirs(W, exist_ok=True); os.makedirs(TMP, exist_ok=True)
LOG = []; REPORT = {"started": time.strftime("%Y-%m-%d %H:%M"), "dry": DRY, "bar": BAR, "train_min": TRAIN_MIN, "per_match": PER_MATCH, "variants": {}}
def L(m):
    LOG.append(f"{(time.time() - t0) / 60:6.1f} min  {m}"); print(LOG[-1], flush=True); open(f"{W}/log.txt", "w").write("\n".join(LOG) + "\n")
def save(): REPORT["minutes"] = round((time.time() - t0) / 60, 1); json.dump(REPORT, open(f"{W}/summary.json", "w"), indent=1)
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); L(f"$ {c} -> {r.returncode} {(r.stdout + r.stderr)[-400:].strip()}"); return r.returncode
def fetch(key, dst):
    """R2 object -> local file; False when missing"""
    if os.path.exists(dst) and os.path.getsize(dst) > 0: return True
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True); t1 = time.time()
    try:
        if R2.startswith("http"):
            url = f"{R2.rstrip('/')}/{key}"                                  # r2.dev answers 403 to Python-urllib's user agent (29 Sep)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ipanema-kaggle)"}), timeout=120) as r, open(dst + ".part", "wb") as f: shutil.copyfileobj(r, f, 16 << 20)
            except Exception as e1:
                if subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", dst + ".part", url]).returncode: raise RuntimeError(f"urllib {e1!r}; curl failed too")
            os.replace(dst + ".part", dst)
        else: shutil.copy(os.path.join(R2, key), dst)
    except Exception as e:
        L(f"missing on R2: {key} ({e!r})"[:300]); return False
    L(f"fetched {key}: {os.path.getsize(dst) / 1e6:.0f} MB in {time.time() - t1:.0f} s"); return True

# ---- code + packages
REPO = os.environ.get("BF_REPO")
if not REPO:
    REPO = f"{TMP}/ia"; sh(f"rm -rf {REPO} && git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}")
    sh("pip install -q ultralytics==8.3.40"); sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sys.path.insert(0, REPO)
from ipanema import ballclicks as BC
from ultralytics import YOLO
sh(f"cd {REPO} && git log -1 --format='repo at %h %s'")

# ---- inputs
M = f"{TMP}/in"; ok = fetch("models/ball/clicks_latest.pt", f"{M}/clicks_latest.pt") and fetch("SFKBP1109/video.mp4", f"{M}/SFKBP1109.mp4")
if fetch("models/ball/best.json", f"{M}/best.json"):
    REPORT["best_json"] = json.load(open(f"{M}/best.json")); BAR = max(BAR, int(REPORT["best_json"].get("correct", BAR))); REPORT["bar"] = BAR
labs = {m: f"{M}/{m}_trainset_clicks.json" for m in MATCHES if fetch(f"labels/{m}_trainset_clicks.json", f"{M}/{m}_trainset_clicks.json")}
if not ok or not labs:
    REPORT["error"] = "inputs missing on R2 (run the [modal2r2] copy first)"; save(); L(REPORT["error"]); sys.exit(0)
CJ = os.environ.get("BF_CLICKS") or f"{REPO}/results/labels/SFKBP1109_ball_clicks.json"
REPORT["exam_frames"] = sum(r["split"] == "exam" for r in json.load(open(CJ))["frames"])

# ---- datasets: SFK-BP click crops + exam once, copied per variant; other matches' crops per variant
base = f"{TMP}/ds_base"; shutil.rmtree(base, ignore_errors=True)
REPORT["sfk_clicks"] = BC.build_crop_dataset(CJ, f"{M}/SFKBP1109.mp4", base, log=L, player_weights=None); save()
DS = {}
for v in VARIANTS:
    DS[v] = f"{TMP}/ds_{v}"; shutil.rmtree(DS[v], ignore_errors=True); shutil.copytree(base, DS[v])
    open(f"{DS[v]}/data.yaml", "w").write(f"path: {DS[v]}\ntrain: images/train\nval: images/val\nnames: ['ball']\n"); REPORT["variants"][v] = {"crops_per_match": {}}
for m, lj in labs.items():
    vid = f"{TMP}/{m}.mp4"
    if not fetch(f"{m}/video.mp4", vid): continue
    all_l = json.load(open(lj))["labels"]
    for v in VARIANTS:
        keep = [l for l in all_l if v == "all" or l.get("wasb_agrees")]
        tj = f"{TMP}/{m}_{v}.json"; json.dump({"labels": [[l["frame"], l["x"], l["y"], l["conf"]] for l in keep]}, open(tj, "w"))
        n = BC.add_auto_crops(tj, vid, DS[v], n_auto=min(len(keep), PER_MATCH), log=L, prefix="x_" + m.replace("-", "_")[:40])
        REPORT["variants"][v]["crops_per_match"][m] = {"labels": len(keep), "of": len(all_l), "crops": n}; save()
    os.remove(vid)                                                          # disk: one match video at a time
for v in VARIANTS: REPORT["variants"][v]["train_crops"] = len(os.listdir(f"{DS[v]}/images/train"))

# ---- the current finder re-graded here (must reproduce ~70/108, else the comparison is not fair)
cur = BC.grade(f"{M}/clicks_latest.pt", CJ, base, log=L); REPORT["clicks_latest_here"] = cur["new"]; save()
bar = max(BAR, cur["new"]["correct"])
if DRY and os.environ.get("BF_BAR"): bar = int(os.environ["BF_BAR"])      # dry run only: exercise the candidate path
REPORT["bar_used"] = bar

# ---- train + grade each variant
best_v = None
for v in VARIANTS:
    t1 = time.time(); L(f"variant {v}: training from clicks_latest.pt on {REPORT['variants'][v]['train_crops']} crops, cap {TRAIN_MIN} min")
    try:
        model = YOLO(f"{M}/clicks_latest.pt")
        model.train(data=f"{DS[v]}/data.yaml", epochs=EPOCHS, time=TRAIN_MIN / 60.0, imgsz=640 if not DRY else 160, batch=16 if not DRY else 2, lr0=0.001, freeze=4, mosaic=0.5, scale=0.3, fliplr=0.5,
                    project=f"{TMP}/ballft", name=v, exist_ok=True, verbose=False, patience=100, val=False, plots=False, workers=4 if not DRY else 0)
        w = f"{TMP}/ballft/{v}/weights/last.pt"; s = BC.grade(w, CJ, base, log=L)
        REPORT["variants"][v].update({"train_min": round((time.time() - t1) / 60, 1), "exam": s["new"], "exam_farzoom": s["new_farzoom"],
                                      "frames": [{"file": r["file"], "hit": r["new"]["correct"], "dist": r["new"]["dist"]} for r in s["frames"]]})
        L(f"variant {v}: exam {s['new']['correct']}/{s['new']['of']} vs bar {bar}")
        if s["new"]["correct"] > bar and (best_v is None or s["new"]["correct"] > REPORT["variants"][best_v]["exam"]["correct"]): best_v = v; shutil.copy(w, f"{TMP}/best_candidate.pt")
    except Exception as e:
        import traceback; REPORT["variants"][v]["error"] = traceback.format_exc()[-1500:]; L(f"variant {v} failed: {e!r}")
    save()
if best_v:
    name = f"candidate_{time.strftime('%Y%m%d')}.pt"; shutil.copy(f"{TMP}/best_candidate.pt", f"{W}/{name}")
    REPORT["candidate"] = {"variant": best_v, "file": name, "exam": REPORT["variants"][best_v]["exam"], "MB": round(os.path.getsize(f"{W}/{name}") / 1e6, 1)}
    L(f"CANDIDATE: {best_v} beats {bar}/108 -> {name}")
else: REPORT["candidate"] = None; L(f"no variant beats {bar}/108: no candidate")
save(); L("done")
