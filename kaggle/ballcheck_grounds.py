# Kaggle (free GPU, 29 Sep, B6-lite): does the new RF-DETR ball finder (exam 84/108 on SFK-BP) hold on grounds it never
# trained on? Uses Daniel's yes/no checks from earlier (results/review/<match>_clicks): Reymersholm (night, 23 match-ball yes,
# 9 not-a-ball) and Solberga (7 yes, 4 not-a-ball). No new clicking.
# Scores per ground: yes-moments where the finder's top guess / any top-3 guess (conf >= 0.25) is within 30 px of the ball;
# not-a-ball moments where the top guess sits on the thing Daniel said is not a ball. Plus a picture sheet per ground.
# Caveat: the yes-moments were picked from the old finder's guesses, so they lean to balls the old finder could see.
import os, sys, json, time, subprocess, traceback, numpy as np
R2 = os.environ.get("R2_BASE", "{{R2}}").rstrip("/"); WORK = os.environ.get("WORK", "/kaggle/working"); TMP = os.environ.get("TMPD", "/kaggle/temp")
os.makedirs(WORK, exist_ok=True); os.makedirs(TMP, exist_ok=True); LOCAL = os.environ.get("LOCAL") == "1"
t0 = time.time(); LOG = []; REPORT = {"started": time.strftime("%Y-%m-%d %H:%M")}
def log(m): LOG.append(f"{(time.time() - t0) / 60:5.1f} min  {m}"); print(LOG[-1], flush=True); open(f"{WORK}/log.txt", "w").write("\n".join(LOG) + "\n")
def save(): REPORT["minutes"] = round((time.time() - t0) / 60, 1); json.dump(REPORT, open(f"{WORK}/result.json", "w"), indent=1)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); log(f"$ {c[:100]} -> {r.returncode} {(r.stdout + r.stderr)[-200:].strip()}"); return r.returncode
GROUNDS = ["p15u-vs-reymersholm-2026-09-18", "solberga-vs-p09-norrviken-2026-09-11"]
try:
    REPO = os.environ.get("REPO_DIR", f"{TMP}/ia")
    if not LOCAL:
        sh(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}")
        sh('pip install -q "rfdetr==1.11.0"')
    sys.path.insert(0, REPO)
    import cv2
    from ipanema import ballrf as BR
    model = BR.load(f"{REPO}/{BR.WEIGHTS}"); log("new ball finder loaded")
    for m in GROUNDS:
        try:
            items = json.load(open(f"{REPO}/results/review/{m}_clicks/items.json")); ans = json.load(open(f"{REPO}/results/review/{m}_clicks/answers.json"))
            ids = sorted(ans["answers"]); spare = set(ans.get("spare_ball", [])); notb = set(ans.get("not_a_ball", []))
            rows = []; src = f"{TMP}/fake.mp4" if LOCAL else f"{R2}/{m}/video.mp4"; cap = cv2.VideoCapture(src)
            if not cap.isOpened(): raise RuntimeError(f"cannot open {m} video")
            tiles = []
            for i, (iid, it) in enumerate(zip(ids, items)):
                kind = "spare" if iid in spare else "not_ball" if iid in notb else "ball" if ans["answers"][iid] == "yes" else "not_ball"
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0 if LOCAL else int(it["frame"])); ok, f = cap.read()
                if LOCAL and i >= 3: break
                if not ok: continue
                if f.shape[:2] != (1080, 1920): f = cv2.resize(f, (1920, 1080))
                g = BR.detect_many(model, [f])[0]; tx, ty = it["xy"]
                d = [float(np.hypot(x - tx, y - ty)) for x, y, _ in g]
                top_hit = bool(g and d[0] <= 30); top3 = bool(any(dd <= 30 and gg[2] >= 0.25 for dd, gg in zip(d[:3], g[:3])))
                rows.append({"id": iid, "kind": kind, "frame": it["frame"], "xy": it["xy"], "top_on_it": top_hit, "top3_on_it": top3,
                             "top": [round(v, 3) for v in g[0]] if g else None, "guesses": [[round(a, 1), round(b, 1), round(c, 3)] for a, b, c in g[:5]]})
                x0, y0 = int(min(max(tx - 160, 0), 1600)), int(min(max(ty - 120, 0), 840)); crop = f[y0:y0 + 240, x0:x0 + 320].copy()
                cv2.circle(crop, (int(tx - x0), int(ty - y0)), 14, (0, 255, 255), 1)
                if g: cv2.drawMarker(crop, (int(g[0][0] - x0), int(g[0][1] - y0)), (0, 0, 255) if not top_hit else (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(crop, f"{iid} {kind} top {g[0][2]:.2f}" if g else f"{iid} {kind} none", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                tiles.append(crop)
            cap.release()
            b = [r for r in rows if r["kind"] == "ball"]; nb = [r for r in rows if r["kind"] == "not_ball"]
            REPORT[m] = {"ball_moments": len(b), "top_guess_on_ball": sum(r["top_on_it"] for r in b), "top3_on_ball": sum(r["top3_on_it"] for r in b),
                         "not_ball_moments": len(nb), "top_guess_on_not_ball": sum(r["top_on_it"] for r in nb), "rows": rows}
            log(f"{m}: ball {REPORT[m]['top_guess_on_ball']}/{len(b)} top guess, {REPORT[m]['top3_on_ball']}/{len(b)} in top 3; fooled by not-a-ball {REPORT[m]['top_guess_on_not_ball']}/{len(nb)}")
            if tiles:
                while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
                cv2.imwrite(f"{WORK}/sheet_{m[:20]}.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
            save()
        except Exception as e: log(f"{m}: skipped ({e!r})"); REPORT[m] = {"error": repr(e)}; save()
    REPORT["ok"] = True; save(); log("done")
except Exception:
    log(traceback.format_exc()); REPORT["ok"] = False; REPORT["error"] = traceback.format_exc()[-3000:]; save()
