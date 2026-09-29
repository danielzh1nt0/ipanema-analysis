# Kaggle (free GPU), P1 29 Sep (v2): the other 4 matches (Vasalund, Djursholm, Solberga, Solheim) after the striped-kit fix v2
# of each training match + the SFK-BP reference clip -> /kaggle/working/<match>_<start>_* (pictures, summary, all rows)
import subprocess, os, glob, shutil, json, time
import cv2
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-3000:], flush=True); return r.returncode
t0 = time.time(); R2 = "{{R2}}"; W = "/kaggle/working"
sh("nvidia-smi --query-gpu=name --format=csv")
sh("git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git /kaggle/temp/ia")
sh("pip install -q rfdetr==1.11.0 supervision")
sh("cd /kaggle/temp/ia && git log -1 --format='repo at %h %s'")
MATCHES = os.environ.get("P2_MATCHES", "solberga-vs-p09-norrviken-2026-09-11,p15u-vs-djursholm-2026-09-26,p15u-vs-vasalund-2026-09-20,p09-norrviken-vs-solheim-2026-08-30"
                         ).split(",")
DUR = os.environ.get("P2_DUR", "60"); FRACS = (0.2, 0.38, 0.72)          # 38%: before half time, 72%: second half
info = {}
for M in MATCHES:
    if M == "SFKBP1109_s1200": starts = [20, 120, 220]                    # the 5-min reference clip
    else:
        c = cv2.VideoCapture(f"{R2}/{M}/video.mp4"); n = c.get(cv2.CAP_PROP_FRAME_COUNT); fps = c.get(cv2.CAP_PROP_FPS) or 29.97; c.release()
        if n < 100: print("not reachable", M); info[M] = "not reachable"; continue
        starts = [int(n / fps * f) for f in FRACS]
    for s in starts:
        tag = f"{M}_{s}"; out = f"/kaggle/temp/out/{tag}"; t1 = time.time()
        env = dict(os.environ, MATCH=M, START_S=str(s), DUR_S=DUR, SKIP_INSTALL="1", SAVE_ALL_ROWS="1", DETECTORS="rfdetr", TRACKTEST_OUT=out,
                   IPANEMA_DET_BATCH="8", IPANEMA_LOG_EVERY="600", R2_PUBLIC_URL=R2)
        r = subprocess.run(["python", "tools/tracktest.py"], cwd="/kaggle/temp/ia", env=env, capture_output=True, text=True)
        open(f"{W}/{tag}_log.txt", "w").write((r.stdout + r.stderr)[-6000:])
        for f in glob.glob(f"/kaggle/temp/ia/{out}/*") + glob.glob(f"{out}/*"):
            b = os.path.basename(f)
            if b.startswith("compare_"): cv2.imwrite(f"{W}/{tag}_{b}", cv2.imread(f), [cv2.IMWRITE_JPEG_QUALITY, 72])
            elif b.startswith("raw_") and b in ("raw_0000.jpg",): cv2.imwrite(f"{W}/{tag}_{b}", cv2.imread(f), [cv2.IMWRITE_JPEG_QUALITY, 85])
            elif b.endswith((".json", ".gz", ".md", ".png")): shutil.copy(f, f"{W}/{tag}_{b}")
        info[tag] = {"returncode": r.returncode, "minutes": round((time.time() - t1) / 60, 1)}; print(tag, info[tag], flush=True)
        json.dump({"pieces": info, "minutes": round((time.time() - t0) / 60, 1)}, open(f"{W}/run_info.json", "w"), indent=1)
print("done", round((time.time() - t0) / 60, 1), "min")
