"""N0-28d: does the runner's OpenCV honour the ffmpeg reconnect options, and do the big R2 clips now open?

1. Stall test (no network): a local HTTP server that goes silent on the header read and on a mid-file read, opened by
   (a) the OpenCV the workflow installs (pinned 4.10.0.84 now) and (b) OpenCV 4.11.0.86 (what numpy<2 used to give;
   its OPENCV_FFMPEG_CAPTURE_OPTIONS check is upside down). Expect (a) all frames after ~10-s reconnects, (b) 0 frames.
2. Real R2 (only when R2_PUBLIC_URL is set): open the big Djursholm and Spånga files and cut the 20-s piece the nightly
   check uses, ROUNDS times each, with tools/clipopen.py - frames, tries, seconds, ffmpeg reconnect lines.
-> results/qa/n0-28d/runner_check.md + .json"""
import os, sys, json, time, subprocess, threading
sys.path.insert(0, os.getcwd())
OUT = os.environ.get("N0_28D_OUT", "results/qa/n0-28d"); os.makedirs(OUT, exist_ok=True)
ROUNDS = int(os.environ.get("ROUNDS", "3"))
MATCHES = os.environ.get("BIG_MATCHES", "p15u-vs-djursholm-2026-09-26,p15u-vs-spanga-2026-09-25").split(",")
OLD_CV = os.environ.get("OLD_CV", "4.11.0.86")


def make_clip(path, n=900, size=(640, 360), fps=30.0):
    import cv2, numpy as np
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0); base = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    for i in range(n):
        f = np.roll(base, i * 3, axis=1); cv2.putText(f, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3); w.write(f)
    w.release(); return path


def stall_server(path, stalls, stall_after):
    """-> (url, request log, server). Requests numbered in `stalls` send headers + `stall_after[k]` bytes, then go silent."""
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    size = os.path.getsize(path); reqs = []; lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            with lock:
                reqs.append(None); k = len(reqs)
            rng = self.headers.get("Range"); a, b = 0, size - 1
            if rng and rng.startswith("bytes="):
                s, e = rng[6:].split("-"); a = int(s) if s else 0; b = int(e) if e else size - 1
                self.send_response(206); self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(b - a + 1)); self.send_header("Accept-Ranges", "bytes"); self.end_headers()
            reqs[k - 1] = f"{k}: bytes {a}-" + (" STALL" if k in stalls else "")
            with open(path, "rb") as f:
                f.seek(a); left, sent = b - a + 1, 0
                try:
                    while left > 0:
                        if k in stalls and sent >= stall_after.get(k, 0):
                            time.sleep(300); return
                        c = f.read(min(65536, left)); self.wfile.write(c); left -= len(c); sent += len(c)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/clip.mp4", reqs, srv


PROBE = r"""
import sys, time, json, os
sys.path.insert(0, os.getcwd())
import cv2
from tools.clipopen import capture
url, k0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); t = time.time()
cap = capture(cv2, url, log=lambda m: print(m, file=sys.stderr))
cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT); t_open = time.time() - t
cap.set(cv2.CAP_PROP_POS_FRAMES, k0); got = 0
for _ in range(n):
    ok, f = cap.read()
    if not ok: break
    got += 1
print("RESULT " + json.dumps({"cv": cv2.__version__, "count": cnt, "open_s": round(t_open, 1), "frames": got, "want": n, "total_s": round(time.time() - t, 1)}))
"""


def probe(url, k0, n, pythonpath=None, timeout=400):
    env = dict(os.environ, OPENCV_FFMPEG_LOGLEVEL="24")                # warnings: ffmpeg prints 'Will reconnect at ...'
    if pythonpath: env["PYTHONPATH"] = pythonpath
    env.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    try:
        r = subprocess.run([sys.executable, "-c", PROBE, url, str(k0), str(n)], capture_output=True, text=True, timeout=timeout, env=env)
        hits = [ln[7:] for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]     # runner 28 Oct: other stdout lines came after the result
        res = json.loads(hits[-1]) if hits else {"error": (r.stdout[-200:] + " | " + r.stderr[-300:]).strip()}
        err = r.stderr
    except subprocess.TimeoutExpired:
        res, err = {"error": f"timed out after {timeout} s"}, ""
    res["reconnects"] = sum("will reconnect" in ln.lower() for ln in err.splitlines())
    res["warnings"] = [ln for ln in err.splitlines() if "WARNING" in ln][:2]
    return res


def old_cv_path():
    d = "/tmp/cv_old"
    if not os.path.exists(os.path.join(d, "cv2")):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", d, f"opencv-python-headless=={OLD_CV}"], capture_output=True, text=True)
        if r.returncode != 0: return None
    return d


def stall_test(clip, pythonpath=None):
    url, reqs, srv = stall_server(clip, stalls={2, 4}, stall_after={2: 0, 4: 300000})   # 2 = header (moov) read, 4 = the seek read
    res = probe(url, 300, 300, pythonpath); srv.shutdown(); res["requests"] = list(reqs); return res


def main():
    import cv2
    print("opencv", cv2.__version__, flush=True)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "opencv_installed": cv2.__version__, "stall": {}, "r2": []}
    clip = make_clip("/tmp/n0_28d_clip.mp4")
    rep["stall"]["installed"] = stall_test(clip)
    p = old_cv_path()
    rep["stall"][f"old {OLD_CV}"] = stall_test(clip, p) if p else {"error": "could not install"}
    base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
    if base:
        from tools.clipopen import open_clip, read_piece
        for r in range(ROUNDS):
            for m in MATCHES:
                t = time.time(); logs = []
                cap, fps, n_all, tries = open_clip(f"{base}/{m}/video.mp4", cv2, tries=3, wait_s=20, log=logs.append)
                row = {"round": r + 1, "match": m, "open_tries": tries, "frames_in_clip": n_all, "open_s": round(time.time() - t, 1)}
                if cap is not None:
                    k0, n = int(float(os.environ.get("START_S", "1500")) * fps), int(20 * fps)
                    got, _, ptries = read_piece(f"{base}/{m}/video.mp4", cv2, k0, n, "/tmp/n0_28d_piece.mp4", tries=3, wait_s=20, log=logs.append, cap=cap)
                    row.update(piece=f"{got}/{n}", piece_tries=ptries)
                row["total_s"] = round(time.time() - t, 1); row["log"] = logs[-3:]
                print(row, flush=True); rep["r2"].append(row)
    json.dump(rep, open(f"{OUT}/runner_check.json", "w"), indent=1)
    L = [f"# N0-28d runner check ({rep['date']})", "", f"OpenCV the workflow installs: **{rep['opencv_installed']}**", "",
         "## Stall test (local server goes silent on the header read and on a mid-file read)", "",
         "| OpenCV | frames read | open s | total s | requests to the server (stalled) |", "|---|---|---|---|---|"]
    for k, v in rep["stall"].items():
        L.append(f"| {v.get('cv', k)} ({k}) | {v.get('frames', '-')}/{v.get('want', 300)} | {v.get('open_s', '-')} | {v.get('total_s', '-')} | {len(v.get('requests', []))} ({sum('STALL' in (q or '') for q in v.get('requests', []))}) |"
                 + (f" {v['error'][:80]}" if "error" in v else ""))
    if rep["r2"]:
        L += ["", "## Real R2 (20-s piece from 25:00, as the nightly check)", "", "| round | match | opens | piece | seconds |", "|---|---|---|---|---|"]
        for row in rep["r2"]:
            L.append(f"| {row['round']} | {row['match']} | {row['open_tries']} | {row.get('piece', 'NOT OPENED')} | {row['total_s']} |")
    open(f"{OUT}/runner_check.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    ok = rep["stall"]["installed"].get("frames", 0) >= 290 and all(r.get("piece", "0/1").split("/")[0] != "0" for r in rep["r2"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
