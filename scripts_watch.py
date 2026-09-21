"""Run a Modal function from GitHub Actions and watch it: stream its logs from the volume into results/live/<match>.log
(committed every ~90 s so they can be read while it runs) and cancel it automatically when a known failure sign appears."""
import os, re, sys, time, json, base64, subprocess

DANGER = [
    (r"mosaic calibration failed", "the panorama calibration crashed, so positions would come from the wrong fallback calibration"),
    (r"RUN FAILED", "the run crashed"),
    (r"too many pieces failed", "too many pieces failed"),
]
FALLBACK = (r"calibration: \d+ frames, keypoints on", "the fallback calibration was used although this match has a panorama calibration")

def read_logs(vol, mid, since):
    out = {}
    try: entries = vol.listdir(f"match_analysis/logs/{mid}", recursive=True)
    except Exception: return out
    for e in entries:
        if not str(e.path).endswith(".log"): continue
        if getattr(e, "mtime", since) < since - 5: continue            # only logs written by this run
        try: out[os.path.basename(str(e.path))] = b"".join(vol.read_file(e.path)).decode(errors="replace")
        except Exception: pass
    return out

def danger_in(logs, check_fallback):
    rules = DANGER + ([FALLBACK] if check_fallback else [])
    for name, text in sorted(logs.items()):
        for pat, why in rules:
            m = re.search(pat, text)
            if m: return f"{why} ({name}: '{m.group(0)}')"
    return None

def render(mid, logs, status):
    parts = [f"# live log · {mid} · {status} · {time.strftime('%H:%M:%S', time.gmtime())} UTC\n"]
    for name in sorted(logs, key=lambda n: (not n.startswith("run_"), n)):
        lines = logs[name].rstrip("\n").split("\n")
        parts.append(f"\n## {name} ({len(lines)} lines, last 60)\n" + "\n".join(lines[-60:]))
    return "\n".join(parts) + "\n"

def push(msg):
    subprocess.run(f"git add results && (git commit -q -m '{msg}' || true) && git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main", shell=True)

def watch(call, vol, mid, since, check_fallback, every=90, poll=45, write=True, log_path=None, cancel=True):
    import modal
    log_path = log_path or f"results/live/{mid}.log"; os.makedirs(os.path.dirname(log_path), exist_ok=True); last = 0.0
    while True:
        done, res, err = False, None, None
        try: res = call.get(timeout=poll); done = True
        except modal.exception.FunctionTimeoutError as e: done, err = True, f"hit its time limit ({e})"
        except modal.exception.OutputExpiredError as e: done, err = True, f"result expired ({e})"
        except modal.exception.TimeoutError: pass                         # still running
        except Exception as e: done, err = True, repr(e)[:500]
        logs = read_logs(vol, mid, since)
        reason = None if done else danger_in(logs, check_fallback)
        status = "STOPPED AUTOMATICALLY: " + reason if reason else ("failed: " + err if err else ("finished" if done else "running"))
        if write: open(log_path, "w").write(render(mid, logs, status))
        if reason:
            if cancel: call.cancel(terminate_containers=True)
            if write: push(f"live log {mid}: stopped automatically")
            return None, "stopped automatically: " + reason
        if write and (done or time.time() - last > every): push(f"live log {mid}: {status}"[:70]); last = time.time()
        if done: return res, err

def main():
    import modal
    kind, mid = sys.argv[1], sys.argv[2]
    R2 = os.environ["R2_PUBLIC_URL"]; base = re.sub(r"_(seg|s|c)\d+$", "", mid)
    check_fallback = os.path.exists(f"calibration/{base}.json")
    vol = modal.Volume.from_name("ipanema-data"); since = time.time()
    fn = modal.Function.from_name("ipanema", "run_full" if kind == "full" else "run_match")
    call = fn.spawn(mid, f"{R2}/{mid}/video.mp4")
    print(f"started {kind} {mid}: {call.object_id}", flush=True)
    res, err = watch(call, vol, mid, since, check_fallback)
    os.makedirs("results/modal", exist_ok=True); name = f"{mid}_full" if kind == "full" else mid
    if res:
        open(f"results/modal/{name}.txt", "w").write("\n".join(res["log_tail"]) + "\n\nSUMMARY " + json.dumps(res["summary"], default=str))
        for fname, b64 in (res.get("files") or {}).items(): open(f"results/modal/{name}_{fname}", "wb").write(base64.b64decode(b64))
        print("\n".join(res["log_tail"][-30:]))
    if err: print("RUN DID NOT FINISH:", err); sys.exit(1)

if __name__ == "__main__": main()
