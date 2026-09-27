"""CI step. [ballmatch-test:<id>] runs 2 pieces of a match and waits (measure). [ballmatch:<id1>,<id2>,...] starts whole
matches in the background (spawn) and returns. [ballstatus] reads every match's progress file from the volume."""
import os, re, json, base64, traceback, modal
msg = os.environ.get("MSG", ""); os.makedirs("results/datasets", exist_ok=True); out = {}
try:
    t = re.search(r"\[ballmatch-test:([A-Za-z0-9_.-]+)\]", msg)
    if t: out["test"] = modal.Function.from_name("ipanema", "ball_match").remote(t.group(1), 300, 2)
    a = re.search(r"\[ballmatch:([A-Za-z0-9_.,-]+)\]", msg)
    if a:
        f = modal.Function.from_name("ipanema", "ball_match")
        out["started"] = {mid: f.spawn(mid, 300, 0).object_id for mid in a.group(1).split(",")}
    if "[ballstatus]" in msg or a:
        ff = modal.Function.from_name("ipanema", "fetch_file"); st = {}
        for m in json.load(open("reference/training_matches.json"))["matches"]:
            r = ff.remote(f"logs/ballmatch/{m['id']}.json", 0)
            if "b64" in r:
                p = json.loads(base64.b64decode(r["b64"])); st[m["id"]] = {k: p.get(k) for k in ("pieces", "done", "failed", "elapsed_min", "finished")}
        out["status"] = st
except Exception: out["error"] = traceback.format_exc()[-3000:]
json.dump(out, open("results/datasets/ballmatch.json", "w"), indent=1); print(json.dumps(out, indent=1)[:4000])
