"""CI step: WASB regen (optional) + picker test on the clip; everything it prints lands in results/ball/pick_log.txt"""
import os, json, traceback, sys
os.makedirs("results/ball", exist_ok=True); log = open("results/ball/pick_log.txt", "w")
def say(*a): s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
try:
    import modal
    msg = os.environ.get("MSG", ""); say("msg:", msg.splitlines()[0] if msg else "(none)")
    if "[ballpick:regen]" in msg:
        r = modal.Function.from_name("ipanema", "wasb_regen").remote("SFKBP1109_s1200", 0.05); say("regen:", r); json.dump(r, open("results/ball/wasb_regen.json", "w"), indent=1)
    out = modal.Function.from_name("ipanema", "ball_pick_test").remote("SFKBP1109_s1200")
    json.dump(out, open("results/ball/pick_test.json", "w"), indent=1); say(json.dumps(out, indent=1))
except Exception:
    say("FAILED:\n" + traceback.format_exc()[-6000:])
