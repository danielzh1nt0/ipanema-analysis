"""CI step: contact sheets of a clip (1 picture/s) into results/gt/<clip>/"""
import os, re, json, base64, traceback, modal
msg = os.environ.get("MSG", ""); mid = (re.search(r"\[sheets:([A-Za-z0-9_-]+)\]", msg) or re.search(r"(SFKBP1109_s1200)", "SFKBP1109_s1200")).group(1)
out_dir = f"results/gt/{mid}"; os.makedirs(out_dir, exist_ok=True)
try:
    r = modal.Function.from_name("ipanema", "contact_sheets").remote(mid)
    if "error" in r: open(f"{out_dir}/error.txt", "w").write(r["error"]); print(r)
    else:
        for i, b in enumerate(r.pop("sheets")): open(f"{out_dir}/sheet_{i:02d}.jpg", "wb").write(base64.b64decode(b))
        json.dump(r, open(f"{out_dir}/sheets.json", "w")); print(r)
except Exception: open(f"{out_dir}/error.txt", "w").write(traceback.format_exc()[-4000:])
