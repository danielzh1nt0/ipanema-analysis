"""GitHub runner: frames for a new venue's pitch-point clicks (Modal cuts them, cents), committed under results/venue/<match>/frames."""
import os, sys, json, base64, modal
trig = open("triggers/venue.txt").read().split(); match_id = next((w for w in trig if w.startswith("match=")), "match=")[6:]
if not match_id: print("no match= in triggers/venue.txt"); sys.exit(1)
OUT = f"results/venue/{match_id}/frames"; os.makedirs(OUT, exist_ok=True)
res = modal.Function.from_name("ipanema", "venue_frames").remote(match_id, 30)
if "error" in res: print(res["error"]); sys.exit(1)
for fr in res["frames"]: open(f"{OUT}/{fr['file']}", "wb").write(base64.b64decode(fr["b64"]))
json.dump({"match": match_id, "fps": res["fps"], "minutes": res["minutes"], "frames": [{"file": f["file"], "t": f["t"]} for f in res["frames"]]}, open(f"results/venue/{match_id}/frames.json", "w"), indent=1)
print(f"{len(res['frames'])} frames from a {res['minutes']:.0f} min video")
