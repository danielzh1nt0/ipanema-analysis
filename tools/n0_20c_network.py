"""N0-20c: pass-network nodes - old (track's whole-match average) vs new (where the player played/received this team's
completed passes). Reads the repo match copies' stats.json; writes a summary and a picture of the SFK-BP clip.
PYTHONPATH=. python tools/n0_20c_network.py [out_dir]"""
import json, glob, os, sys, numpy as np, cv2
from ipanema import metrics as M

def compare(st):
    pl = {p["id"]: p for p in st["players"]}; old = M.pass_network(st["passes"], st["players"], nodes="avg"); new = M.pass_network(st["passes"], st["players"], nodes="passes")
    row = {}
    for tm in ("A", "B"):
        e = new[tm]["edges"]; no, nn = old[tm]["nodes"], new[tm]["nodes"]
        dangling = sum(1 for x in e if str(x["from"]) not in no or str(x["to"]) not in no)
        other = sum(1 for x in e if pl.get(x["from"], {}).get("team", tm) != tm or pl.get(x["to"], {}).get("team", tm) != tm)
        mv = [float(np.hypot(no[i]["x"] - nn[i]["x"], no[i]["y"] - nn[i]["y"])) for i in nn if i in no]
        mv_other = [float(np.hypot(no[i]["x"] - nn[i]["x"], no[i]["y"] - nn[i]["y"])) for i in nn if i in no and pl.get(int(i), {}).get("team", tm) != tm]
        row[tm] = {"edges": len(e), "passes": sum(x["count"] for x in e), "edges_without_node_old": dangling, "edges_without_node_new": sum(1 for x in e if str(x["from"]) not in nn or str(x["to"]) not in nn),
                   "edges_touching_other_team_track": other, "node_move_median_m": round(float(np.median(mv)), 1) if mv else None,
                   "node_move_other_team_tracks_median_m": round(float(np.median(mv_other)), 1) if mv_other else None}
    return old, new, row

def draw(net, L, W, title, s=8):
    img = np.full((int(W * s) + 60, int(L * s) + 20, 3), (60, 120, 60), np.uint8); o = (10, 50)
    P = lambda x, y: (int(o[0] + x * s), int(o[1] + y * s))
    cv2.rectangle(img, P(0, 0), P(L, W), (255, 255, 255), 1); cv2.line(img, P(L / 2, 0), P(L / 2, W), (255, 255, 255), 1)
    for x in (L / 3, 2 * L / 3): cv2.line(img, P(x, 0), P(x, W), (150, 190, 150), 1)
    col = {"A": (40, 40, 40), "B": (245, 245, 245)}
    for tm in ("A", "B"):
        nd = net[tm]["nodes"]
        for e in net[tm]["edges"]:
            a, b = nd.get(str(e["from"])), nd.get(str(e["to"]))
            if a and b: cv2.line(img, P(a["x"], a["y"]), P(b["x"], b["y"]), col[tm], 1 + e["count"], cv2.LINE_AA)
            else:                                                                    # edge with a missing node: red stub at the known end
                k = a or b
                if k: cv2.circle(img, P(k["x"], k["y"]), 9, (0, 0, 255), 2)
        for i, n in nd.items(): cv2.circle(img, P(n["x"], n["y"]), 4, col[tm], -1); cv2.circle(img, P(n["x"], n["y"]), 4, (0, 200, 255), 1)
    cv2.putText(img, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); return img

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-20c"; os.makedirs(out, exist_ok=True); rep = {}
    for f in sorted(glob.glob("results/volume/runs/matches/*/stats.json")):
        st = json.load(open(f)); mid = f.split("/")[-2]; old, new, row = compare(st); rep[mid] = row
        if mid == "SFKBP1109_s1200":
            L, W = st.get("pitch", {}).get("length", 105), st.get("pitch", {}).get("width", 68)
            cv2.imwrite(f"{out}/network_SFKBP1109_s1200.jpg", np.vstack([draw(old, L, W, "OLD nodes: track average (red ring = edge to a player with no node)"), draw(new, L, W, "NEW nodes: where the player passed / received")]))
    json.dump(rep, open(f"{out}/network_compare.json", "w"), indent=1); print(json.dumps(rep, indent=1))
