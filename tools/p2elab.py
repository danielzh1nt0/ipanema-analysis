"""P2e (4 Oct, worker, local $0): on grounds without calibration (tracktest / P2 pieces) positions are screen positions, so
the goalmouth keeper rules in tracking.clean fire on whoever stands at the picture's left / right edge and force them into
one team. For every P2 piece (results/qa/p2/<piece>/rows_all.json.gz + log.txt) list the tracks the old rule made keepers,
their own colour vote over the whole track (same rule as tracking.track: K if >= 60% 'neither', else majority A/B) and
cut a crop sheet of those whose team changes, from the key frames, to check by eye.
Output: results/qa/p2e/p2elab.json, results/qa/p2e/edge_keepers.jpg"""
import gzip, json, collections, re, glob, os, cv2, numpy as np
OUT = "results/qa/p2e"

def vote(raw):
    v = [x for x in raw if x != "O"]; ab = [x for x in v if x in ("A", "B")]
    if not v: return None
    return "K" if (v.count("K") >= 0.6 * len(v) or not ab) and "K" in v else max(set(ab), key=ab.count)

def piece_tracks(d):
    log = open(d + "log.txt").read(); m = re.findall(r"keepers \[([^\]]*)\]", log)
    kp = [int(x) for x in m[-1].split(",") if x.strip()] if m else []
    r = json.load(gzip.open(d + "rows_all.json.gz")); v = list(r["rows"])[-1]
    rows, raw, bh = r["rows"][v], r["raw_team"][v], r["box_h"][v]
    reads = collections.defaultdict(list); asg = collections.defaultdict(collections.Counter); seen = collections.defaultdict(list)
    for k, rs in rows.items():
        for i, row in enumerate(rs):
            asg[row[0]][row[1]] += 1
            if raw[k][i]: reads[row[0]].append(raw[k][i])
            if row[2] and bh[k][i] and not row[3]: seen[row[0]].append((int(k), row[2], bh[k][i]))
    out = []
    for t in kp:
        old = asg[t].most_common(1)[0][0]; new = vote(reads[t])
        out.append({"track": t, "rows": sum(asg[t].values()), "old": old, "own_vote": new, "reads": dict(collections.Counter(reads[t])), "seen": seen[t]})
    return out

if __name__ == "__main__":
    res = {}; tiles = []
    for d in sorted(glob.glob("results/qa/p2/*_*/")):
        if not os.path.exists(d + "rows_all.json.gz"): continue
        piece = d.rstrip("/").split("/")[-1]; tr = piece_tracks(d)
        keys = sorted(int(os.path.basename(f)[4:8]) for f in glob.glob(d + "raw_*.jpg"))
        for t in tr:
            t["changes"] = t["own_vote"] != t["old"]
            if not t["changes"]: continue
            # nearest key frame where the track is seen
            best = min(((min(abs(k - kk) for kk in keys), k, px, h) for k, px, h in t["seen"]), default=None)
            if best is None or best[0] > 15: continue
            kk = min(keys, key=lambda x: abs(x - best[1])); img = cv2.imread(f"{d}raw_{kk:04d}.jpg")
            x, y, h = best[2][0], best[2][1], max(40.0, best[3])
            x0, y0 = int(max(0, x - 0.8 * h)), int(max(0, y - 1.3 * h)); x1, y1 = int(min(img.shape[1], x + 0.8 * h)), int(min(img.shape[0], y + 0.3 * h))
            c = cv2.resize(img[y0:y1, x0:x1], (160, 200)); c = np.vstack([c, np.zeros((40, 160, 3), np.uint8)])
            txt = f"{len(tiles)}: {t['old']}->{t['own_vote']}"
            cv2.putText(c, txt, (4, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); tiles.append(c); t["tile"] = len(tiles) - 1
        for t in tr: t.pop("seen")
        res[piece] = tr
    json.dump(res, open(f"{OUT}/p2elab.json", "w"), indent=1)
    rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8]))) for i in range(0, len(tiles), 8)]
    if rows: cv2.imwrite(f"{OUT}/edge_keepers.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    n = sum(len(v) for v in res.values()); ch = sum(t["changes"] for v in res.values() for t in v)
    print(f"{n} edge 'keepers' on {len(res)} pieces, {ch} change team, {len(tiles)} on the sheet")
