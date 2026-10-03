"""S3 (3 Oct): build pasted-ball pictures and the sheets to check them by eye. Free, CPU, local.
    python tools/s3_paste.py [per_frame=6] [seed=0]
Sources: checked ball crops (SFK-BP train clicks, results/volume/cache/ballcrops/SFKBP1109_crops.npz, test clips never
used; K1 balls of 4 more grounds, results/free/k1b/k1_crops.npz). Targets: our own frames with people boxes
(results/qa/kitprobe/<ground>/, 3 grounds incl. Reymersholm night). Output results/ball/s3/:
  labels.json      every pasted ball: frame, x, y, d, box, source crop (frames rebuild exactly from seed)
  sheet_real_vs_pasted.jpg   real balls next to pasted ones at the same size (top: real, bottom: pasted)
  sheet_blind.jpg + blind_key.json   48 windows, half real / half pasted, shuffled: can the eye tell them apart?
  frame_<ground>.jpg  one whole frame per ground, pasted balls circled
  summary.json"""
import sys, os, json, glob, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import pasteball as PB

PER = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = os.environ.get("S3_OUT", "results/ball/s3"); os.makedirs(OUT, exist_ok=True)
BLIND_SKIP = "p15u-vs-spanga-2026-09-25"
GROUNDS = ["SFKBP1109_s1200", "p15u-vs-reymersholm-2026-09-18", "p15u-vs-spanga-2026-09-25"]


def load_sources():
    src = []
    d = np.load("results/volume/cache/ballcrops/SFKBP1109_crops.npz", allow_pickle=True); X, M = d["X"], d["meta"]
    for i in range(len(M)):
        if M[i][6] == 1 and M[i][1] == "train": src.append(("SFKBP1109", f"sfk:{i}", X[i][1]))
    k = np.load("results/free/k1b/k1_crops.npz", allow_pickle=True); KX, KM = k["X"], k["meta"]
    for i in range(len(KM)):
        if KM[i][6] == 1 and int(KM[i][8]) == 0: src.append((str(KM[i][7]), f"k1:{i}", KX[i][1]))   # checked frame only
    return src


def main():
    rng = np.random.default_rng(SEED)
    src = load_sources(); patches = []
    for g, sid, cr in src:
        p = PB.cut_patch(cr)
        if p: p.update(ground=g, sid=sid, crop=cr); patches.append(p)
    print(f"{len(src)} checked balls, {len(patches)} clean patches", flush=True)
    labels, frames_out, fake_win = [], {}, []

    def win(img, x, y, S=32):
        x, y = int(x), int(y); return img[y - S // 2:y + S // 2, x - S // 2:x + S // 2].copy()
    allframes = []
    for g in GROUNDS:
        boxes = json.load(open(f"results/qa/kitprobe/{g}/boxes.json"))
        for fp in sorted(glob.glob(f"results/qa/kitprobe/{g}/f*.jpg")):
            fn = os.path.basename(fp); allframes.append((g, fn, boxes.get(fn, [])))
            fr = cv2.imread(fp); sp = PB.spots(fr, boxes.get(fn, []), PER, rng); img = fr; done = []
            for x, y, dd, kind in sp:
                p = patches[int(rng.integers(len(patches)))]
                img, box = PB.paste(img, p, x, y, dd, rng)
                if box is None: continue
                labels.append(dict(ground=g, frame=fn, x=x, y=y, d=round(dd, 2), box=[round(v, 1) for v in box], src=p["sid"], kind=kind))
                done.append((x, y, dd))
            for x, y, dd in done: fake_win.append(win(img, x, y))
            if g not in frames_out and len(done) >= 3: frames_out[g] = (img, done)
    print(f"{len(labels)} pasted balls on {len(set((l['ground'], l['frame']) for l in labels))} frames", flush=True)
    Z = 4
    # sheet 1: real far-ish balls (top) vs pasted ones as they will be used (bottom)
    real = []
    for g, sid, cr in src:
        q = PB.cut_patch(cr, ring_dirty=1.0, max_d=30)
        if q: real.append((q["d"], cr))
    real.sort(key=lambda t: t[0]); small = real[:48]                     # the smallest real checked balls (d up to ~10)
    ri = rng.permutation(len(small)); fi = rng.choice(len(fake_win), 24, replace=False)
    R = [cv2.resize(small[i][1], (32 * Z, 32 * Z), interpolation=cv2.INTER_NEAREST) for i in ri[:24]]
    F = [cv2.resize(fake_win[i], (32 * Z, 32 * Z), interpolation=cv2.INTER_NEAREST) for i in fi]
    rows = [np.hstack(R[r * 8:(r + 1) * 8]) for r in range(3)] + [np.full((12, 32 * Z * 8, 3), 255, np.uint8)] + \
           [np.hstack(F[r * 8:(r + 1) * 8]) for r in range(3)]
    cv2.imwrite(f"{OUT}/sheet_real_vs_pasted.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # sheet 2, blind: the other 24 small real balls vs 24 pasted at the SAME sizes on our frames, shuffled
    items = []
    for i in ri[24:48]:
        d0, cr = small[i]; items.append(("real", cr, round(d0, 1)))
        for _ in range(50):
            g, fn, b = allframes[int(rng.integers(len(allframes)))]
            if g == BLIND_SKIP: continue                                 # floodlit orange ground, no small real balls there: colour alone would give it away
            fr = cv2.imread(f"results/qa/kitprobe/{g}/{fn}")
            sp = PB.spots(fr, b, 1, rng, d_lo=0, d_hi=99)
            if sp:
                x, y = sp[0][:2]; img, box = PB.paste(fr, patches[int(rng.integers(len(patches)))], x, y, d0, rng)
                if box: items.append(("pasted", win(img, x, y), round(d0, 1))); break
    order = rng.permutation(len(items)); key = {}; tiles = []
    for n, j in enumerate(order):
        kind, im, d0 = items[j]; key[str(n)] = dict(kind=kind, d=d0)
        t = cv2.resize(im, (32 * Z, 32 * Z), interpolation=cv2.INTER_NEAREST).copy()
        cv2.putText(t, str(n), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); tiles.append(t)
    while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{OUT}/sheet_blind.jpg", np.vstack([np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(len(tiles) // 8)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump(key, open(f"{OUT}/blind_key.json", "w"), indent=0)
    for g, (img, sp) in frames_out.items():
        im = img.copy()
        for x, y, dd in sp: cv2.circle(im, (int(x), int(y)), int(dd) + 10, (255, 0, 255), 2)
        cv2.imwrite(f"{OUT}/frame_{g[:24]}.jpg", cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        x, y, dd = sp[0]; z = img[max(0, int(y) - 120):int(y) + 120, max(0, int(x) - 200):int(x) + 200]
        cv2.imwrite(f"{OUT}/zoom_{g[:24]}.jpg", cv2.resize(z, (z.shape[1] * 2, z.shape[0] * 2), interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump(labels, open(f"{OUT}/labels.json", "w"))
    ds = np.array([l["d"] for l in labels])
    summ = dict(seed=SEED, per_frame=PER, checked_balls=len(src), clean_patches=len(patches),
                patches_by_ground={g: sum(1 for p in patches if p["ground"] == g) for g in sorted(set(p["ground"] for p in patches))},
                pasted=len(labels), by_ground={g: sum(1 for l in labels if l["ground"] == g) for g in GROUNDS},
                d_px_p10_p50_p90=[round(float(v), 1) for v in np.percentile(ds, [10, 50, 90])] if len(ds) else [])
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
