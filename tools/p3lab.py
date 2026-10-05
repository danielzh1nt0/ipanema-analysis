"""P3 (5 Oct, local, $0): tracklet joining on saved tracking rows. Pieces:
  Reymersholm 726 / 2227 (results/qa/p2b/track), 4227 (results/qa/p2e/track), Spanga 1159 / 2576 / 3994 (results/qa/p2),
  SFK-BP: the app's full-match export (results/volume/runs/matches/SFKBP1109, 10 Hz, metres), 3 x 20 s windows.
Measures pieces per 20 s and time tracked before/after joining; writes candidates for the by-eye crops (tools/p3_crops.py).
    python tools/p3lab.py [--emb results/qa/p3/emb.json]   -> results/qa/p3/p3lab.json"""
import os, sys, json, gzip, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ipanema import tracklets as T

PIECES = {"reym_726": ("p15u-vs-reymersholm-2026-09-18", "results/qa/p2b/track/p15u-vs-reymersholm-2026-09-18_726"),
          "reym_2227": ("p15u-vs-reymersholm-2026-09-18", "results/qa/p2b/track/p15u-vs-reymersholm-2026-09-18_2227"),
          "reym_4227": ("p15u-vs-reymersholm-2026-09-18", "results/qa/p2e/track/p15u-vs-reymersholm-2026-09-18_4227"),
          "spanga_1159": ("p15u-vs-spanga-2026-09-25", "results/qa/p2/p15u-vs-spanga-2026-09-25_1159"),
          "spanga_2576": ("p15u-vs-spanga-2026-09-25", "results/qa/p2/p15u-vs-spanga-2026-09-25_2576"),
          "spanga_3994": ("p15u-vs-spanga-2026-09-25", "results/qa/p2/p15u-vs-spanga-2026-09-25_3994")}
SFK_WIN = [900.0, 1800.0, 2700.0]          # app export seconds (video time), 20 s each
SFK_DIR = "results/volume/runs/matches/SFKBP1109"


def load_piece(name):
    """-> rows {k: [(id, team, px, filled, box_h, m)]}, fps, k0 (video frame of k=0), metres?"""
    if name.startswith("sfk_"):
        t0 = float(name.split("_")[1]); rows = {}; fps = 10.0
        for c in range(1, 11):
            p = f"{SFK_DIR}/frames_{c:03d}.json"
            if not os.path.exists(p): continue
            D = json.load(open(p))
            if D["t_end"] < t0 or D["t_start"] > t0 + 20: continue
            for f in D["frames"]:
                if t0 <= f["t"] < t0 + 20:
                    k = int(round((f["t"] - t0) * fps))
                    rows[k] = [(p_["id"], p_["team"], p_.get("px"), p_["state"] != "observed", None, p_.get("m")) for p_ in f["players"] if p_.get("m")]
        return rows, fps, t0, True
    d = json.load(gzip.open(f"{PIECES[name][1]}/rows_all.json.gz", "rt")); v = "new, RF-DETR"
    rows = {int(k): [tuple(r) + (h,) for r, h in zip(rs, d["box_h"][v][k])] for k, rs in d["rows"][v].items()}
    return rows, d["fps"], d["k0"] / d["fps"], False


def all_names(): return list(PIECES) + [f"sfk_{int(t)}" for t in SFK_WIN]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--emb"); ap.add_argument("--max_app", type=float, default=None); ap.add_argument("--out", default="results/qa/p3/p3lab.json")
    a = ap.parse_args()
    EMB = json.load(open(a.emb)) if a.emb else {}
    rep = {}
    for name in all_names():
        rows, fps, t0, use_m = load_piece(name); n = len(rows)
        P = T.pieces(rows, fps)
        before = T.measure(P, fps, n)
        C = T.candidates(P, fps, rows, use_m=use_m)
        emb = {int(i): np.asarray(v) / np.linalg.norm(v) for i, v in EMB.get(name, {}).items()} if EMB else None
        remap, used = T.join(P, C, emb=emb, max_app=a.max_app)
        after = T.measure(P, fps, n, remap)
        rep[name] = {"t0": t0, "fps": fps, "frames": n, "metres": use_m, "before": before, "after": after, "candidates": len(C), "joins": used,
                     "cands": C, "pieces": {str(i): {"start": d["start"], "end": d["end"], "team": d["team"], "n": len(d["k"])} for i, d in P.items()}}
        print(f"{name:13s} pieces/20s {before['pieces_per_20s']:5.1f} -> {after['pieces_per_20s']:5.1f}   median s {before['median_s']} -> {after['median_s']}"
              f"   5s+ share {before['share_time_in_5s_plus']} -> {after['share_time_in_5s_plus']}   cands {len(C)} joins {len(used)}")
    json.dump(rep, open(a.out, "w"), indent=0, default=float)
