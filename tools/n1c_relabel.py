"""N1c (30 Oct, worker, local, $0, nothing on Modal): new AIK team labels from the shirt-colour votes the free runner read
over the whole match (results/free/n1c/votes.json, tools/n1c_kitvote_run.py), then possession recomputed and re-scored
on the 75 decided by-eye moments of N0-30a + N0-30b.

Per export track id (counts of dark / white / red / blue reads, 'none' ignored):
    red  >= RED_SHARE of its reads   -> AIK keeper (team B, gk)
    blue >= BLUE_SHARE                -> referee (no team: never a carrier)
    dark >= DOM x white               -> A (SFK)
    white >= DOM x dark               -> B (AIK)
    else / fewer than MIN_READS reads -> label kept as exported
Possession is recomputed with the rule the pipeline uses (possession_simple: the team whose nearest player's feet are
closest to the ball in the picture, within NEAR_M with the player's height in the picture as ruler, else loose; frames
with no ball keep the previous state; majority over +-SMOOTH rows) - first with the EXPORTED labels (to check the
re-computation copies the export) and then with the new ones, so the change is measured like for like.

    python tools/n1c_relabel.py            # -> results/qa/n1c/result.json, relabel.json
"""
import os, sys, json, gzip, bisect
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n0_30a_aik_poss import analyse, wilson, export_share
import n1c_kitread as K

FR_PATH = K.FR_PATH
SRC_DIR = os.environ.get("N1C_READS", "results/free/n1c3")
VOTES = os.path.join(SRC_DIR, "votes.json")
G1 = "results/qa/n0-30a/aik_poss_grades.json"
G2 = "results/qa/n0-30b/aik_h2_grades.json"
OUT = "results/qa/n1c"
H1, H2 = (366.0, 3225.0), (3995.0, 6899.0)
RED_SHARE, BLUE_SHARE, DOM, MIN_READS = 0.4, 0.4, 2.0, 3
NEAR_M, SMOOTH, BODY_M = 1.5, 3, 1.75
WIN_S = float(os.environ.get("N1C_WIN_S", "5"))     # window vote +-s; 0 = one vote per id for the whole match


def decide(v, red_share=RED_SHARE, blue_share=BLUE_SHARE, dom=DOM, min_reads=MIN_READS, field=True):
    """v = [dark, white, red, blue, none] -> ('A' | 'B' | 'ref' | None, gk). None = keep the export's label."""
    dark, white, red, blue = v[:4]; n = dark + white + red + blue
    if n < min_reads: return None, False
    if red >= red_share * n: return "B", True
    if blue >= blue_share * n: return "ref", False
    if not field: return None, False                 # keepers + referee only: field players keep the export's team
    if dark >= dom * max(white, 1e-9) and dark > 0: return "A", False
    if white >= dom * max(dark, 1e-9) and white > 0: return "B", False
    return None, False


def relabel_map(votes, **kw):
    out = {}
    for pid, v in votes.items():
        t, gk = decide(v, **kw)
        if t is not None: out[int(pid)] = (t, gk)
    return out


READS = os.path.join(SRC_DIR, "reads.json.gz")


def reads_by_id(path=READS):
    """{id: (times, classes)} from the free runner's per-row reads (every 0.9 s)"""
    r = json.load(gzip.open(path)); by = {}
    for t, ps in r["rows"]:
        for pid, c in ps:
            a = by.setdefault(pid, ([], [])); a[0].append(t); a[1].append(c)
    return by


# Which team defends which goal, by eye (N1c, 30 Oct): first half the orange keeper's LEFT goal is AIK's (p15 'dark #5 at
# the ball by AIK goal', p04) and the red keeper's RIGHT goal SFK's (p25 goal kick, darks around; N0-26b's kick-off picture
# ko1: dark SFK in the right half); second half the RIGHT goal is AIK's (q03, q09, q17, q65, q76; white kick-off after SFK's
# goal q70-q71) and the LEFT goal SFK's (q30 'orange SFK keeper'). So the keepers stayed at their goals and the red shirt
# is SFK's keeper in the first half - a keeper's team must come from his goal, not his shirt. (The export's own keeper
# labels follow periods attack_right_A=true, the mirrored direction still open under 'Waiting for Daniel', N0-26b.)
MID_X = 58.0                                   # pitch middle in the export's metres (players' x 1-99%: 4.5-114)
END_TEAM = {1: {"left": "B", "right": "A"}, 2: {"left": "A", "right": "B"}}
KEEPER_RULE = os.environ.get("N1C_KEEPER", "end")
FIELD = os.environ.get("N1C_FIELD", "0") == "1"      # 1 = also re-team field players by shirt (by eye ~half of those changes wrong)   # end = team of the goal he stands nearest; shirt = red -> AIK


def keeper_team(t, x_m, rule=None):
    rule = rule or KEEPER_RULE
    if rule != "end" or x_m is None: return "B"
    half = 1 if t < (H1[1] + H2[0]) / 2 else 2
    return END_TEAM[half]["right" if x_m < MID_X else "left"]          # the export's x runs from the picture's RIGHT goal (x ~0) to the left one (q09 keeper at the right goal: x 11.5; p15 left goal: x 111)


def window_labels(fr, by, win=WIN_S, keeper_rule=None, **kw):
    """{(row index, id): (team, gk)} - the same decide() rule on the reads of that id within +-win s of the row only:
    export ids slide from one player to another over a long life (the keeper's id ends on a dark SFK player: q30, q51),
    so a whole-match vote moves both."""
    out = {}; rule = keeper_rule or KEEPER_RULE
    for i, f in enumerate(fr):
        for p in f["players"]:
            a = by.get(p["id"]); t, gk = None, False
            if a:
                lo = bisect.bisect_left(a[0], f["t"] - win); hi = bisect.bisect_right(a[0], f["t"] + win)
                v = [0] * 5
                for c in a[1][lo:hi]: v[c] += 1
                t, gk = decide(v, **kw)
            x = (p.get("m") or [None])[0]
            if gk: t = keeper_team(f["t"], x, rule)
            elif t is None and p.get("gk") and rule == "end":     # the export's own keeper flag: team from his goal
                t, gk = keeper_team(f["t"], x, "end"), True
            if t is not None: out[(i, p["id"])] = (t, gk)
    return out


def apply(fr, rmap):
    """copy of the export rows with new team labels; referee rows get team None. -> (rows, row-level change counts)"""
    ch = Counter(); out = []
    keyed = bool(rmap) and isinstance(next(iter(rmap)), tuple)
    for i, f in enumerate(fr):
        ps = []
        for p in f["players"]:
            r = rmap.get((i, p["id"]) if keyed else p["id"])
            if r:
                t = None if r[0] == "ref" else r[0]
                if t != p["team"] or r[1] != bool(p.get("gk")): ch[f"{p['team']}{'k' if p.get('gk') else ''}->{t or 'ref'}{'k' if r[1] else ''}"] += 1
                p = dict(p, team=t, gk=r[1])
            ps.append(p)
        out.append(dict(f, players=ps))
    return out, ch


def recompute(fr, near_m=NEAR_M, smooth=SMOOTH):
    """possession_simple on the export rows (A / B / None per row)"""
    raw = []; last = None
    for f in fr:
        b = (f.get("ball") or {}).get("px")
        if not b or not f["players"]:
            raw.append(last); continue
        best = {}
        for p in f["players"]:
            if p.get("team") not in ("A", "B") or not p.get("px"): continue
            h = K.player_height_px(p["px"][1])
            d = ((p["px"][0] - b[0]) ** 2 + (p["px"][1] - b[1]) ** 2) ** 0.5 * BODY_M / h
            best[p["team"]] = min(d, best.get(p["team"], 1e9))
        if not best: raw.append(last); continue
        t = min(best, key=best.get); s = t if best[t] <= near_m else None
        raw.append(s); last = s
    out = []
    for k in range(len(raw)):
        w = Counter(raw[max(0, k - smooth):k + smooth + 1]); out.append(w.most_common(1)[0][0])
    return [dict(f, possession=s) for f, s in zip(fr, out)]


def flip(fr_old, fr_new, max_m=5.0):
    """like-for-like edit of the EXPORT's own possession (the re-computation above copies only ~60-70% of its rows: the
    export was made at the pipeline's frame rate from rows frames_lite does not keep): in each row with possession X,
    the player who made it X is the carrier if he is labelled X, else the nearest X player to the ball (within max_m,
    height ruler); possession becomes his NEW label (a referee -> loose). Rows with nobody to pin it on keep X."""
    out = []; n = Counter()
    for fo, fn in zip(fr_old, fr_new):
        s = fo.get("possession")
        if s not in ("A", "B"):
            out.append(fn); continue
        newt = {p["id"]: p["team"] for p in fn["players"]}
        cid = fo.get("carrier")
        who = None
        if cid is not None and any(p["id"] == cid and p["team"] == s for p in fo["players"]):
            who = cid
        else:
            b = (fo.get("ball") or {}).get("px"); best = max_m
            if b:
                for p in fo["players"]:
                    if p["team"] != s or not p.get("px"): continue
                    d = ((p["px"][0] - b[0]) ** 2 + (p["px"][1] - b[1]) ** 2) ** 0.5 * BODY_M / K.player_height_px(p["px"][1])
                    if d < best: best, who = d, p["id"]
        if who is None:
            n["kept_nobody"] += 1; out.append(dict(fn, possession=s)); continue
        t = newt.get(who, s)
        n["same" if t == s else f"{s}->{t or 'ref'}"] += 1
        out.append(dict(fn, possession=t))
    return out, n


def score(fr):
    g1 = json.load(open(G1))["items"]; g2 = json.load(open(G2))["items"]
    r1, i1 = analyse(fr, g1, half=H1); r2, i2 = analyse(fr, g2, half=H2)
    items = [x for x in i1 + i2 if x["eye"] in ("A", "B")]
    both = [x for x in items if x["ours"] in ("A", "B")]; win = [x for x in items if x["ours_win"] in ("A", "B")]
    k = sum(x["ours"] == "A" for x in both)
    return {"agree": sum(x["eye"] == x["ours"] for x in both), "agree_of": len(both),
            "agree_win": sum(x["eye"] == x["ours_win"] for x in win), "agree_win_of": len(win),
            "ours_sfk_pct_same_moments": round(100 * k / len(both)) if both else None,
            "misses": [{"t": x["t"], "file": x["file"], "eye": x["eye"], "ours": x["ours"]} for x in both if x["eye"] != x["ours"]],
            "sfk_pct_h1": export_share(fr, *H1), "sfk_pct_h2": export_share(fr, *H2),
            "sfk_pct_match": export_share(fr, H1[0], H1[1]) and round(
                100 * sum(f["possession"] == "A" for f in fr if inside(f["t"])) /
                max(1, sum(f["possession"] in ("A", "B") for f in fr if inside(f["t"]))), 1)}


def inside(t):
    return H1[0] <= t < H1[1] or H2[0] <= t < H2[1]


def agreement(a, b):
    n = sum(1 for x, y in zip(a, b) if x["possession"] in ("A", "B") or y["possession"] in ("A", "B"))
    k = sum(1 for x, y in zip(a, b) if (x["possession"] in ("A", "B") or y["possession"] in ("A", "B")) and x["possession"] == y["possession"])
    return round(100 * k / n, 1) if n else None


def run_variant(fr, votes, by, win, keeper_rule=None, **kw):
    rmap = window_labels(fr, by, win=win, keeper_rule=keeper_rule, **kw) if win > 0 else relabel_map(votes, **kw)
    new_fr, ch = apply(fr, rmap)
    fl, fln = flip(fr, new_fr)
    return new_fr, {"win_s": win, "keeper_rule": keeper_rule or KEEPER_RULE, "row_changes": dict(ch.most_common()), "export_flipped": score(fl), "flip_rows": dict(fln),
                    "recomputed": score(recompute(new_fr))}


def main(wins=(0, 2, 5, 10, 20), **kw):
    fr = json.load(gzip.open(FR_PATH))["frames"]
    votes = json.load(open(VOTES))["votes"]; by = reads_by_id()
    res = {"rule": {"red_share": RED_SHARE, "blue_share": BLUE_SHARE, "dom": DOM, "min_reads": MIN_READS, **kw},
           "tracks_voted": len(votes), "export": score(fr), "recomputed_old_labels": score(recompute(fr)),
           "recompute_copies_export_pct": agreement(fr, recompute(fr)), "variants": []}
    keep = None
    for w in wins:
        for kr in (("shirt", "end") if w > 0 else ("shirt",)):
            for field in ((True, False) if w > 0 else (True,)):
                new_fr, r = run_variant(fr, votes, by, w, keeper_rule=kr, field=field, **kw); r["field_players"] = field
                res["variants"].append(r)
                if w == WIN_S and kr == KEEPER_RULE and field == FIELD: keep = new_fr
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    return res, keep


def change_sheets(out=OUT, per_kind=100, win=None):
    """by-eye check: crops of every player whose label the chosen rule changes on the 150 full-size frames in the repo
    (nearest export row), one sheet per kind of change (old_to_new in the file name; the tile shows the frame name)."""
    import cv2, numpy as np
    win = WIN_S if win is None else win
    fr, ts = K.load_export(); votes = json.load(open(VOTES))["votes"]
    new_fr, _ = apply(fr, window_labels(fr, reads_by_id(), win=win, field=FIELD) if win > 0 else relabel_map(votes))
    tiles = {}
    for path, t in K.local_frames():
        img = cv2.imread(path)
        if img is None: continue
        j = K.nearest(ts, t); row = K.interp_row(fr, ts, t); newl = {p["id"]: p for p in new_fr[j]["players"]}
        for p in row["players"]:
            q = newl.get(p["id"])
            if not q or not p.get("px"): continue
            new = "ref" if q["team"] is None else q["team"] + ("k" if q.get("gk") else "")
            old = p["team"] + ("k" if p.get("gk") else "")
            if new == old: continue
            x, y = p["px"]; h = K.player_height_px(y)
            X0, X1 = int(max(0, x - 0.8 * h)), int(min(img.shape[1], x + 0.8 * h)); Y0, Y1 = int(max(0, y - 1.3 * h)), int(min(img.shape[0], y + 0.3 * h))
            if X1 - X0 < 8 or Y1 - Y0 < 8: continue
            c = cv2.resize(img[Y0:Y1, X0:X1].copy(), (80, 100))
            cv2.drawMarker(c, (int((x - X0) * 80 / (X1 - X0)), int((y - Y0) * 100 / (Y1 - Y0))), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 8, 1)
            cv2.putText(c, os.path.basename(path)[:3], (2, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            tiles.setdefault(f"{old}_to_{new}", []).append(c)
    os.makedirs(out, exist_ok=True); counts = {}
    for kind, ts_ in tiles.items():
        counts[kind] = len(ts_); ts_ = ts_[:per_kind]; cols = 10
        rows = [np.hstack(ts_[i:i + cols] + [np.zeros_like(ts_[0])] * (cols - len(ts_[i:i + cols]))) for i in range(0, len(ts_), cols)]
        cv2.imwrite(os.path.join(out, f"changed_{kind}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps(counts)); return counts


if __name__ == "__main__":
    if sys.argv[1:2] == ["sheets"]:
        change_sheets()
    else:
        r, _ = main()
        print(json.dumps({k: v for k, v in r.items() if "misses" not in str(k)}, indent=1)[:3000])
