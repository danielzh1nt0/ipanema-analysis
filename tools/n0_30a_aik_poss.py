"""N0-30a (30 Oct, worker, local, $0): AIK whole-match possession vs a blind by-eye sample.

The 25 Oct accuracy status says AIK's possession (46% SFK in the Kaggle join) is "not trusted yet" and asks for a
blind by-eye sample. results/kaggle/aik_poss_sample/poss has 60 random first-half frames (seed 7, nothing drawn);
they were graded by eye into results/qa/n0-30a/aik_poss_grades.json (A = dark SFK, B = white AIK, null = can't tell).

    python tools/n0_30a_aik_poss.py [frames_lite.json.gz] [grades.json] [out.json]

Prints / writes: the by-eye SFK share (all decided, sure only, in play only), the export's possession at the same
moments (nearest row and a +-1 s majority), agreement on the decided moments, and the export's own first-half share.
"""
import sys, json, gzip, bisect, math

DEF_FR = "results/kaggle/full_match_52d795-join/frames_lite.json.gz"
DEF_G = "results/qa/n0-30a/aik_poss_grades.json"


def nearest(ts, t):
    j = bisect.bisect_left(ts, t)
    return min([k for k in (j - 1, j) if 0 <= k < len(ts)], key=lambda k: abs(ts[k] - t))


def window_owner(fr, ts, t, half=1.0):
    """Majority A/B of the export's possession within +-half s (None if no A/B row or a tie)."""
    a = bisect.bisect_left(ts, t - half); b = bisect.bisect_right(ts, t + half)
    n = {"A": 0, "B": 0}
    for f in fr[a:b]:
        if f.get("possession") in n: n[f["possession"]] += 1
    if n["A"] == n["B"]: return None
    return "A" if n["A"] > n["B"] else "B"


def share(items, key="eye"):
    d = [x[key] for x in items if x[key] in ("A", "B")]
    return (sum(1 for v in d if v == "A"), len(d))


def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (c - h)), round(100 * (c + h)))


def export_share(fr, a, b):
    n = {"A": 0, "B": 0}
    for f in fr:
        if a <= f["t"] < b and f.get("possession") in n: n[f["possession"]] += 1
    tot = n["A"] + n["B"]
    return round(100 * n["A"] / tot, 1) if tot else None


def analyse(fr, grades, half=(366.0, 3225.0)):
    ts = [f["t"] for f in fr]
    items = []
    for g in grades:
        j = nearest(ts, g["t"]); f = fr[j]
        items.append(dict(g, ours=f.get("possession"), phase=f.get("phase"), ours_win=window_owner(fr, ts, g["t"])))
    res = {}
    for name, sel in (("all_decided", lambda x: True), ("sure_only", lambda x: x["sure"]), ("in_play_only", lambda x: not x["dead"])):
        sub = [x for x in items if sel(x)]
        k, n = share(sub)
        both = [x for x in sub if x["eye"] in ("A", "B") and x["ours"] in ("A", "B")]
        bothw = [x for x in sub if x["eye"] in ("A", "B") and x["ours_win"] in ("A", "B")]
        ko, no = share([x for x in sub if x["eye"] in ("A", "B")], "ours")
        res[name] = {"eye_sfk": k, "eye_decided": n, "eye_sfk_pct": round(100 * k / n) if n else None, "eye_ci95": wilson(k, n),
                     "ours_sfk_pct_same_moments": round(100 * ko / no) if no else None, "ours_decided_same_moments": no,
                     "agree": sum(1 for x in both if x["eye"] == x["ours"]), "agree_of": len(both),
                     "agree_win": sum(1 for x in bothw if x["eye"] == x["ours_win"]), "agree_win_of": len(bothw)}
    res["export_first_half_sfk_pct"] = export_share(fr, *half)
    res["export_whole_match_sfk_pct"] = export_share(fr, 0, 1e9)
    res["eye_undecided"] = sum(1 for x in items if x["eye"] is None)
    return res, items


def main():
    a = sys.argv[1:] + [None] * 3
    fr = json.load(gzip.open(a[0] or DEF_FR))["frames"]
    G = json.load(open(a[1] or DEF_G))
    res, items = analyse(fr, G["items"])
    print(json.dumps(res, indent=1))
    for x in items:
        if x["eye"] in ("A", "B") and x["ours"] in ("A", "B") and x["eye"] != x["ours"]:
            print("disagree", x["file"], x["t"], "eye", x["eye"], "ours", x["ours"], x["phase"], "|", x["note"])
    if a[2]: json.dump({"result": res, "items": items}, open(a[2], "w"), indent=1)


if __name__ == "__main__": main()
