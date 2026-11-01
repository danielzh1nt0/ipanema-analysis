"""N0-01b (1 Nov, local, $0): passes whose passer is on the OTHER team in the players list (stats.players[].team, the
track's team over the whole match) than the pass itself (stats.passes[].team, the shirt read in that frame). Picks a
sample of those plus agreeing passes as controls, and for each one the passer's spot and one confident player of each
team in the same frame as references. The references are shown as '1' and '2' in random order, so the sheets are
blind; the key (which number is A, which pass is a control) goes to a separate file.

    python tools/n0_01b_moments.py   -> results/qa/n0-01b/moments.json + moments_key.json
"""
import os, sys, json, random, bisect, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n0_30d_clips import load

MATCHES = {"SFKBP1109": 12, "p15u-vs-vallentuna-2026-10-03-6cce": 18}   # mismatches per match. AIK: most frame files not in the repo;
# the SFK-BP clip (SFKBP1109_s1200): its stats.json and match_data.json come from different runs (38 of 273 ids shared)
N_CTRL = {"SFKBP1109": 5, "p15u-vs-vallentuna-2026-10-03-6cce": 6}


def frame_at(fr, ts, t, tol=0.2):
    i = bisect.bisect_left(ts, t)
    j = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t), default=None)
    return fr[j] if j is not None and abs(ts[j] - t) <= tol else None


def track_teams(fr):
    c = collections.defaultdict(collections.Counter)
    for f in fr:
        for q in f["players"]: c[q["id"]][q["team"]] += 1
    return c


def refs(f, passer, tt, team, k=1):
    """the nearest player of `team` to the passer whose track never changes team, observed, not a keeper"""
    px = next(q["px"] for q in f["players"] if q["id"] == passer)
    c = [q for q in f["players"] if q["id"] != passer and q["team"] == team and not q.get("gk") and q.get("state") == "observed"
         and len(tt[q["id"]]) == 1 and q.get("px")]
    c.sort(key=lambda q: (q["px"][0] - px[0]) ** 2 + (q["px"][1] - px[1]) ** 2)
    return c[:k]


def build(seed=1):
    rnd = random.Random(seed); out, key = [], {}
    for mid, nmis in MATCHES.items():
        st = json.load(open(f"results/volume/runs/matches/{mid}/stats.json")); _, fr = load(f"results/volume/runs/matches/{mid}/match_data.json")
        ts = [f["t"] for f in fr]; tt = track_teams(fr); ptm = {p["id"]: p["team"] for p in st["players"]}
        mis, agree = [], []
        for i, p in enumerate(st["passes"]):
            if p["from"] not in ptm: continue
            f = frame_at(fr, ts, p["t"])
            if f is None or not any(q["id"] == p["from"] and q.get("px") for q in f["players"]): continue
            ra, rb = refs(f, p["from"], tt, "A"), refs(f, p["from"], tt, "B")
            if not ra or not rb: continue
            q = next(q for q in f["players"] if q["id"] == p["from"])
            row = {"match": mid, "i": i, "t": f["t"], "passer": p["from"], "passer_px": q["px"], "ref": {"A": ra[0]["px"], "B": rb[0]["px"]},
                   "pass_team": p["team"], "player_team": ptm[p["from"]], "frame_team": q["team"], "track_teams": dict(tt[p["from"]])}
            (mis if ptm[p["from"]] != p["team"] else agree).append(row)
        pick = rnd.sample(mis, min(nmis, len(mis))) + rnd.sample(agree, min(N_CTRL[mid], len(agree)))
        print(mid, "mismatches usable", len(mis), "agree", len(agree), "picked", len(pick))
        for r in pick:
            one = rnd.choice("AB"); r["one_is"] = one; out.append(r)
    rnd.shuffle(out)
    for n, r in enumerate(out):
        r["id"] = f"p{n:02d}"; key[r["id"]] = {k: r[k] for k in ("match", "i", "t", "passer", "pass_team", "player_team", "frame_team", "track_teams", "one_is")}
    blind = [{"id": r["id"], "match": r["match"], "t": r["t"], "passer_px": r["passer_px"],
              "ref1_px": r["ref"][r["one_is"]], "ref2_px": r["ref"]["B" if r["one_is"] == "A" else "A"]} for r in out]
    return blind, key


if __name__ == "__main__":
    b, k = build(); os.makedirs("results/qa/n0-01b", exist_ok=True)
    json.dump(b, open("results/qa/n0-01b/moments.json", "w"), indent=1); json.dump(k, open("results/qa/n0-01b/moments_key.json", "w"), indent=1)
    print(len(b), "moments")
