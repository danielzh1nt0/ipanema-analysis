"""K4 (9 Oct): overrides/<match>_extras.json from results/kaggle/vall_extras/extras/extras.jsonl. Each line is one person the
export was missing on one exported frame (10 per s): [t, team, feet_x, feet_y, m_x, m_y, box]. Linked into short tracks
(same team, within 1.2 m of the previous detection < 0.35 s earlier, nearest wins) so clean()/fill_gaps() treat them like
any other player; tracks shorter than MIN_S are noise and dropped.   python tools/make_extras.py [match]"""
import sys, json, os, math, collections
M = sys.argv[1] if len(sys.argv) > 1 else "p15u-vs-vallentuna-2026-10-03-6cce"
SRC = os.environ.get("SRC", "results/kaggle/vall_extras/extras/extras.jsonl"); MIN_S = float(os.environ.get("MIN_S", "0.5")); ID0 = 9_000_000

def link(rows, gap_s=0.35, max_m=1.2):
    """rows sorted by t -> rows with an id each"""
    out = []; active = []   # (t, mx, my, team, id)
    nid = ID0
    for t, team, fx, fy, mx, my, box in rows:
        best = None
        for a in active:
            if a[3] != team or t - a[0] > gap_s or t - a[0] <= 0: continue
            d = math.hypot(mx - a[1], my - a[2])
            if d <= max_m and (best is None or d < best[0]): best = (d, a)
        if best: i = best[1][4]; active.remove(best[1])
        else: i = nid; nid += 1
        active.append((t, mx, my, team, i)); active = [a for a in active if t - a[0] <= gap_s]
        out.append({"t": t, "id": i, "team": team, "px": [fx, fy], "m": [mx, my], "box": box})
    return out

def main():
    rows = sorted(json.loads(l) for l in open(SRC) if l.strip())
    linked = link(rows); life = collections.defaultdict(list)
    for r in linked: life[r["id"]].append(r["t"])
    keep = {i for i, ts in life.items() if ts[-1] - ts[0] >= MIN_S}
    linked = [r for r in linked if r["id"] in keep]
    by_team = collections.Counter(r["team"] for r in linked)
    json.dump({"match": M, "source": SRC, "min_track_s": MIN_S, "rows": linked}, open(f"overrides/{M}_extras.json", "w"))
    print(f"{len(rows)} detections -> {len(linked)} rows in {len(keep)} tracks (>= {MIN_S} s); by team {dict(by_team)}")

if __name__ == "__main__":
    main()
