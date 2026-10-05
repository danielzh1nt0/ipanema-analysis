"""K3c v2 (5 Oct): per-moment team decisions from the Kaggle re-label readings (results/kaggle/vall_relabel/relabel/
labels.jsonl: [t, id, label, red share, dark share, box height]). A track that switches between two people (one SFK, one
Vallentuna) gets pieces: readings smoothed over a 7-reading window (~1.4 s at 5 per s), runs shorter than MIN_RUN_S
merged into the neighbour, one team per run. Tracks with < 3 readings keep the whole-track decision.
    python tools/k3c_segments.py [labels.jsonl] [out.json]"""
import sys, json, collections
IN = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle/vall_relabel/relabel/labels.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "overrides/p15u-vs-vallentuna-2026-10-03-6cce_teams.json"
WIN, MIN_RUN_S, GAP_S = 7, 3.0, 1.0

def runs_for(reads):
    """reads: sorted [(t, label)] with label A/B -> [[t0, t1, team]]"""
    labs = [l for _, l in reads]; sm = []
    for i in range(len(reads)):
        w = labs[max(0, i - WIN // 2):i + WIN // 2 + 1]; sm.append("A" if w.count("A") >= w.count("B") else "B")
    runs = []
    for (t, _), s in zip(reads, sm):
        if runs and runs[-1][2] == s: runs[-1][1] = t
        else: runs.append([t, t, s])
    changed = True
    while changed and len(runs) > 1:                     # merge short runs into the longer neighbour
        changed = False
        for i, r in enumerate(runs):
            if r[1] - r[0] < MIN_RUN_S:
                nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]; j = max(nb, key=lambda j: runs[j][1] - runs[j][0])
                runs[j][0], runs[j][1] = min(runs[j][0], r[0]), max(runs[j][1], r[1]); del runs[i]; changed = True; break
        merged = []
        for r in runs:
            if merged and merged[-1][2] == r[2]: merged[-1][1] = r[1]
            else: merged.append(r)
        runs = merged
    # widen to cover the time between readings (half way to the next run), so every exported frame falls in a run
    for a, b in zip(runs, runs[1:]): mid = (a[1] + b[0]) / 2; a[1], b[0] = mid, mid
    runs[0][0] -= GAP_S; runs[-1][1] += GAP_S
    return [[round(a, 2), round(b, 2), t] for a, b, t in runs]

def main():
    by = collections.defaultdict(list); kc = collections.Counter(); allc = collections.Counter()
    for line in open(IN):
        t, pid, lab = json.loads(line)[:3]; allc[str(pid)] += 1
        if lab in ("A", "B"): by[str(pid)].append((t, lab))
        elif lab == "K": kc[str(pid)] += 1
    old = json.load(open(OUT)) if OUT.endswith(".json") else {}
    out, split = dict(old), 0
    for pid, reads in by.items():
        if len(reads) < 3: continue
        reads.sort(); r = runs_for(reads)
        if len(r) == 1: out[pid] = r[0][2]
        else: out[pid] = r; split += 1
    neither = [p for p in allc if allc[p] >= 10 and kc[p] >= 0.7 * allc[p]]      # referee / staff in white: not a player
    for p in neither: out[p] = "K"
    json.dump(out, open(OUT, "w")); print(f"{len(by)} tracks with readings, {split} split into pieces, {len(neither)} not players, {len(out)} in the override")

if __name__ == "__main__":
    main()
