"""E2 (28 Sep, worker): possession_simple settings chosen on who-keys 1+2 only, then scored on key 3 (frames chosen blind,
graded without seeing model output). Also lists key-3 misses. Local, free. Run with PYTHONPATH=."""
exec(open("tools/wholab.py").read().split("A = json.load(")[0].replace('print(', '(lambda *a, **k: None)('))
A = json.load(open("results/review/who_answers.json"))["moments"]
W = {"dark": 0, "white": 1, "loose": 2}
def grade(st, keys):
    s = [int(st[a["frame"]] == W[a["truth"]]) for a in A if a["truth"] != "unsure" and a["n"] // 100 + 1 in keys]
    return sum(s), len(s)
res = []
for near in (1.0, 1.25, 1.5, 2.0, 2.5):
    for sm in (0, 3, 6, 9, 12):
        st = P.possession_simple(per, ball, Hd, n, near_m=near, smooth=sm); res.append((grade(st, {1, 2}), grade(st, {3}), near, sm))
res.sort(key=lambda r: -r[0][0])
for r in res[:8]: print(f"near {r[2]} m, smooth {r[3]}: keys 1+2 {r[0][0]}/{r[0][1]}, key 3 {r[1][0]}/{r[1][1]}")
best = res[0]; dflt = [r for r in res if r[2] == 1.5 and r[3] == 6][0]
print(f"chosen on 1+2: near {best[2]}, smooth {best[3]} -> key 3 {best[1][0]}/{best[1][1]}; default (1.5, 6) -> key 3 {dflt[1][0]}/{dflt[1][1]}")
st = P.possession_simple(per, ball, Hd, n); N = {0: "dark", 1: "white", 2: "loose"}
for a in A:
    if a["n"] >= 200 and a["truth"] != "unsure" and st[a["frame"]] != W[a["truth"]]: print(f"  key-3 miss #{a['n'] - 200} t={a['t']}s: truth {a['truth']}, model {N[int(st[a['frame']])]}")
