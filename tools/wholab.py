"""E1 (28 Sep): possession model on the real SFK-BP clip vs the who-has-the-ball answer key graded by Claude (results/review/who_answers.json). Free, local. Run with PYTHONPATH=."""
exec(open("/home/claude/ipanema-analysis/tools/fusedlab.py").read().split("q = lambda *a: None")[0].split('"""', 2)[2])
import pickle, collections
from ipanema import possession as P
q = lambda *a: None
C = "results/volume/cache/SFKBP1109_s1200/"
def load(f):
    o = pickle.load(open(C + f, "rb")); o = o[0] if isinstance(o, tuple) else o
    return {int(k): [(float(a), float(b), float(c)) for a, b, c in v] for k, v in o.items()}
cd = BL.fuse_candidates(load("ball_cands_clicks_round_20260928_0108.pkl"), load("ball_cands_wasb_1790008894_t2x2_thr0.05.pkl"))
ball = BL.bridge(BL.pick_v2(cd, H, L, W, per=new, fps=fps, log=q), fps)
per = {k: new.get(k, []) for k in range(n)}
Hd = {k: H[k] for k in range(n)}
frames_, bm = P.carriers(per, ball, Hd)
jit = [np.linalg.norm(bm[k] - bm[k - 1]) for k in bm if k - 1 in bm]
print("real clip: frame-to-frame ball step median m", round(float(np.median(jit)), 2), "-> x fps =", round(float(np.median(jit)) * fps, 1), "m/s")
for sw in (None,):
    st, _ = P.viterbi(per, P.clean_ball(dict(bm), L, W), fps, L, W, speed_win_s=sw); st = np.asarray(st)
    live = np.isin(st, [0, 1]).sum()
    ar = P.direction_from_keepers(per, L, log=q) or {"A": True, "B": False}
    tv = P.turnovers(per, frames_, st, bm, fps, ar, min_before_s=1.0, min_after_s=1.0)
    tv3 = P.turnovers(per, frames_, st, bm, fps, ar)
    print(f"speed window {sw}: states {dict(sorted(collections.Counter(st.tolist()).items()))} share loose {100*(st==2).mean():.0f}%, possession A {100*(st==0).sum()/max(1,live):.0f}%, turnovers (3 s rule) {len(tv3)}, (1 s rule) {len(tv)}")
A = json.load(open("results/review/who_answers.json"))["moments"]
new_team_dark = "A"
PD = P.pixel_dist(per, ball, Hd)
for sw, dd in ((None, None), (0.6, None), (None, PD), (0.6, PD), (-0.6, PD)):
    st, _ = P.viterbi(per, P.clean_ball(dict(bm), L, W), fps, L, W, speed_win_s=sw, dist=dd); st = np.asarray(st)
    tm = ok_t = lo = ok_l = 0; half = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    for a in A:
        s = int(st[a["frame"]]) if a["frame"] < len(st) else None
        if a["truth"] in ("white", "dark"):
            tm += 1; want = 0 if a["truth"] == "dark" else 1; ok_t += (s == want); r = s == want
        elif a["truth"] == "loose": lo += 1; ok_l += (s == 2); r = s == 2
        else: continue
        h = half[a["n"] // 100 + 1]; h[0] += r; h[1] += 1
    print(f"speed window {sw}, {'pixel' if dd else 'metre'} distances: team moments right {ok_t}/{tm}, loose moments right {ok_l}/{lo} | all {ok_t + ok_l}/{tm + lo} (first key {half[1][0]}/{half[1][1]}, second key {half[2][0]}/{half[2][1]}, third key {half[3][0]}/{half[3][1]})")
st = P.possession_simple(per, ball, Hd, n)
r = {"team": [0, 0], "loose": [0, 0]}; kb = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
for a in A:
    if a["truth"] == "unsure": continue
    want = {"dark": 0, "white": 1, "loose": 2}[a["truth"]]; key = "loose" if want == 2 else "team"
    ok = int(st[a["frame"]] == want); r[key][0] += ok; r[key][1] += 1; kb[a["n"] // 100 + 1][0] += ok; kb[a["n"] // 100 + 1][1] += 1
print(f"possession_simple: team moments right {r['team'][0]}/{r['team'][1]}, loose moments right {r['loose'][0]}/{r['loose'][1]}; possession dark {100 * (st == 0).sum() / max(1, np.isin(st, [0, 1]).sum()):.0f}%, loose {100 * (st == 2).mean():.0f}% of the clip; keys 1/2/3: {kb[1][0]}/{kb[1][1]}, {kb[2][0]}/{kb[2][1]}, {kb[3][0]}/{kb[3][1]}")
