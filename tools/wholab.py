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
for sw in ():
    st, _ = P.viterbi(per, P.clean_ball(dict(bm), L, W), fps, L, W, speed_win_s=sw); st = np.asarray(st)
    live = np.isin(st, [0, 1]).sum()
    ar = P.direction_from_keepers(per, L, log=q) or {"A": True, "B": False}
    tv = P.turnovers(per, frames_, st, bm, fps, ar, min_before_s=1.0, min_after_s=1.0)
    tv3 = P.turnovers(per, frames_, st, bm, fps, ar)
    print(f"speed window {sw}: states {dict(sorted(collections.Counter(st.tolist()).items()))} share loose {100*(st==2).mean():.0f}%, possession A {100*(st==0).sum()/max(1,live):.0f}%, turnovers (3 s rule) {len(tv3)}, (1 s rule) {len(tv)}")
A = json.load(open("results/review/who_answers.json"))["moments"]
new_team_dark = "A"
for sw in ():
    st, _ = P.viterbi(per, P.clean_ball(dict(bm), L, W), fps, L, W, speed_win_s=sw); st = np.asarray(st)
    tm = ok_t = lo = ok_l = 0
    for a in A:
        s = int(st[a["frame"]]) if a["frame"] < len(st) else None
        if a["truth"] in ("white", "dark"):
            tm += 1; want = 0 if a["truth"] == "dark" else 1; ok_t += (s == want)
        elif a["truth"] == "loose": lo += 1; ok_l += (s == 2)
    print(f"speed window {sw}: team moments right {ok_t}/{tm}, loose moments right {ok_l}/{lo}")
st, _ = P.viterbi(per, P.clean_ball(dict(bm), L, W), fps, L, W); st = np.asarray(st)
from collections import Counter
print("team per kit check: rows at frame 0", Counter(r[1] for r in per[0]))
for a in A:
    if a["truth"] == "unsure": continue
    k = a["frame"]; s = int(st[k]); f = frames_[k]
    print(a["n"], a["truth"], "state", P.STATES[s], "| carrier team", f.get("team"), "| ball seen", k in bm, "| states +-0.5s", "".join(P.STATES[x][0] for x in st[k-15:k+16:5]))
