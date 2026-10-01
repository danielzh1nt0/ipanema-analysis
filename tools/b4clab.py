"""B4c (1 Oct): spare-ball rule only while play is on (ball.pick_v2 recur_play_s). Grid over the look-back time and the
moving-ball threshold, graded like B4b on the 34 moments, the B4 key (321), the 99 who-has-the-ball key and the AIK 39-ball key, on the exact
app inputs of both clips (picker_inputs.pkl) with the picker tuned on 1 Oct. Free, local.
Writes results/ball/b4c/b4clab.json."""
import sys, os, json, gzip, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL
M = "SFKBP1109_s1200"; OUT = "results/ball/b4c"; os.makedirs(OUT, exist_ok=True)
# exact app inputs of the clip (saved by the app run, same as tools/picktune.py), picker at today's tuned defaults
def load(m):
    P = pickle.load(open(f"results/volume/cache/{m}/picker_inputs.pkl", "rb"))
    P["per"] = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in P["per"].items()}
    return P
PS, PA = load(M), load("p15u-vs-aik-2026-09-21-bd09_s2520")
cd, H, L, W, fps, per = PS["cands"], PS["H"], PS["L"], PS["W"], PS["fps"], PS["per"]; n = len(H)
gt = {int(k): v[:2] for k, v in json.load(open(f"results/volume/reference/{M}/ball_gt.json")).items() if v}
agt = {int(k): v for k, v in json.load(open("reference/p15u-vs-aik-2026-09-21-bd09_s2520/ball_gt.json")).items()}
def to_m(Hk, px): v = np.linalg.inv(Hk) @ np.array([px[0], px[1], 1.0]); return v[:2] / v[2]
key = json.load(open("results/ball/b4/key.json"))["moments"]
on = lambda b, f, x, y: f in b and np.hypot(b[f][0] - x, b[f][1] - y) <= 30
spare_m = np.median([to_m(H[m["frame"]], (m["x"], m["y"])) for m in key if m["verdict"] == "spare_ball"], axis=0)
def grade(b):
    balls = [m for m in key if m["verdict"] == "ball"]; restart = [m for m in balls if 240 <= m["id"] <= 252]
    spare = [m for m in key if m["verdict"] == "spare_ball"]
    # frames where the picker sits on the spare ball (within 2 m of its spot), whole clip
    on_spare = sum(1 for f, p in b.items() if np.hypot(*(to_m(H[f], p) - spare_m)) <= 2.0)
    return {k: (int(v) if isinstance(v, (np.integer, np.bool_)) else v) for k, v in {"moments34": sum(on(b, i, *g) for i, g in gt.items()), "b4_ball": sum(on(b, m["frame"], m["x"], m["y"]) for m in balls),
            "b4_ball_of": len(balls), "restart_kept": sum(on(b, m["frame"], m["x"], m["y"]) for m in restart), "restart_of": len(restart),
            "spare_still_picked": sum(on(b, m["frame"], m["x"], m["y"]) for m in spare), "spare_of": len(spare),
            "frames_on_spare_spot_s": round(on_spare / fps, 1), "frames_with_ball": len(b)}.items()}
res = {}
E4 = {"__file__": os.path.abspath("tools/e4lab.py")}; exec(compile(open("tools/e4lab.py").read().split("\nif __name__")[0], "e4lab", "exec"), E4)
c = dict(name="SFK-BP", per={k: per.get(k, []) for k in range(n)}, H=[H[k] for k in range(n)], L=L, W=W, fps=fps, boxh=None,
         key=json.load(open("results/review/who_answers.json"))["moments"])
def state(b):
    c["ball"] = b; fr_, bm_ = E4["P"].carriers(c["per"], b, c["H"], 2.5, 5.0)
    return E4["P"].pipeline_state(c["per"], b, bm_, c["H"], fps, L, W, mode="simple", log=lambda *a: None)[0]
def who(b):
    c["ball"] = b; s = E4["stats"](c, "simple")
    return {k: s[k] for k in ("who_has_ball_right", "dead_pct", "restarts", "sequences", "passes")}
STOP = (248.0, 256.5)   # s: match ball out behind the goal, keeper takes the ball at the post for the goal kick (B4b sheets #25-39)
def split(b):
    sp = [m for m in key if m["verdict"] == "spare_ball"]; st = lambda m: STOP[0] <= m["frame"] / fps <= STOP[1]
    return {"spare_in_play_picked": sum(on(b, m["frame"], m["x"], m["y"]) for m in sp if not st(m)), "spare_in_play_of": sum(not st(m) for m in sp),
            "stoppage_ball_followed": sum(on(b, m["frame"], m["x"], m["y"]) for m in sp if st(m)), "stoppage_of": sum(st(m) for m in sp)}
def pick(**kw): return BL.bridge(BL.pick_v2(cd, H, L, W, per=per, fps=fps, log=lambda *a: None, **kw), fps)
def aik(**kw):
    b = BL.bridge(BL.pick_v2(PA["cands"], PA["H"], PA["L"], PA["W"], per=PA["per"], fps=PA["fps"], log=lambda *a: None, **kw), PA["fps"])
    return {"aik39": int(sum(on(b, f, *g) for f, g in agt.items()))}
res = {"grid": {}}
runs = {"off": {}, "B4b (rule always)": dict(recur_r=2.0)}
for T in (3.0, 4.0, 5.0):
    for mv in (0.6, 1.0, 1.5):
        for ins in (-1.5, 2.0):   # -1.5 = the pitch plus the usual 1.5 m margin; 2 / 4 = that far inside the lines
            runs[f"play {T} s, move {mv} m, inside {ins} m"] = dict(recur_r=2.0, recur_play_s=T, recur_move_m=mv, recur_inside_m=ins)
B = {}
for nm, kw in runs.items():
    b = B[nm] = pick(**kw); g = grade(b); g.update(split(b)); g.update(who(b)); g.update(aik(**kw)); res["grid"][nm] = g
    print(nm, g, flush=True)
    json.dump(res, open(f"{OUT}/b4clab.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
# picture check without a new free-runner job: every tile of the B4b old/new sheets (results/free/b4b/) is matched to the
# pick of the chosen setting (old = same as no rule, new = same as the rule always on)
CHOSEN = "play 5.0 s, move 1.0 m, inside 2.0 m"
chk = json.load(open("results/ball/b4b/check_moments.json"))["moments"]; T = {}
for m in chk: T.setdefault(m["id"], {"frame": m["frame"]})[m["pick"]] = (m["x"], m["y"])
b = B[CHOSEN]; tiles = {}
for k, t in sorted(T.items()):
    p = b.get(t["frame"]); near = lambda q: q is not None and p is not None and np.hypot(p[0] - q[0], p[1] - q[1]) <= 30
    tiles[k] = {"s": round(t["frame"] / fps, 1), "same_as": "old" if near(t.get("old")) else ("new" if near(t.get("new")) else "other")}
res["chosen"] = CHOSEN; res["b4b_tiles"] = tiles
print(CHOSEN, " ".join(f"{k}:{v['same_as']}" for k, v in tiles.items()))
json.dump(res, open(f"{OUT}/b4clab.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
# where the chosen setting and no rule disagree (> 30 px): one check moment every 0.5 s, old and new pick side by side,
# for picture sheets on the free runner (tools/b4c_sheet.py); and which who-has-the-ball moments flip
b0, b1 = B["off"], B[CHOSEN]
diff = [f for f in range(n) if (f in b0) != (f in b1) or (f in b0 and np.hypot(b0[f][0] - b1[f][0], b0[f][1] - b1[f][1]) > 30)]
segs = []
for f in diff:
    if segs and f - segs[-1][1] <= 3: segs[-1][1] = f
    else: segs.append([f, f])
chk = []
for a, z in segs:
    for f in range(a, z + 1, 15):
        k = len({m["frame"] for m in chk})
        for tag, b in (("old", b0), ("new", b1)):
            if f in b: chk.append({"id": k, "frame": f, "x": round(float(b[f][0]), 1), "y": round(float(b[f][1]), 1), "pick": tag})
res["differ"] = {"frames": len(diff), "seconds": round(len(diff) / fps, 1), "stretches": [[round(a / fps, 1), round(z / fps, 1)] for a, z in segs]}
json.dump({"clip": M, "moments": chk}, open(f"{OUT}/check_moments.json", "w"), indent=0)
nm_ = {0: "dark", 1: "white", 2: "loose"}; s0, s1 = state(b0), state(b1)
res["who_flips"] = [{"frame": a["frame"], "s": round(a["frame"] / fps, 1), "truth": a["truth"], "off": nm_.get(int(s0[a["frame"]])),
                     "new": nm_.get(int(s1[a["frame"]]))} for a in c["key"] if a["frame"] < n and s0[a["frame"]] != s1[a["frame"]]]
res["moments34_changed"] = [{"frame": i, "s": round(i / fps, 1), "was_right": bool(on(b0, i, *g))} for i, g in gt.items() if on(b0, i, *g) != on(b1, i, *g)]
print(res["differ"], len(chk), "tiles"); print(res["who_flips"]); print(res["moments34_changed"])
json.dump(res, open(f"{OUT}/b4clab.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
