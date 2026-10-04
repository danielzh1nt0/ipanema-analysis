"""PC1 (4 Oct): passes and possession from PLAYER MOVEMENT ONLY with PathCRF (Kim et al., KDD 2026, MPL-2.0,
https://github.com/hyunsungkim-ds/pathcrf), on Metrica's free sample games 1-2, compared with Metrica's own labels and
with OUR ball-based stats (tools/metricalab.py, clean and noisy ball).

PathCRF never sees the ball: per frame it picks one edge of the player graph (player keeps the ball, or ball travels
from player A to player B / out). It only needs a dead/alive flag to cut the match into stretches of play; here that
flag is "Metrica has a ball position inside the pitch" (in our product it would come from the stoppage detector).
The released model (trial 140, dynamic masked CRF) was trained on 7 Bundesliga matches (Sportec open data), so Metrica
is unseen data for it.

    python tools/pc1lab.py [--games 1,2] [--minutes N] [--noise 0.0] [--view 35]  -> results/pathcrf/pc1_<date>.json

Needs: torch (CPU is enough), torch_geometric, scipy, pandas, pyarrow; PathCRF cloned to /tmp/pathcrf (done here).
"""
import os, sys, json, re, subprocess, argparse, datetime
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
PCRF, PCRF_SHA = "/tmp/pathcrf", "1b7c6f85a6b78c92e4543fde72ca2b86a933f997"
METRICA = "/tmp/metrica/data"
L, W, FPS = 105.0, 68.0, 25


def ensure_repos():
    if not os.path.exists(METRICA):
        subprocess.run("git clone -q --depth 1 https://github.com/metrica-sports/sample-data.git /tmp/metrica", shell=True, check=True)
    if not os.path.exists(PCRF):
        subprocess.run(f"git clone -q https://github.com/hyunsungkim-ds/pathcrf.git {PCRF} && cd {PCRF} && git checkout -q {PCRF_SHA}", shell=True, check=True)


def _read_team(path, team):
    df = pd.read_csv(path, skiprows=2)
    cols = list(df.columns); out = {"period_id": df["Period"].astype(int).values, "frame_raw": df["Frame"].astype(int).values}
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            n = int(c[6:]); out[f"{team}_{n}_x"] = df.iloc[:, i].astype(float).values * L
            out[f"{team}_{n}_y"] = df.iloc[:, i + 1].astype(float).values * W
    out["ball_x"] = df.iloc[:, -2].astype(float).values * L; out["ball_y"] = df.iloc[:, -1].astype(float).values * W
    return pd.DataFrame(out)


def alive_mask(bx, by, margin=0.0):
    """dead/alive flag: Metrica gives a ball position (inside the pitch) only while the ball is in play"""
    bx = np.asarray(bx, float); by = np.asarray(by, float)
    ok = np.isfinite(bx) & np.isfinite(by)
    return ok & (bx >= -margin) & (bx <= L + margin) & (by >= -margin) & (by <= W + margin)


def label_episodes(alive, period, min_len=5 * FPS):
    """1, 2, ... for each run of alive frames (gaps < 5 frames joined, runs < min_len dropped), 0 = dead"""
    ep = np.zeros(len(alive), int); cur = 0; i = 0; n = len(alive)
    while i < n:
        if not alive[i]: i += 1; continue
        j = i
        while j < n and period[j] == period[i] and (alive[j] or (j + 5 < n and alive[j:j + 5].any())): j += 1
        while j > i and not alive[j - 1]: j -= 1
        if j - i >= min_len: cur += 1; ep[i:j] = cur
        i = max(j, i + 1)
    return ep


def follow_cam(t, view, pan_s=2.0):
    """Stand-in for our Veo follow-cam (F2b: ~30% of the pitch in view, 13-17 of 22 players): the camera centre follows
    the ball along the pitch, smoothed over pan_s seconds (held when the ball is unknown). Players further than view/2
    metres from the centre along the pitch are hidden, then filled by straight lines between sightings (held at the
    ends), which is the best a gap filler can do without seeing them. Returns a copy; t['seen_share'] = share seen."""
    t = t.copy(); out = []
    for _, x in t.groupby("period_id", sort=True):
        c = x["ball_x"].ffill().bfill().fillna(L / 2).rolling(int(pan_s * FPS), center=True, min_periods=1).mean().values
        seen = []
        for col in [c_ for c_ in x.columns if re.match(r"(home|away)_\d+_x$", c_)]:
            px = x[col].values; on = np.isfinite(px); vis = on & (np.abs(px - c) <= view / 2); seen.append(vis[on].mean() if on.any() else np.nan)
            if on.any() and not vis.any(): vis = on.copy()                   # never in view in this stretch: keep him (rare; PathCRF needs 22)
            for ax in ("x", "y"):
                s = x[col[:-1] + ax].where(vis)
                s = s.interpolate(limit_area="inside").ffill().bfill()
                x[col[:-1] + ax] = np.where(on, s.values, np.nan)          # keep 'not on the pitch' (subs) as NaN
        x["seen_share"] = np.nanmean(seen); out.append(x)
    return pd.concat(out)


def add_noise(t, noise, seed=0, smooth_s=1.0):
    """position error per player, std `noise` m: smooth_s=0 = independent every frame (jumps of ~noise*1.4 m per
    1/25 s, i.e. impossible speeds); smooth_s>0 = error that drifts over ~smooth_s seconds, like calibration / box
    errors on a tracked player (our tracks are smoothed, M1b)"""
    t = t.copy(); rng = np.random.default_rng(seed); k = max(1, int(smooth_s * FPS))
    for c in [c for c in t.columns if re.match(r"(home|away)_\d+_[xy]$", c)]:
        e = rng.normal(0, 1, len(t) + 4 * k)
        if k > 1:
            w = np.exp(-0.5 * (np.arange(-2 * k, 2 * k + 1) / k) ** 2); e = np.convolve(e, w / np.sqrt((w ** 2).sum()), "same")
        t[c] = t[c] + noise * e[2 * k: 2 * k + len(t)]
    return t


def build_tracking(game, minutes=None, noise=0.0, seed=0, view=None, smooth_s=1.0):
    d = f"{METRICA}/Sample_Game_{game}"; g = f"Sample_Game_{game}"
    h = _read_team(f"{d}/{g}_RawTrackingData_Home_Team.csv", "home"); a = _read_team(f"{d}/{g}_RawTrackingData_Away_Team.csv", "away")
    t = pd.concat([h, a.drop(columns=["period_id", "frame_raw", "ball_x", "ball_y"])], axis=1)
    if minutes:                                                            # first N minutes of each period (quick runs)
        t = t[t.groupby("period_id").cumcount() < int(minutes * 60 * FPS)]
    t = t.reset_index(drop=True)
    if noise > 0:                                                          # position error on players, std `noise` metres:
        t = add_noise(t, noise, seed, smooth_s)                            # smooth_s=0 -> new error every frame, else drifting
    if view:                                                               # follow-cam: only players within view/2 metres (along
        t = follow_cam(t, view)                                            # the pitch) of the camera centre are seen, rest filled
    t["frame_id"] = t["frame_raw"] - 1
    t["timestamp"] = (t["frame_id"] - t.groupby("period_id")["frame_id"].transform("min")) / FPS
    alive = alive_mask(t["ball_x"], t["ball_y"])
    t["ball_state"] = np.where(alive, "alive", "dead"); t["ball_owning_team_id"] = np.nan
    t["episode_id"] = label_episodes(alive, t["period_id"].values)
    return t


def load_model(trial=140):
    sys.path.insert(0, PCRF); os.chdir(PCRF)
    import torch
    from models.utils import build_model, load_trial_args, resolve_model_path
    args = load_trial_args(f"saved/{trial:03d}")
    m = build_model(args, device="cpu")
    m.load_state_dict(torch.load(resolve_model_path(f"saved/{trial:03d}", "state_dict_best_acc.pt"), map_location="cpu", weights_only=False))
    return m.eval(), args


def run_pathcrf(model, args, tracking):
    import torch
    from datatools import utils
    from datatools.postprocess import detect_events
    from inference import inference
    tr, _ = utils.label_phases(tracking)
    tr = utils.calculate_running_features(tr)
    with torch.no_grad():
        _, _, micro, stats = inference(model, tr, use_crf=args.get("crf_weight", 0) > 0, decode="indep",
                                       window_seconds=args.get("window_seconds"), correct_episode_lasts=True, evaluate=False)
    ev = detect_events(tr, micro)
    return tr, micro, ev, stats


def stats_from_edges(tr, micro, ev, game_frames_by_period):
    """possession % per period (frames where a player holds the ball, by team; ball in flight between teammates counts
    for that team), completed passes per team (kick to a teammate), balls lost (kick to an opponent / out excluded)"""
    out = {}
    tr = tr.set_index("frame_id") if "frame_id" in tr.columns else tr
    src = micro["edge_src"].reindex(tr.index); dst = micro["edge_dst"].reindex(tr.index)
    team_of = lambda s: s.astype(str).str.split("_").str[0]
    ts, td = team_of(src), team_of(dst)
    poss = np.where((ts == td) & ts.isin(["home", "away"]), ts, None)
    for per in sorted(tr["period_id"].unique()):
        m = (tr["period_id"] == per).values & (tr["episode_id"] > 0).values
        p = pd.Series(poss[m]).dropna(); n = max(1, len(p))
        E = ev[ev["period_id"] == per] if len(ev) else ev
        kicks = E[E["event_type"] == "kick"] if len(E) else E
        rec_team = kicks["receiver_id"].astype(str).str.split("_").str[0]; kt = kicks["player_id"].astype(str).str.split("_").str[0]
        comp = kicks[(rec_team == kt).values]; lost = kicks[(rec_team != kt).values & rec_team.isin(["home", "away"]).values]
        out[int(per)] = {"possession_pct": {"Home": round(100 * (p == "home").sum() / n), "Away": round(100 * (p == "away").sum() / n)},
                         "passes": {"Home": int((comp["player_id"].str.startswith("home")).sum()), "Away": int((comp["player_id"].str.startswith("away")).sum())},
                         "balls_lost": {"Home": int((lost["player_id"].str.startswith("home")).sum()), "Away": int((lost["player_id"].str.startswith("away")).sum())},
                         "pass_frames": {"Home": comp[comp["player_id"].str.startswith("home")]["frame_id"].astype(int).tolist(),
                                         "Away": comp[comp["player_id"].str.startswith("away")]["frame_id"].astype(int).tolist()}}
    return out, pd.Series(poss, index=tr.index)


def truth_team_per_frame(events, frames_index, period_of):
    """Metrica labels -> which team has the ball per frame: the team of the last on-ball event (PASS, CARRY, RECOVERY,
    SET PIECE, SHOT, BALL LOST), until the next one (frame numbers 0-based like PathCRF)"""
    E = sorted([e for e in events if e["type"] in ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST", "CHALLENGE")
                and e["team"] in ("Home", "Away")], key=lambda e: e["f0"])
    lab = pd.Series(index=frames_index, dtype=object)
    for a, b in zip(E, E[1:] + [None]):
        f0 = a["f0"] - 1; f1 = (b["f0"] - 1) if b else frames_index.max() + 1
        if a["type"] == "BALL LOST": f0 = (a["f1"] or a["f0"]) - 1                                      # after the loss: other team / loose
        lab.loc[(lab.index >= f0) & (lab.index < f1)] = a["team"].lower()
    return lab


def pass_match(true_f, pred_f, tol=FPS):
    """greedy one-to-one match of pass start frames within +-1 s -> (found, real)"""
    pairs = sorted((abs(p - t), i, j) for i, t in enumerate(true_f) for j, p in enumerate(pred_f) if abs(p - t) <= tol)
    ut, up = set(), set()
    for _, i, j in pairs:
        if i in ut or j in up: continue
        ut.add(i); up.add(j)
    return len(ut), len(up)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--games", default="1,2"); ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--noise", type=float, default=0.0); ap.add_argument("--view", type=float, default=None); ap.add_argument("--smooth", type=float, default=1.0); ap.add_argument("--out", default=None); ap.add_argument("--skip-ours", action="store_true")
    a = ap.parse_args(); ensure_repos()
    if a.out: a.out = os.path.abspath(a.out)                                 # load_model() changes the working folder
    import metricalab as ML
    model, args = load_model(); cwd = os.getcwd(); rep = {"pathcrf_commit": PCRF_SHA, "model": "trial 140 (dynamic masked CRF), state_dict_best_acc.pt",
                                                            "minutes": a.minutes, "player_noise_m": a.noise, "noise_smooth_s": a.smooth, "follow_cam_view_m": a.view, "games": {}}
    for g in [int(x) for x in a.games.split(",")]:
        t = build_tracking(g, a.minutes, a.noise, view=a.view, smooth_s=a.smooth)
        tr, micro, ev, st = run_pathcrf(model, args, t)
        frames_by_per = {int(p): x["frame_id"].tolist() for p, x in t.groupby("period_id")}
        pc, poss = stats_from_edges(tr, micro, ev, frames_by_per)
        evs = ML.load_events(f"{METRICA}/Sample_Game_{g}/Sample_Game_{g}_RawEventsData.csv")
        keep = set(t["frame_raw"]); evs = [e for e in evs if e["f0"] in keep]
        lab = truth_team_per_frame(evs, poss.index, None)
        trI = tr.set_index("frame_id") if "frame_id" in tr.columns else tr
        G = {"pathcrf_stats": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in st.items() if not str(k).startswith("ep_")}}
        if not a.skip_ours:                                                    # our ball-based stats on the same frames
            home = ML.load_tracking(f"{METRICA}/Sample_Game_{g}/Sample_Game_{g}_RawTrackingData_Home_Team.csv", 100)
            away = ML.load_tracking(f"{METRICA}/Sample_Game_{g}/Sample_Game_{g}_RawTrackingData_Away_Team.csv", 200)
            frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home if k in keep}
            byper = {}
            for k, (p, _, _) in frames.items(): byper.setdefault(p, []).append(k)
            T = ML.truth(evs, byper)
        for per in sorted(pc):
            m = (trI["period_id"] == per).values & (trI["episode_id"] > 0).values
            lp, pp = lab[m], poss[m]; both = lp.notna() & pp.notna()
            row = {"pathcrf": {k: v for k, v in pc[per].items() if k != "pass_frames"},
                   "frames_in_play": int(m.sum()), "pathcrf_holder_share": round(float(pp.notna().mean()), 3),
                   "team_per_frame_agree": round(float((lp[both] == pp[both]).mean()), 3), "frames_compared": int(both.sum())}
            tp = {tm: [e["f0"] - 1 for e in evs if e["period"] == per and e["type"] == "PASS" and e["team"] == tm] for tm in ("Home", "Away")}
            fr = [pass_match(tp[tm], pc[per]["pass_frames"][tm]) for tm in ("Home", "Away")]
            nt = sum(len(v) for v in tp.values()); npd = sum(len(v) for v in pc[per]["pass_frames"].values())
            if a.view: row["players_seen_share"] = round(float(t.loc[t["period_id"] == per, "seen_share"].iloc[0]), 3)
            row["passes_found"] = f"{sum(f[0] for f in fr)}/{nt}"; row["passes_real"] = f"{sum(f[1] for f in fr)}/{npd}"
            if not a.skip_ours:
                row["truth"] = {k: v for k, v in T[per].items() if k in ("passes", "balls_lost", "possession_pct", "dead_pct")}
                for nz in (False, True):
                    O = ML.ours(frames, per, nz)
                    row["ours_" + ("noisy" if nz else "clean")] = {k: O[k] for k in ("passes", "balls_lost", "possession_pct")}
            rep["games"].setdefault(str(g), {})[str(per)] = row
            print(g, per, json.dumps(row), flush=True)
    os.chdir(cwd)
    out = a.out or os.path.join(ROOT, f"results/pathcrf/pc1_{datetime.date.today()}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rep, open(out, "w"), indent=1); print("->", out)


if __name__ == "__main__":
    main()
