"""N0-22d: possession-over-time windows (metrics.tilt_windows) old (empty window = 0.0) vs new (None), on the repo copies.
Rebuilds the state + ball metres from the exported frame files (every 3rd frame), with N0-21d applied (frames outside the
match periods are nobody's, as the next join will export them), then draws the app's chart as specified to Lovable
(results/app/lovable_fixes_2026-10-04.md section 10: 15-s windows, 5-min rolling mean, null windows skipped).
The spec also skips windows outside the periods; the chart is drawn both ways (filter on / off).
Usage: PYTHONPATH=. python tools/n0_22d_tilt.py"""
import os, sys, json, glob, importlib
import numpy as np

MATCHES = ["SFKBP1109", "p15u-vs-vallentuna-2026-10-03-6cce"]   # AIK left out: its repo copy has 2 of 12 frame files (N0-19c)
OUT = "results/qa/n0-22d"


def load(mid):
    d = f"results/volume/runs/matches/{mid}/"
    md = json.load(open(d + "match_data.json")); fr = []
    for p in sorted(glob.glob(d + "frames_*.json")): fr += json.load(open(p))["frames"]
    return md, fr


def windows(fr, periods, L, ar, mode):
    os.environ["IPANEMA_TILT_EMPTY"] = mode
    import ipanema.metrics as M; importlib.reload(M)
    ts = np.array([f["t"] for f in fr]); step = float(np.median(np.diff(ts[:500]))); fps = 1 / step
    # a frame grid from 0 s so windows line up with the export's (missing frame files = no frames = nobody's)
    n = int(round(ts.max() * fps)) + 1; state = [2] * n; ballm = {}
    inper = lambda t: any(p["t_start"] <= t < p["t_end"] for p in periods)
    for f in fr:
        k = int(round(f["t"] * fps))
        if k >= n or not inper(f["t"]): continue                     # N0-21d: outside match time = nobody's
        if f["possession"] in ("A", "B"): state[k] = 0 if f["possession"] == "A" else 1
        if f.get("ball") and f["ball"].get("m"): ballm[k] = np.array(f["ball"]["m"])
    return M.tilt_windows(state, ballm, fps, L, ar)


def rolling(tw, periods, filt):
    t = np.array([w["t"] for w in tw]); out = []
    inper = lambda w: any(p["t_start"] <= (w["t"] + w["t_end"]) / 2 < p["t_end"] for p in periods)
    keep = [w for w in tw if w["possession_A"] is not None and (not filt or inper(w))]
    kt = np.array([w["t"] for w in keep]); kv = np.array([w["possession_A"] for w in keep])
    for w in tw:
        if filt and not inper(w): out.append(np.nan); continue
        sel = np.abs(kt - w["t"]) <= 150
        out.append(kv[sel].mean() if sel.any() else np.nan)
    return t, np.array(out)


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = {}; fig, axes = plt.subplots(len(MATCHES), 1, figsize=(12, 3.2 * len(MATCHES)))
    for ax, mid in zip(axes, MATCHES):
        md, fr = load(mid); periods = md["periods"]; L = md["pitch"]["length"]; ar = md["attack_right"]
        old = windows(fr, periods, L, ar, "zero"); new = windows(fr, periods, L, ar, "none")
        inper = lambda w: any(p["t_start"] <= (w["t"] + w["t_end"]) / 2 < p["t_end"] for p in periods)
        ch = [(o, w) for o, w in zip(old, new) if o["possession_A"] != w["possession_A"]]
        assert all(w["possession_A"] is None and o["possession_A"] == 0.0 for o, w in ch)
        assert all(o["possession_A"] == w["possession_A"] for o, w in zip(old, new) if w["possession_A"] is not None)
        r = {"windows": len(new), "changed_to_null": len(ch), "changed_inside_periods": sum(1 for o, w in ch if inper(w)),
             "changed_t": [w["t"] for o, w in ch][:80], "frame_files_missing": len(md.get("frame_chunks", [])) - len(glob.glob(f"results/volume/runs/matches/{mid}/frames_*.json"))}
        for filt in (False, True):
            t, a = rolling(old, periods, filt); _, b = rolling(new, periods, filt)
            both = ~np.isnan(a) & ~np.isnan(b)
            r[f"rolling_{'periods_only' if filt else 'no_filter'}"] = {"max_diff_pts": round(float(np.nanmax(np.abs(a - b)[both]) * 100), 1) if both.any() else 0.0,
                                                                         "windows_moved_5pts": int((np.abs(a - b)[both] > 0.05).sum())}
        t, a = rolling(old, periods, False); _, b = rolling(new, periods, False)
        ax.plot(t / 60, a * 100, color="#c0392b", lw=1.5, label="old: empty window = 0.0 (opponent 100%)")
        ax.plot(t / 60, b * 100, color="#2471a3", lw=1.5, label="new: empty window skipped")
        for p in periods: ax.axvspan(p["t_start"] / 60, p["t_end"] / 60, color="#2ecc71", alpha=0.08)
        for o, w in ch: ax.plot(w["t"] / 60, 2, "|", color="#555", ms=6)
        ax.axhline(50, color="#999", lw=0.6); ax.set_ylim(0, 100); ax.set_ylabel("dark team (A) %")
        ax.set_title(f"{mid}: 5-min rolling possession as the app draws it, no period filter (green = match periods, ticks = windows now null)", fontsize=9)
        res[mid] = r
    axes[0].legend(fontsize=8, loc="upper right"); axes[-1].set_xlabel("video minute")
    fig.tight_layout(); fig.savefig(f"{OUT}/possession_over_time.png", dpi=90)
    json.dump(res, open(f"{OUT}/tilt_windows_compare.json", "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
