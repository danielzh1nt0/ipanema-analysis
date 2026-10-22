"""N0-22c picture: one real track from an app clip - raw positions, old smoothing (across gaps) and new (per seen run).
Usage: python tools/n0_22c_plot.py <picker_inputs.pkl> <out.png> [track id]"""
import sys, os, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ipanema import analytics as AN

d = pickle.load(open(sys.argv[1], "rb")); per, fps = d["per"], d["fps"]; tr = collections.defaultdict(dict)
for k in range(len(per)):
    for r in per[k]: tr[r[0]][k] = np.asarray(r[2], float)
def change(t):
    ks = np.array(sorted(tr[t])); P = np.array([tr[t][k] for k in ks])
    return abs(AN.track_motion(ks, P, fps, runs=False, gap_line_s=0)[2].sum() - AN.track_motion(ks, P, fps)[2].sum())
tid = int(sys.argv[3]) if len(sys.argv) > 3 else max((t for t in tr if 4 * fps <= len(tr[t]) <= 20 * fps), key=change)
ks = np.array(sorted(tr[tid])); P = np.array([tr[tid][k] for k in ks]); t = ks / fps
old = AN.smooth_runs(ks, P, int(0.8 * fps), runs=False); new = AN.smooth_runs(ks, P, int(0.8 * fps), runs=True)
so, sto, hio = AN.track_motion(ks, P, fps, runs=False, gap_line_s=0); sn, stn, hin = AN.track_motion(ks, P, fps)
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for j, lab in enumerate(("x (m)", "y (m)")):
    ax[j].plot(t, P[:, j], ".", ms=3, color="0.6", label="tracked feet (raw)")
    ax[j].plot(t, old[:, j], "-", color="tab:red", lw=1.5, label="old smoothing (across gaps)")
    ax[j].plot(t, new[:, j], "-", color="tab:blue", lw=1.5, label="new (per seen run)")
    ax[j].set_ylabel(lab)
ax[2].plot(t, so, "-", color="tab:red", lw=1, label=f"old speed: {sto.sum():.0f} m, high-int. {hio.sum():.0f} m, top {np.nanmax(so):.1f} m/s")
ax[2].plot(t, sn, "-", color="tab:blue", lw=1, label=f"new speed: {stn.sum():.0f} m, high-int. {hin.sum():.0f} m, top {np.nanmax(sn):.1f} m/s")
ax[2].axhline(5.5, color="0.5", ls=":", lw=1); ax[2].set_ylabel("speed (m/s)"); ax[2].set_xlabel("clip time (s)")
for a in ax:
    for i in np.flatnonzero(np.diff(ks) > 3): a.axvspan(ks[i] / fps, ks[i + 1] / fps, color="orange", alpha=0.2, lw=0)
ax[0].legend(fontsize=8, loc="best"); ax[2].legend(fontsize=8, loc="best")
ax[0].set_title(f"{os.path.basename(os.path.dirname(sys.argv[1]))} track {tid} (orange = not seen)")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=90); print("track", tid, "->", sys.argv[2])
