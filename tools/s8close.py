"""S8 close-out (6 Oct): with today's code, how often does the possession state flip on the exact app inputs of both clips -
raw (shown as possession %) vs the spell state the counted stats read (P.spell_state, run.py). Also sequences, turnovers,
passes (counted ones filtered by the spell state like run.py since S4b; graded SFK-BP ones: real/fake kept). Free, local. PYTHONPATH=. python tools/s8close.py"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P
from tools.s8lab import load, run

CLIPS = ("SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520")

def counted_flips_per_min(clip):
    c = load(clip); mins = len(c["per"]) / c["fps"] / 60
    st = P.spell_state(c["state"], c["fps"], P.SPELL_TAKE_S, P.SPELL_JOIN_S)
    return c, mins, st

if __name__ == "__main__":
    A = json.load(open("results/review/passcheck_answers.json")); fake = set(A["fake"]); unsure = set(A["unsure"])
    graded = [i for i in range(A["graded_first_n"]) if i not in unsure]
    exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]
    def grade(ps):
        keep = [any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in ps) for i in graded]
        return sum(k for i, k in zip(graded, keep) if i not in fake), sum(k for i, k in zip(graded, keep) if i in fake)
    out = {"date": time.strftime("%Y-%m-%d"), "take_s": P.SPELL_TAKE_S, "join_s": P.SPELL_JOIN_S, "clips": {}}
    for clip in CLIPS:
        c, mins, st = counted_flips_per_min(clip); row = {"minutes": round(mins, 2)}
        for name, s in (("raw", c["state"]), ("counted", st)):
            r = run(c, s, pass_state=(name == "counted")); per5 = 5 / mins
            row[name] = dict(flips_per_min=round(r["flips"] / mins, 2), sequences_per_5min=round(r["seqs"] * per5, 1),
                             turnovers_per_5min=round(r["turnovers"] * per5, 1), passes_per_5min=round(len(r["passes"]) * per5, 1))
            if clip.startswith("SFK"): row[name]["graded_passes_real_fake_kept"] = list(grade(r["passes"]))
        out["clips"][clip] = row; print(clip, json.dumps(row))
    json.dump(out, open("results/possession/s8/s8close.json", "w"), indent=1)
