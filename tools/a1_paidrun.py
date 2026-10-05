"""A1 (5 Oct): the "one paid run" list - what goes to the app, on which match, at what cost, with the exact push.

Nothing here talks to Modal. It only builds and checks the plan:
  PYTHONPATH=. python tools/a1_paidrun.py            # check + write results/app/a1_paid_run.md
Checks: every Modal job tag in a commit message exists in .github/workflows/run-on-modal.yml, the match has a periods
file, no message carries a tag it should not (e.g. '[run'), and a GPU re-track never runs while an id-keyed team override
file is still in place (after a re-track the player ids change, so the old override would put players in wrong teams).

Prices: Modal list prices as known to Claude (not read from Daniel's bill) - CPU $0.0000131 per core-second, memory
$0.00000222 per GiB-second, L4 GPU $0.000222 per second. Minutes are measured from the last runs (git log of the live logs).
"""
import os, re, sys, json

CPU_S, MEM_S, L4_S, USD_EUR = 0.0000131, 0.00000222, 0.000222, 0.92
VALL, BP, AIK = "p15u-vs-vallentuna-2026-10-03-6cce", "SFKBP1109", "p15u-vs-aik-2026-09-21-bd09"
JOB_TAGS = ("full", "full-rf", "qa-frames", "fetch", "half", "run", "export", "events", "players-rf", "ball-rf")

# cpu/mem = the Modal function's resources (modal_app.py); minutes = measured wall time of the last comparable run.
PLAN = [
    dict(id="1", prio="do first", match=VALL, title="Vallentuna teams: apply the K3c team decisions (join only, CPU)",
         why="App now shows nearly everyone as SFK (K3b, possession 79/21). The override puts ~3 of 4 players in the right team (app now ~half), owner check 13/17.",
         msg=f"A1-1 Vallentuna join with the K3c team override, CPU only (Daniel's go) [full:{VALL}] [allow-fallback] [budget:40]",
         minutes=25, cpu=8, mem=32, gpu_min=1, measured="5 Oct K3b join: 17-24 min"),
    dict(id="2", prio="do first", match=VALL, title="Vallentuna by-eye pictures after the join (CPU, cents)",
         why="Check the 12 QA moments by eye before saying it is fixed.",
         msg=f"A1-2 Vallentuna QA pictures + stats after the K3c join [qa-frames:{VALL}] [fetch:runs/matches/{VALL}/stats.json]",
         minutes=4, cpu=4, mem=16, gpu_min=0, measured="5 Oct qa-frames: ~2-4 min"),
    dict(id="3", prio="after the Lovable badge", match=BP, title="SFK-BP re-join: 'needs review' flags in stats.json (CPU)",
         why="Numbers should not move (same pieces); adds the V2 review flags and rewrites match_data (checks the goal 'B' vs 'A' found by V2).",
         msg=f"A1-3 SFK-BP re-join for the V2 review flags, CPU only (Daniel's go) [full:{BP}] [allow-fallback] [budget:30]",
         minutes=15, cpu=8, mem=32, gpu_min=1, measured="4 Oct night join: ~10-16 min"),
    dict(id="4", prio="after the Lovable badge", match=AIK, title="SFK-AIK re-join: 'needs review' flags in stats.json (CPU)",
         why="Same as 3 for AIK (flags 10% of frames with 12+ of one team).",
         msg=f"A1-4 AIK re-join for the V2 review flags, CPU only (Daniel's go) [full:{AIK}] [allow-fallback] [budget:30]",
         minutes=12, cpu=8, mem=32, gpu_min=1, measured="4 Oct night join: ~5-12 min"),
    dict(id="5", prio="after the Lovable badge", match=BP, title="Fetch both new match_data/stats files to check them (CPU, cents)",
         why="Confirms the SFK-BP goal team in match_data and reads the flags offline.",
         msg=f"A1-5 fetch the re-joined files [fetch:runs/matches/{BP}/match_data.json] [fetch:runs/matches/{BP}/stats.json] [fetch:runs/matches/{AIK}/stats.json]",
         minutes=3, cpu=1, mem=2, gpu_min=0, measured="fetch: ~1-3 min"),
    dict(id="6", prio="optional, instead of 1", match=VALL, title="Vallentuna clean fix: GPU re-track with the K3c rule, then join",
         why="Cleaner than the override (each detection decided by the rule, no ids carried from the old export). Push 6a, wait for it to finish, then push 1 and 2.",
         msg=f"A1-6a Vallentuna re-track with the K3c kit rule; team override moved aside (ids change) (Daniel's go) [full-rf:{VALL}] [retrack]",
         minutes=37, cpu=4, mem=16, gpu_min=78, override_aside=True, measured="5 Oct K3/K3b re-track: 36.7 min, ~78 GPU-min (10 L4s in parallel)"),
]

def cost_eur(s):
    cpu_usd = s["minutes"] * 60 * (s["cpu"] * CPU_S + s["mem"] * MEM_S)
    gpu_usd = s["gpu_min"] * 60 * (L4_S + 4 * CPU_S + 16 * MEM_S)          # rf_piece: L4 + 4 cores + 16 GiB per piece
    return round((cpu_usd + gpu_usd) * USD_EUR, 2)

def workflow_tags(text):
    return set(re.findall(r"contains\(github\.event\.head_commit\.message, '\[([a-z0-9-]+)", text))

def tags_in(msg):
    return re.findall(r"\[([a-z0-9-]+)(?::[^\]]*)?\]", msg)

def check(plan, wf_text, root=".", overrides_present=None):
    """list of problems (empty = plan is safe to hand to Daniel)"""
    known = workflow_tags(wf_text); bad = []
    for s in plan:
        tags = tags_in(s["msg"]); jobs = [t for t in tags if t in JOB_TAGS]
        if not jobs: bad.append(f"{s['id']}: no Modal job tag in the message")
        for t in jobs:
            if t not in known: bad.append(f"{s['id']}: tag [{t}: is not in the workflow")
        if "[run" in s["msg"] and "run" not in jobs: bad.append(f"{s['id']}: message contains '[run' (would start run_match)")
        if not os.path.exists(f"{root}/periods/{s['match']}.json"): bad.append(f"{s['id']}: no periods file for {s['match']}")
        if "full" in jobs and "[budget:" not in s["msg"]: bad.append(f"{s['id']}: a join without a [budget:N] stop")
        ov = os.path.exists(f"{root}/overrides/{s['match']}_teams.json") if overrides_present is None else overrides_present
        if "retrack" in tags and ov and not s.get("override_aside"):
            bad.append(f"{s['id']}: re-track while overrides/{s['match']}_teams.json is in place - player ids change, move it aside in the same commit")
        if s.get("override_aside") and "retrack" not in tags: bad.append(f"{s['id']}: override moved aside without a re-track")
    return bad

def push_cmd(s):
    pre = f"git mv overrides/{s['match']}_teams.json overrides/{s['match']}_teams.before_retrack.json && " if s.get("override_aside") else ""
    note = s["msg"].split(" [")[0]
    return f"{pre}echo \"{note} $(date -u)\" >> triggers/last.txt && git add -A triggers overrides && git commit -m \"{s['msg']}\" && git push origin main"

def render(plan):
    out = ["# A1: the one paid run (prepared 5 Oct by the worker; nothing started)", "",
           "Nothing below has run. Each line is one push; push only after Daniel says go in chat.",
           "Demo rule: nothing is re-run after Monday noon, so this is for after the demo (Vallentuna is not in the demo library, so step 1 does not touch the demo).", "",
           "| step | when | what | minutes (measured) | est. cost |", "|---|---|---|---|---|"]
    for s in plan: out.append(f"| {s['id']} | {s['prio']} | {s['title']} | {s['minutes']} ({s['measured']}) | ~EUR {cost_eur(s):.2f} |")
    first = [s for s in plan if s["prio"] == "do first"]; badge = [s for s in plan if s["prio"].startswith("after")]
    out += ["", f"Recommended now: steps 1-2, about EUR {sum(map(cost_eur, first)):.2f}. After the Lovable 'needs review' badge: 3-5, about EUR {sum(map(cost_eur, badge)):.2f}.",
            f"Optional clean Vallentuna fix (6a, then 1-2): about EUR {cost_eur(plan[-1]) + sum(map(cost_eur, first)):.2f} by list prices; earlier notes said ~EUR 5 for a re-track, so take EUR 5 as the cap.",
            "Costs = measured minutes x Modal list prices (not checked against the bill), +-50%.", "", "## Why each step"]
    for s in plan: out.append(f"- **{s['id']}** {s['why']}")
    out += ["", "## Exact pushes (from a clean, pulled main; one at a time, wait for 'finished' in results/live/<match>.log)"]
    for s in plan: out += [f"- {s['id']}:", "```", push_cmd(s), "```"]
    out += ["", "## Not in this run (and why)",
            "- F1 'full SFK-BP with the unsure-frames fix': already in - the 4 Oct joins ran with that code (stale waiting item).",
            "- B4c spare-ball rule: Daniel's yes/no, off by default; would need its own re-join.",
            "- K3b-only GPU re-track: superseded by 6a (K3c rule).",
            "- D4 Bundesliga (~EUR 1-2), V1c shot clicks (1/40 clicked), S2 frame fetch, F1 AIK picker inputs: separate decisions, not app updates.",
            "", "## Checks done", "- tools/a1_paidrun.py: every tag exists in run-on-modal.yml, periods files present, joins carry a [budget:] stop, no stray '[run', re-track moves the id-keyed override aside.",
            "- tests/test_a1_paidrun.py, tests/test_team_override.py pass locally."]
    return "\n".join(out) + "\n"

def main():
    wf = open(".github/workflows/run-on-modal.yml").read(); bad = check(PLAN, wf)
    if bad: print("\n".join(bad)); sys.exit(1)
    os.makedirs("results/app", exist_ok=True); md = render(PLAN); open("results/app/a1_paid_run.md", "w").write(md)
    json.dump([{**s, "eur": cost_eur(s), "push": push_cmd(s)} for s in PLAN], open("results/app/a1_paid_run.json", "w"), indent=1)
    print(md)

if __name__ == "__main__":
    main()
