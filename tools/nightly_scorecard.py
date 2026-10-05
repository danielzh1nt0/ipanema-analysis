"""nightly: combine the parallel results into results/nightly/<date>.md + history.tsv

N1 (5 Oct): the card now compares itself with the day before and adds a
'Worse than <date>' row, so a drop is seen without reading two cards side by side.
"""
import json, glob, os, re, time


def player_row(f, v):
    name = f.split('/')[2].replace('tracktest_', '').replace('tracktest', 'SFKBP1109')[:30]
    return (f"| Players, {name} | seen/frame dark {v['observed_per_frame']['dark']} light {v['observed_per_frame']['light']}, "
            f"tracked {v['median_track_s']['dark']}/{v['median_track_s']['light']} s |")


def build_lines(d, q, summaries):
    """summaries: list of (path, summary dict)."""
    lines = [f"# Nightly check {d}", "", "| Check | Result |", "|---|---|"]
    t = q.get("tests", {}); lines.append(f"| Tests | {t.get('total', '?') - len(t.get('failed', [])) if isinstance(t.get('total'), int) else '?'}/{t.get('total', '?')} pass" + (f" — FAILED: {', '.join(t['failed'])}" if t.get("failed") else "") + " |")
    b = q.get("ball_34_moments", {}); lines.append(f"| Ball, 34 checked moments | {b.get('right', '?')}/{b.get('of', '?')} (best possible {b.get('best_possible', '?')}) |")
    s = q.get("stats_pro_data", {}); lines.append(f"| Stats on pro data (worst half) | passes {s.get('worst_pass_err_pct', '?')}% off, possession {s.get('worst_poss_err_pts', '?')} pts off |")
    if "sequences_found_pct" in s: lines.append(f"| Sequences on pro data (all halves) | {s['sequences_found_pct']}% of real ones found, {s['sequences_real_pct']}% of ours real (since 3 Oct counted on the spell state like the app, S6: expect ~8 pts fewer found than before (47 -> 40%), not a regression) |")
    if s.get("possession_model") == "simple": lines.append("| Note | From 29 Sep the pro-data stats use the pipeline's possession model (possession_simple), not the old one; numbers before that date are not comparable (E5) |")
    for f, r in summaries:
        v = r.get("new, RF-DETR")
        if v: lines.append(player_row(f, v))
    return lines


def _num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def read_card(lines):
    """Pull the comparable numbers out of a card's table rows."""
    out = {}
    for ln in lines:
        m = re.match(r"\| Tests \| (\S+)/(\S+) pass", ln)
        if m: out["tests_pass"] = _num(m.group(1)); out["tests_total"] = _num(m.group(2))
        m = re.match(r"\| Ball, 34 checked moments \| (\S+)/", ln)
        if m: out["ball"] = _num(m.group(1))
        m = re.match(r"\| Stats on pro data \(worst half\) \| passes (\S+)% off, possession (\S+) pts off", ln)
        if m: out["pass_err"] = _num(m.group(1)); out["poss_err"] = _num(m.group(2))
        m = re.match(r"\| Sequences on pro data \(all halves\) \| (\S+)% of real ones found, (\S+)% of ours real", ln)
        if m: out["seq_found"] = _num(m.group(1)); out["seq_real"] = _num(m.group(2))
        m = re.match(r"\| Players, (.+?) \| seen/frame dark (\S+) light (\S+), tracked (\S+)/(\S+) s", ln)
        if m:
            g = m.group(1)
            for k, i in (("dark", 2), ("light", 3), ("trk_dark", 4), ("trk_light", 5)):
                out[f"pl|{g}|{k}"] = _num(m.group(i))
    return out


# key -> (plain name, higher is better, smallest change that counts)
RULES = {"ball": ("ball moments right", True, 1), "pass_err": ("pass count error %", False, 1),
         "poss_err": ("possession error pts", False, 1), "seq_found": ("sequences found %", True, 3),
         "seq_real": ("sequences real %", True, 3)}


def worse_than(prev, new):
    """List plain-language lines for everything that got worse from prev to new."""
    out = []
    if new.get("tests_pass") is not None and new.get("tests_total") is not None and new["tests_pass"] < new["tests_total"]:
        out.append(f"tests failing ({int(new['tests_total'] - new['tests_pass'])})")
    for k, (name, up, eps) in RULES.items():
        a, b = prev.get(k), new.get(k)
        if a is None or b is None: continue
        if (b < a - eps + 1e-9) if up else (b > a + eps - 1e-9): out.append(f"{name} {a:g} -> {b:g}")
    for k, b in new.items():
        if not k.startswith("pl|"): continue
        a = prev.get(k)
        if a is None or b is None: continue
        _, g, what = k.split("|")
        if what in ("dark", "light") and b <= a - 1:
            out.append(f"{g}: {what} players seen per frame {a:g} -> {b:g}")
        if what.startswith("trk_") and a > 0 and b < 0.8 * a:
            out.append(f"{g}: {what[4:]} median time tracked {a:g} -> {b:g} s")
    return out


def previous_card(d, folder="results/nightly"):
    days = sorted(p for p in glob.glob(f"{folder}/????-??-??.md") if os.path.basename(p)[:10] < d)
    return days[-1] if days else None


def compare_row(d, lines, folder="results/nightly"):
    p = previous_card(d, folder)
    if not p: return None
    w = worse_than(read_card(open(p).read().splitlines()), read_card(lines))
    pd = os.path.basename(p)[:10]
    return f"| Worse than {pd} | " + ("; ".join(w) if w else "nothing") + " |"


def main():
    d = time.strftime("%Y-%m-%d"); q = json.load(open("results/nightly/quick.json")) if os.path.exists("results/nightly/quick.json") else {}
    summaries = [(f, json.load(open(f))) for f in sorted(glob.glob("results/qa/tracktest*/summary.json"))]
    lines = build_lines(d, q, summaries)
    row = compare_row(d, lines)
    if row: lines.insert(4, row)
    t = q.get("tests", {}); b = q.get("ball_34_moments", {}); s = q.get("stats_pro_data", {})
    os.makedirs("results/nightly", exist_ok=True); open(f"results/nightly/{d}.md", "w").write("\n".join(lines) + "\n")
    h = "results/nightly/history.tsv"; new = not os.path.exists(h)
    with open(h, "a") as fh:
        if new: fh.write("date\ttests_failed\tball_right\tball_of\tpass_err\tposs_err\n")
        fh.write(f"{d}\t{len(t.get('failed', []))}\t{b.get('right', '')}\t{b.get('of', '')}\t{s.get('worst_pass_err_pct', '')}\t{s.get('worst_poss_err_pts', '')}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
