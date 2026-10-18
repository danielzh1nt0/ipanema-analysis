"""18 Oct 2026 (data strategy): the coach's own corrections become the pipeline's answer keys and training labels.
Reads the app's tables (event_reviews: confirmed / deleted / retimed events; match_corrections: possession, score, direction;
match_lineups: who played) and writes, per match:
  reference/<match>/events_key.json   {items: [{t, type, team, verdict}]}  - graded by tools/export_qa.py (passes, turnovers)
  reference/<match>/coach_key.json    {possession_pct, score, attack_right, lineup: [...]} - graded the same way
  results/labels/<match>_kick_windows.json  confirmed pass / turnover moments as 2-s windows: training clips for our own
                                       kick spotter (plan item L1), never T-DEED.
Needs SUPABASE_URL + SUPABASE_SERVICE_KEY (the GitHub Actions secrets); prints what is missing otherwise and exits 2.
Nothing in the pipeline data is modified; a review is a label, not an edit.
    python tools/reviews_to_labels.py [match_id ...]"""
import os, sys, json, collections

def main(match_ids):
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key): print("reviews_to_labels: SUPABASE_URL / SUPABASE_SERVICE_KEY not set"); sys.exit(2)
    try:
        from supabase import create_client
    except ImportError:
        import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "supabase"], capture_output=True)
        from supabase import create_client
    sb = create_client(url, key)
    def rows(table, col=None, val=None):
        q = sb.table(table).select("*")
        if col: q = q.eq(col, val)
        return q.execute().data or []
    reviews = rows("event_reviews"); corrections = rows("match_corrections"); lineups = rows("match_lineups")
    by_match = collections.defaultdict(lambda: {"reviews": [], "corrections": [], "lineups": []})
    for r in reviews: by_match[r["match_id"]]["reviews"].append(r)
    for r in corrections: by_match[r["match_id"]]["corrections"].append(r)
    for r in lineups: by_match[r["match_id"]]["lineups"].append(r)
    out = {}
    for m, d in by_match.items():
        if match_ids and m not in match_ids: continue
        ref = f"reference/{m}"; os.makedirs(ref, exist_ok=True)
        # the review carries the pipeline's event id (lost_<frame>, pass_<t>_<from>, ...); time, type and team come from the
        # match's own export (match_data.json in the app's storage), the coach's corrections override them
        ev = {}
        try:
            md = json.loads(sb.storage.from_("matches").download(f"{m}/match_data.json").decode())
            ev = {e["id"]: e for e in md.get("events", [])}
        except Exception as e: print(f"  {m}: match_data.json not readable ({e!r}); times only from t_corrected")
        items = []
        for r in d["reviews"]:
            eid = r["event_id"]; e = ev.get(eid, {})
            t = r.get("t_corrected") if r.get("t_corrected") is not None else e.get("t")
            items.append({"event_id": eid, "t": None if t is None else float(t), "type": e.get("type") or eid.split("_", 1)[0], "team": r.get("team_corrected") or e.get("team"), "verdict": r["verdict"], "note": r.get("note")})
        if items:
            json.dump({"match": m, "note": "coach reviews from the app (event_reviews), 18 Oct tool; a label, never an edit of the pipeline data", "items": items}, open(f"{ref}/events_key.json", "w"), indent=1)
            kicks = [{"t0": round(i["t"] - 1.0, 2), "t1": round(i["t"] + 1.0, 2), "type": i["type"], "team": i["team"]} for i in items if i["verdict"] in ("confirmed", "retimed") and isinstance(i["t"], float)]
            os.makedirs("results/labels", exist_ok=True); json.dump({"match": m, "windows": kicks}, open(f"results/labels/{m}_kick_windows.json", "w"), indent=1)
        coach = {}
        for c in d["corrections"]: coach[c["field"]] = c["value"]
        if d["lineups"]: coach["lineup"] = [{"player_id": l["player_id"], "shirt": l.get("shirt"), "started": l["started"], "minute_on": l["minute_on"], "minute_off": l.get("minute_off")} for l in d["lineups"]]
        if coach: json.dump({"match": m, **coach}, open(f"{ref}/coach_key.json", "w"), indent=1)
        out[m] = {"reviews": len(items), "corrections": len(d["corrections"]), "lineup_rows": len(d["lineups"])}
        print(m, out[m])
    json.dump(out, open("results/labels/reviews_to_labels.json", "w"), indent=1); print(json.dumps(out))

if __name__ == "__main__": main(sys.argv[1:])
