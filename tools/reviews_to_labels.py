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

# event id prefix -> the export's event type (ids: lost_<frame>, won_<frame>, hto_<t>, seq_<frame>, restart_<t>,
# pass_<t>_<player>, shot_<t>); a review whose event is no longer in the export still gets the right type
PREFIX_TYPE = {"lost": "turnover_lost", "won": "turnover_won", "hto": "high_turnover", "seq": "sequence_end",
               "restart": "set_piece", "pass": "pass", "shot": "shot"}
TIME_IN_ID = ("hto", "restart", "pass", "shot")  # these ids carry the time in seconds after the prefix


def id_type(eid):
    return PREFIX_TYPE.get(eid.split("_", 1)[0], eid.split("_", 1)[0])


def id_time(eid):
    parts = eid.split("_")
    if parts[0] in TIME_IN_ID and len(parts) > 1:
        try: return float(parts[1])
        except ValueError: return None
    return None


def build_match(m, reviews, corrections, lineups, events):
    """Pure part: rows from the app + the export's events {id: event} -> (events_key, coach_key, kick_windows); None when empty."""
    items = []
    for r in reviews:
        eid = r["event_id"]; e = events.get(eid, {})
        t = r.get("t_corrected") if r.get("t_corrected") is not None else e.get("t", id_time(eid))
        items.append({"event_id": eid, "t": None if t is None else float(t), "type": e.get("type") or id_type(eid),
                      "team": r.get("team_corrected") or e.get("team"), "verdict": r["verdict"], "note": r.get("note")})
    ek = kw = None
    if items:
        ek = {"match": m, "note": "coach reviews from the app (event_reviews), 18 Oct tool; a label, never an edit of the pipeline data", "items": items}
        kw = {"match": m, "windows": [{"t0": round(max(0.0, i["t"] - 1.0), 2), "t1": round(i["t"] + 1.0, 2), "type": i["type"], "team": i["team"]}
                                      for i in items if i["verdict"] in ("confirmed", "retimed") and isinstance(i["t"], float)]}
    coach = {}
    # several corrections of one field: the latest one wins (the app does not return rows in order)
    for c in sorted(corrections, key=lambda c: str(c.get("updated_at") or c.get("created_at") or "")): coach[c["field"]] = c["value"]
    if lineups: coach["lineup"] = [{"player_id": l["player_id"], "shirt": l.get("shirt"), "started": l["started"], "minute_on": l["minute_on"], "minute_off": l.get("minute_off")} for l in lineups]
    ck = {"match": m, **coach} if coach else None
    return ek, ck, kw


def main(match_ids, root="."):
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key): print("reviews_to_labels: SUPABASE_URL / SUPABASE_SERVICE_KEY not set"); sys.exit(2)
    try:
        from supabase import create_client
    except ImportError:
        import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "supabase"], capture_output=True)
        from supabase import create_client
    sb = create_client(url, key)
    def rows(table):
        return sb.table(table).select("*").execute().data or []
    reviews = rows("event_reviews"); corrections = rows("match_corrections"); lineups = rows("match_lineups")
    by_match = collections.defaultdict(lambda: {"reviews": [], "corrections": [], "lineups": []})
    for r in reviews: by_match[r["match_id"]]["reviews"].append(r)
    for r in corrections: by_match[r["match_id"]]["corrections"].append(r)
    for r in lineups: by_match[r["match_id"]]["lineups"].append(r)
    out = {}; lab = f"{root}/results/labels"; os.makedirs(lab, exist_ok=True)
    for m, d in by_match.items():
        if match_ids and m not in match_ids: continue
        # the review carries the pipeline's event id; time, type and team come from the match's own export
        # (match_data.json in the app's storage), the coach's corrections override them
        ev = {}
        if d["reviews"]:
            try:
                md = json.loads(sb.storage.from_("matches").download(f"{m}/match_data.json").decode())
                ev = {e["id"]: e for e in md.get("events", [])}
            except Exception as e: print(f"  {m}: match_data.json not readable ({e!r}); times from t_corrected / the event id")
        ek, ck, kw = build_match(m, d["reviews"], d["corrections"], d["lineups"], ev)
        ref = f"{root}/reference/{m}"
        if ek or ck: os.makedirs(ref, exist_ok=True)
        if ek: json.dump(ek, open(f"{ref}/events_key.json", "w"), indent=1); json.dump(kw, open(f"{lab}/{m}_kick_windows.json", "w"), indent=1)
        if ck: json.dump(ck, open(f"{ref}/coach_key.json", "w"), indent=1)
        out[m] = {"reviews": len(d["reviews"]), "corrections": len(d["corrections"]), "lineup_rows": len(d["lineups"])}
        print(m, out[m])
    json.dump(out, open(f"{lab}/reviews_to_labels.json", "w"), indent=1); print(json.dumps(out))

if __name__ == "__main__": main(sys.argv[1:])
