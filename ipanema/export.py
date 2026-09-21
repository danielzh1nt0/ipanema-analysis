"""Write the v1 data contract: match_data.json (frames + one events list), stats.json, kits, thumb, library.json, summary.json."""
import os, json, shutil, cv2, numpy as np
from . import SCHEMA_VERSION
from .possession import STATES

def _json_default(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

def events(turnovers_, passes_, restarts_, sequences_, fps):
    ev = []
    for t in turnovers_:
        rx = t.get("reactions", {})
        ev.append({"id": f"lost_{t['frame']}", "t": t["t"], "type": "turnover_lost", "team": t["lost_by"], "title": f"{t['lost_by']} lost the ball",
                   "subtitle": f"press {t['time_to_press'] if t['time_to_press'] is not None else '—'} s · {t['near_at_2s'] if t['near_at_2s'] is not None else '—'} within 5 m · {'regained' if t['regained_within_5s'] else 'not regained'}",
                   "payload": {k: t.get(k) for k in ("time_to_press", "near_at_2s", "regained_within_5s", "ball_m_before_press")} | {"reactions": rx}})
        ev.append({"id": f"won_{t['frame']}", "t": t["t"], "type": "turnover_won", "team": t["won_by"], "title": f"{t['won_by']} won the ball",
                   "subtitle": f"forward pass {t['time_to_forward_pass'] if t['time_to_forward_pass'] is not None else '—'} s · {t['gain_5s_m'] if t['gain_5s_m'] is not None else '—'} m in 5 s",
                   "payload": {k: t.get(k) for k in ("time_to_forward_pass", "gain_5s_m", "lost_back_5s")}})
    for p in passes_:
        if p["quality"] in ("bad_lost", "risky_completed") or p.get("better_option"):
            typ = "better_option" if p.get("better_option") else ("pass_bad" if p["quality"] == "bad_lost" else "pass_risky")
            ev.append({"id": f"pass_{p['t']}_{p['from']}", "t": p["t"], "type": typ, "team": p["team"], "title": f"pass {p['from']} → {p['to']} · {p['kind']} · {p['quality'].replace('_', ' ')}",
                       "subtitle": (f"better option → {p['best_to']} (open, beats {p['best_bypassed']}, {p['best_space_m']} m space)" if p.get("better_option") else f"{p['length_m']} m, gain {p['gain_m']} m"), "payload": p})
    for r in restarts_:
        ev.append({"id": f"restart_{r['t']}", "t": r["t"], "type": "set_piece", "team": r["team"], "title": f"{r['kind']} · {r['team'] or '—'}", "subtitle": "", "payload": r})
    for s in sequences_:
        if s["end_reason"] in ("pass intercepted", "dispossessed"):
            ev.append({"id": f"seq_{s['start']}", "t": s["t_end"], "type": "sequence_end", "team": s["team"], "title": f"{s['team']} sequence ended · {s['end_reason']}", "subtitle": f"{s['duration_s']} s · gain {s['gain_m']} m", "payload": s})
    seen = set(); ev = [e for e in ev if not (e["id"] in seen or seen.add(e["id"]))]
    return sorted(ev, key=lambda e: e["t"])

def write(out_dir, match_id, video, vinfo, per, frames_, ball, ballm, state, H, L, W, attack_right, conf, turnovers_, passes_, restarts_, sequences_, lanes_, shapes_, stats_, team_model, summary, log=print,
          frame_stride=1, split_s=None, copy_video=True, make_zip=True, video_url=None, periods=None):
    """frame_stride: keep every Nth frame for overlays (the app interpolates); split_s: write frames in files of this many
    seconds (full matches) instead of inside match_data.json; copy_video/make_zip off for full matches (video already in R2)."""
    root = os.path.join(out_dir, "matches", match_id); os.makedirs(root, exist_ok=True); fps = vinfo["fps"]; n = len(per)
    frames_out = []
    for k in range(0, n, max(1, int(frame_stride))):
        f = frames_[k]; sh = shapes_[k]
        frames_out.append({"t": round(k / fps, 3),
            "players": [{"id": int(r[0]), "team": r[1], "gk": bool(r[5]), "state": "observed", "conf": 1.0, "px": [round(float(r[3][0]), 1), round(float(r[3][1]), 1)], "m": [round(float(r[2][0]), 2), round(float(r[2][1]), 2)]} for r in per[k]],
            "ball": ({"px": [round(ball[k][0], 1), round(ball[k][1], 1)], "m": [round(float(ballm[k][0]), 2), round(float(ballm[k][1]), 2)] if k in ballm else None, "state": "observed"} if k in ball else None),
            "possession": STATES[state[k]] if state[k] < 2 else None, "phase": "control" if state[k] < 2 else STATES[state[k]],
            "carrier": f["carrier"], "pressure_m": f["pressure_m"], "near_opps": f["near_opps"], "shape": sh, "lanes": lanes_.get(k),
            "pitch_lines": [float(v) for v in H[k].ravel()]})
    ev = events(turnovers_, passes_, restarts_, sequences_, fps)
    if stats_.get('metrics'):
        from .metrics import extra_events
        ev = sorted(ev + extra_events(stats_['metrics']), key=lambda x: x['t'])
    md = {"schema_version": SCHEMA_VERSION, "match_id": match_id, "video": os.path.basename(video), "fps": fps, "width": vinfo["width"], "height": vinfo["height"], "pitch": {"length": L, "width": W},
          "teams": {"light": "A", "dark": "B"}, "periods": ([dict(p, attack_right=attack_right, confidence=conf) for p in periods] if periods else [{"index": 1, "t_start": 0.0, "t_end": round(n / fps, 2), "attack_right": attack_right, "confidence": conf}]),
          "attack_right": attack_right, "attack_right_confidence": conf, "kits": {"A": "kit_A.png", "B": "kit_B.png"}, "contract": "1.1", "frames": frames_out, "events": ev,
          "turnovers": turnovers_, "sequences": sequences_, "restarts": restarts_}
    import tempfile
    chunk_files = {}
    if split_s:
        md["frames"] = []; md["frame_chunks"] = []; per_file = max(1, int(round(split_s * fps / max(1, frame_stride))))
        for i in range(0, len(frames_out), per_file):
            part = frames_out[i:i + per_file]; key = f"frames_{i // per_file:03d}"
            json.dump({"t_start": part[0]["t"], "t_end": part[-1]["t"], "frames": part}, open(f"{root}/{key}.json", "w"), default=_json_default, separators=(",", ":"))
            md["frame_chunks"].append({"key": key, "t_start": part[0]["t"], "t_end": part[-1]["t"]}); chunk_files[key] = f"matches/{match_id}/{key}.json"
        log(f"  wrote {len(chunk_files)} frame files (every {frame_stride} frames, {split_s:.0f} s each, largest {max(os.path.getsize(f'{root}/{k}.json') for k in chunk_files)/1e6:.1f} MB)")
    tmp = os.path.join(tempfile.gettempdir(), f"{match_id}_match_data.json"); json.dump(md, open(tmp, "w"), default=_json_default); shutil.copy(tmp, f"{root}/match_data.json"); log(f"  wrote match_data.json ({os.path.getsize(tmp)/1e6:.1f} MB)")
    st = dict(stats_); st["passes"] = passes_; st["sequences"] = sequences_; st["restarts"] = restarts_; st["pitch"] = {"length": L, "width": W}
    json.dump(st, open(f"{root}/stats.json", "w"), default=_json_default)
    if team_model is not None and getattr(team_model, "strips", None):
        for ab, img in team_model.strips.items(): cv2.imwrite(f"{root}/kit_{ab}.png", img)
    from .video import frame_at
    th = frame_at(video, int(n * 0.3))
    if th is not None: cv2.imwrite(f"{root}/thumb.jpg", cv2.resize(th, (640, 360)))
    if copy_video: shutil.copy(video, f"{root}/{os.path.basename(video)}")
    json.dump(summary, open(f"{root}/summary.json", "w"), indent=1, default=_json_default)
    entry = {"id": match_id, "title": "Team A – Team B (label in app)", "date": None, "competition": None, "home": "A", "away": "B", "score": None, "duration_s": round(n / fps, 1),
             "thumbnail": f"matches/{match_id}/thumb.jpg", "schema_version": SCHEMA_VERSION,
             "files": {"video": f"matches/{match_id}/{os.path.basename(video)}", "match_data": f"matches/{match_id}/match_data.json", "stats": f"matches/{match_id}/stats.json", "kit_A": f"matches/{match_id}/kit_A.png", "kit_B": f"matches/{match_id}/kit_B.png", **chunk_files},
             "video_url": video_url,
             "status": "ready", "tags": [], "attack_right": attack_right, "attack_right_confidence": conf, "labels": None, "summary": summary}
    lib_path = os.path.join(out_dir, "library.json"); lib = json.load(open(lib_path)) if os.path.exists(lib_path) else {"matches": []}
    lib["matches"] = [m for m in lib["matches"] if m["id"] != match_id] + [entry]; json.dump(lib, open(lib_path, "w"), indent=1, default=_json_default)
    zpath = None
    if make_zip:
        stage = os.path.join(tempfile.gettempdir(), f"{match_id}_lovable"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(os.path.join(stage, "matches"))
        shutil.copytree(root, os.path.join(stage, "matches", match_id)); json.dump({"matches": [entry]}, open(os.path.join(stage, "library.json"), "w"), indent=1, default=_json_default)
        zlocal = shutil.make_archive(os.path.join(tempfile.gettempdir(), f"{match_id}_lovable"), "zip", root_dir=stage, base_dir=".")
        zpath = os.path.join(out_dir, f"{match_id}_lovable.zip"); shutil.copy(zlocal, zpath)
    log(f"export: {root} ({len(ev)} events) · library.json updated · zip {zpath}")
    return root, zpath
