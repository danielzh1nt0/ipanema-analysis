"""Possession as a 4-state HMM over the whole clip, plus everything that follows from it."""
import numpy as np
from .calibration import to_m

STATES = ["A", "B", "loose", "dead"]

def carriers(per, ball, H, carrier_r=2.5, near_r=5.0):
    n = len(per); frames_ = []; ballm = {}
    for k in range(n):
        rec = {"frame": k, "carrier": None, "team": None, "pressure_m": None, "near_opps": None, "pos": None}
        if k in ball:
            bm = to_m(H[k], [ball[k]])[0]; ballm[k] = bm; rows = per[k]
            if rows:
                d = [np.linalg.norm(r[2] - bm) for r in rows]; ci = int(np.argmin(d))
                if d[ci] <= carrier_r:
                    tid, tm, pos = rows[ci][0], rows[ci][1], rows[ci][2]; opps = [r[2] for r in rows if r[1] != tm]
                    od = [np.linalg.norm(o - pos) for o in opps]
                    rec.update({"carrier": tid, "team": tm, "pos": pos.tolist(), "pressure_m": round(float(min(od)), 2) if od else None, "near_opps": int(sum(x <= near_r for x in od))})
        frames_.append(rec)
    return frames_, ballm

def viterbi(per, ballm, fps, L, W, max_speed=35.0, flight_speed=8.0):
    n = len(per)
    # 1) impossible jumps are ball-pick errors: drop those frames from the observations
    bad = set()
    for k in sorted(ballm):
        if k - 1 in ballm and np.linalg.norm(ballm[k] - ballm[k - 1]) * fps > max_speed: bad.add(k)
    for k in bad: ballm.pop(k, None)
    bspeed = {k: float(np.linalg.norm(ballm[k] - ballm[k - 1]) * fps) for k in ballm if k - 1 in ballm}
    E = np.zeros((n, 4))
    for k in range(n):
        if k not in ballm: continue
        bm = ballm[k]; onpitch = (-1 <= bm[0] <= L + 1) and (-1 <= bm[1] <= W + 1); d = {}
        for tm in ("A", "B"):
            ds = [np.linalg.norm(r[2] - bm) for r in per[k] if r[1] == tm]; d[tm] = min(ds) if ds else 30.0
        s = bspeed.get(k, 0.0); flight = 1.0 if s > flight_speed else 0.0
        # 2) a ball in flight belongs to nobody: control is (almost) forbidden while it travels
        E[k, 0] = 0.6 * max(0.0, d["A"] - 2.5) + 6.0 * flight
        E[k, 1] = 0.6 * max(0.0, d["B"] - 2.5) + 6.0 * flight
        E[k, 2] = 2.2 - 2.0 * flight + (1.2 if min(d.values()) < 2.5 else 0.0)
        E[k, 3] = 0.5 if not onpitch else (1.0 if (s < 0.5 and min(d.values()) > 3.0) else 4.0)
    SW = np.array([[0, 8.0, 4.0, 3.0], [8.0, 0, 4.0, 3.0], [4.0, 4.0, 0, 3.0], [3.0, 3.0, 3.0, 0]])
    D = np.full((n, 4), np.inf); B = np.zeros((n, 4), int); D[0] = E[0]
    for k in range(1, n):
        for s in range(4):
            c = D[k - 1] + SW[:, s]; B[k, s] = int(np.argmin(c)); D[k, s] = c[B[k, s]] + E[k, s]
    state = np.zeros(n, int); state[-1] = int(np.argmin(D[-1]))
    for k in range(n - 1, 0, -1): state[k - 1] = B[k, state[k]]
    # post-process: control runs shorter than 0.6 s that sit between the same team's runs become that team; loose gaps < 0.8 s inside one team's control are absorbed
    min_run = int(1.5 * fps); i = 0
    while i < n:
        j = i
        while j + 1 < n and state[j + 1] == state[i]: j += 1
        if state[i] in (0, 1) and (j - i + 1) < min_run:
            prev_s = state[i - 1] if i > 0 else None; next_s = state[j + 1] if j + 1 < n else None
            if prev_s == next_s and prev_s is not None: state[i:j + 1] = prev_s
        if state[i] == 2 and (j - i + 1) < int(0.8 * fps):
            prev_s = state[i - 1] if i > 0 else None; next_s = state[j + 1] if j + 1 < n else None
            if prev_s == next_s and prev_s in (0, 1): state[i:j + 1] = prev_s
        i = j + 1
    return state, bspeed

def direction(state, ballm, log=print):
    net, gross = {"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0}
    for k in sorted(ballm):
        if k - 1 not in ballm: continue
        st = STATES[state[k]]
        if st in ("A", "B") and STATES[state[k - 1]] == st:
            dx = float(ballm[k][0] - ballm[k - 1][0])
            if abs(dx) < 3.0: net[st] += dx; gross[st] += abs(dx)
    conf = {tm: (abs(net[tm]) / gross[tm] if gross[tm] > 0 else 0.0) for tm in net}
    right = {tm: net[tm] > 0 for tm in net}
    if right["A"] == right["B"]:
        lead = max(conf, key=conf.get); other = "B" if lead == "A" else "A"; right[other] = not right[lead]
    log(f"direction: A {'→' if right['A'] else '←'}  B {'→' if right['B'] else '←'}  confidence A {conf['A']:.2f} / B {conf['B']:.2f}" + ("  WARNING low confidence" if min(conf.values()) < 0.15 else ""))
    return right, {k: round(v, 2) for k, v in conf.items()}

def direction_from_keepers(per, L, log=print):
    """the goal a team's keeper stands at is the goal it defends"""
    votes = {"A": {"left": 0, "right": 0}, "B": {"left": 0, "right": 0}}
    for k in per:
        for r in per[k]:
            if r[5]: votes[r[1]]["left" if r[2][0] < L / 2 else "right"] += 1
    right = {}
    for tm in ("A", "B"):
        v = votes[tm]
        if v["left"] + v["right"] < 10: return None
        right[tm] = v["left"] > v["right"]        # defends left -> attacks right
    if right["A"] == right["B"]: return None
    log(f"direction (from keepers): A {'→' if right['A'] else '←'}  B {'→' if right['B'] else '←'}"); return right

def direction_fallback(per, L, log=print):
    """If the ball signal is weak: the team whose players sit nearer the left goal on average defends it (attacks right)."""
    med = {}
    for tm in ("A", "B"):
        xs = [r[2][0] for k in per for r in per[k] if r[1] == tm and not r[5]]; med[tm] = float(np.median(xs)) if xs else L / 2
    right = {"A": med["A"] < med["B"], "B": med["A"] >= med["B"]}
    log(f"direction (fallback from player positions): A {'→' if right['A'] else '←'}  B {'→' if right['B'] else '←'}  median x A {med['A']:.0f} / B {med['B']:.0f}")
    return right

def sequences(state, ballm, bspeed, fps, L, attack_right):
    n = len(state); seqs = []; cur = None
    for k in range(n):
        st = STATES[state[k]]
        if st in ("A", "B"):
            if cur and cur["team"] == st and k - cur["end"] <= int(fps): cur["end"] = k
            else:
                if cur: seqs.append(cur)
                cur = {"team": st, "start": k, "end": k}
    if cur: seqs.append(cur)
    for i, sq in enumerate(seqs):
        nxt = seqs[i + 1] if i + 1 < len(seqs) else None; after = STATES[state[min(n - 1, sq["end"] + 1)]]
        sp = max((bspeed.get(j, 0.0) for j in range(sq["end"], min(n, sq["end"] + int(0.4 * fps)))), default=0.0)
        if after == "dead": reason = "out of play / stoppage"
        elif nxt and nxt["team"] != sq["team"] and nxt["start"] - sq["end"] <= int(2 * fps): reason = "pass intercepted" if sp > 9 else "dispossessed"
        elif nxt is None: reason = "clip ends"
        else: reason = "loose ball"
        b0, b1 = ballm.get(sq["start"]), ballm.get(sq["end"]); sgn = 1 if attack_right[sq["team"]] else -1
        gain = float((b1[0] - b0[0]) * sgn) if (b0 is not None and b1 is not None) else None
        sq.update({"t_start": round(sq["start"] / fps, 2), "t_end": round(sq["end"] / fps, 2), "duration_s": round((sq["end"] - sq["start"] + 1) / fps, 2), "end_reason": reason,
                   "gain_m": round(gain, 1) if gain is not None else None, "reached_final_third": bool(b1 is not None and ((b1[0] > 2 * L / 3) if sgn == 1 else (b1[0] < L / 3)))})
    return seqs

def restarts(state, ballm, fps, L, W):
    n = len(state); out = []; k = 0
    while k < n:
        if STATES[state[k]] == "dead":
            j = k
            while j + 1 < n and STATES[state[j + 1]] == "dead": j += 1
            if (j - k + 1) / fps >= 0.8:
                bm = next((ballm[q] for q in range(j, min(n, j + int(1.5 * fps))) if q in ballm), None)
                team_after = next((STATES[state[q]] for q in range(j + 1, min(n, j + int(3 * fps))) if state[q] < 2), None); kind = "unknown"
                if bm is not None:
                    x, y = bm; near_end = x < 3 or x > L - 3; near_side = y < 3 or y > W - 3
                    kind = "corner" if (near_end and near_side) else "goal kick" if (near_end and abs(y - W / 2) < 12) else "throw-in" if near_side else "corner" if near_end else "free kick"
                out.append({"t": round(j / fps, 2), "kind": kind, "team": team_after, "x_m": round(float(bm[0]), 1) if bm is not None else None, "y_m": round(float(bm[1]), 1) if bm is not None else None})
            k = j + 1
        else: k += 1
    return out

def turnovers(per, frames_, state, ballm, fps, attack_right, press_r=2.0, near_r=5.0, fwd_m=5.0, react_s=2.0):
    n = len(per); poss = [STATES[s] if s < 2 else None for s in state]; out = []; last = None
    for k in range(1, n):
        cur = poss[k]
        if cur is None: continue
        if last is not None and cur != last:
            loser, winner = last, cur; k0 = k
            hold = 0
            for j in range(k0, min(n, k0 + int(3.0 * fps))):
                if poss[j] == winner: hold += 1
                elif poss[j] == loser: break
            prev_ctrl = 0
            for j in range(k0 - 1, max(-1, k0 - int(6.0 * fps)), -1):
                if poss[j] == loser: prev_ctrl += 1
                elif poss[j] == winner: break
            # 3) two real possessions: loser had it >= 2 s, winner keeps it >= 2 s
            if hold < int(2.0 * fps) or prev_ctrl < int(2.0 * fps): last = cur; continue
            tv = {"frame": k0, "t": round(k0 / fps, 2), "lost_by": loser, "won_by": winner, "time_to_press": None, "near_at_2s": None, "regained_within_5s": False, "time_to_forward_pass": None, "ball_m_before_press": None, "gain_5s_m": None, "lost_back_5s": False}
            for j in range(k0, min(n, k0 + int(8 * fps))):
                f = frames_[j]
                if f["team"] == winner and f["pressure_m"] is not None and f["pressure_m"] <= press_r: tv["time_to_press"] = round((j - k0) / fps, 2); break
            j2 = min(n - 1, k0 + int(2 * fps)); tv["near_at_2s"] = frames_[j2]["near_opps"] if frames_[j2]["team"] == winner else None
            tv["regained_within_5s"] = any(poss[j] == loser for j in range(k0, min(n, k0 + int(5 * fps))))
            tv["lost_back_5s"] = tv["regained_within_5s"]
            prev = None
            for j in range(k0, min(n, k0 + int(8 * fps))):
                f = frames_[j]
                if f["team"] == winner and f["carrier"] is not None:
                    if prev is not None and f["carrier"] != prev["carrier"]:
                        gain = (f["pos"][0] - prev["pos"][0]) * (1 if attack_right[winner] else -1)
                        if gain >= fwd_m: tv["time_to_forward_pass"] = round((j - k0) / fps, 2); break
                    prev = f
                elif poss[j] not in (None, winner): break
            b0 = ballm.get(k0)
            if tv["time_to_press"] is not None and b0 is not None:
                bp = ballm.get(k0 + int(tv["time_to_press"] * fps))
                if bp is not None: tv["ball_m_before_press"] = round(float(np.linalg.norm(bp - b0)), 1)
            b5 = ballm.get(min(n - 1, k0 + int(5 * fps)))
            if b0 is not None and b5 is not None: tv["gain_5s_m"] = round(float((b5[0] - b0[0]) * (1 if attack_right[winner] else -1)), 1)
            # reactions of the losing team over the first 2 s
            k2 = min(n - 1, k0 + int(react_s * fps)); b2 = ballm.get(k2)
            if b0 is not None and b2 is not None:
                p0 = {r[0]: r[2] for r in per[k0] if r[1] == loser}; p2 = {r[0]: r[2] for r in per[k2] if r[1] == loser}
                rx = {"pressed": [], "jogged": [], "stood": []}
                for tid in p0:
                    if tid not in p2: continue
                    d0, d2 = np.linalg.norm(p0[tid] - b0), np.linalg.norm(p2[tid] - b2); closing = (d0 - d2) / react_s
                    rx["pressed" if (d2 <= near_r or closing >= 2.0) else "jogged" if closing >= 0.7 else "stood"].append(int(tid))
                tv["reactions"] = rx
            out.append(tv)
        last = cur
    return out
