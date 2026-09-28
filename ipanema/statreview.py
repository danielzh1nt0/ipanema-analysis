"""28 Sep: answer keys for the stats from Daniel's yes/no pictures (no Veo data, no timing from him).
1. every stoppage we detected: 3 pictures (2 s before, at, 2 s after) -> "was play stopped here?"
2. n random moments spread over the clip -> "who has the ball?" (our answer is NOT drawn, so the check is unbiased)"""
import random, base64, numpy as np, cv2

def _jpg(img, q=78): return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()

def plan(md, n_poss=40, seed=0, min_gap_s=4.0):
    fps = md["fps"]; nf = len(md["frames"]); dur = nf / fps
    stops = [{"kind": "stoppage", "t": float(r["t"]), "ours": r.get("kind"), "team": r.get("team")} for r in md.get("restarts", [])]
    rng = random.Random(seed); cand = list(range(int(fps * 2), nf - int(fps * 2), int(fps))); rng.shuffle(cand); pick = []
    for k in cand:
        if all(abs(k - p) >= min_gap_s * fps for p in pick): pick.append(k)
        if len(pick) >= n_poss: break
    poss = [{"kind": "possession", "t": round(k / fps, 2), "frame": k, "ours": md["frames"][k].get("possession")} for k in sorted(pick)]
    return stops, poss, dur

def pictures(get_frame, md, stops, poss, fps):
    """get_frame(k) -> BGR image or None"""
    out = []
    for s in stops:
        tiles = []
        for dt, lab in ((-2.0, "2 s before"), (0.0, "our stoppage"), (2.0, "2 s after")):
            k = int(round((s["t"] + dt) * fps)); f = get_frame(max(0, min(k, len(md["frames"]) - 1)))
            if f is None: continue
            f = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(f, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5); cv2.putText(f, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            tiles.append(f)
        if tiles: out.append(dict(s, img=_jpg(np.vstack(tiles))))
    for p in poss:
        f = get_frame(p["frame"])
        if f is None: continue
        out.append(dict(p, img=_jpg(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA), 82)))
    return out
