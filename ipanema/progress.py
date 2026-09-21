"""Live analysis progress for the app: reads the pipeline's own log lines and writes stage, percent and an ETA
into matches.summary.progress (only while the match is still 'processing', so finished matches are never touched)."""
import os, re, time
from datetime import datetime, timezone

STAGES = [("Reading the pitch", 0.15), ("Finding players", 0.40), ("Following the ball", 0.25), ("Working out possession", 0.10), ("Writing findings", 0.10)]

class Progress:
    def __init__(self, match_id, every_s=20):
        self.row = re.sub(r"_s\d+$", "", match_id); self.segment = match_id if self.row != match_id else None
        self.t0 = time.time(); self.every = every_s; self.last = 0.0; self.stage = 0; self.frac = 0.0; self.frames = None; self.db = None
        try:
            if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
                from supabase import create_client
                self.db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        except Exception: self.db = None

    def _set(self, stage, frac):
        if stage < self.stage: return
        if stage > self.stage: self.stage, self.frac = stage, 0.0
        self.frac = max(self.frac, min(1.0, frac))

    def feed(self, line):
        s = str(line); before = self.stage
        m = re.search(r"(\d+) frames \(", s)
        if m: self.frames = int(m.group(1))
        n = self.frames or 0
        m = re.search(r"calibration frame (\d+)", s)
        if m: self._set(0, int(m.group(1)) / max(n, 1))
        elif s.startswith("calibration:"): self._set(0, 1.0)
        if s.startswith("teams:"): self._set(1, 0.08)
        m = re.search(r"tracking frame (\d+)", s)
        if m: self._set(1, 0.08 + 0.9 * int(m.group(1)) / max(n, 1))
        if s.startswith("clean:") or "tracking: cached" in s: self._set(1, 1.0)
        m = re.search(r"(?:wasb|ball) frame (\d+)", s)
        if m: self._set(2, int(m.group(1)) / max(n, 1))
        if s.startswith("ball check") or s.startswith("ball (global") or s.startswith("ball:"): self._set(2, 0.95)
        if s.startswith("direction"): self._set(3, 0.1)
        m = re.search(r"step: (\w+)", s)
        if m:
            order = ["sequences", "turnovers", "lanes", "passes", "shapes", "stats", "metrics"]
            if m.group(1) in order: self._set(3, (order.index(m.group(1)) + 1) / len(order))
            if m.group(1) == "export": self._set(4, 0.3)
        if s.startswith("upload:"): self._set(4, 0.9)
        if self.stage != before or time.time() - self.last > self.every: self.push()

    def overall(self):
        return sum(w for _, w in STAGES[: self.stage]) + STAGES[self.stage][1] * self.frac

    def push(self, extra=None):
        self.last = time.time()
        if not self.db: return
        el = time.time() - self.t0; ov = self.overall()
        eta = (el / ov - el) if ov > 0.05 else None
        prog = {"stage": self.stage, "label": STAGES[self.stage][0], "pct": round(ov * 100), "elapsed_s": int(el),
                "eta_s": int(eta) if eta is not None else None, "updated_at": datetime.now(timezone.utc).isoformat(),
                "scope": "5-minute test" if self.segment else "match"}
        if extra: prog.update(extra)
        try: self.db.table("matches").update({"summary": {"progress": prog}}).eq("id", self.row).eq("status", "processing").execute()
        except Exception as e: print("progress update failed:", e)

    def finished(self, ok):
        if self.segment and ok: self.stage, self.frac = 4, 1.0; self.push({"segment_ready": self.segment, "label": "5-minute test ready"})
        elif not ok: self.push({"error": True})
