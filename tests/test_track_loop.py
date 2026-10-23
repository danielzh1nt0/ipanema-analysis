"""N0-21b (21 Oct): tracking.track - the app's player loop (detect -> ByteTrack -> team vote -> metres) had no test.
Fake detector + fake kit model, the real supervision ByteTrack, a tiny synthetic video. Checks: one stable id per person,
team = majority of the readings (a 'neither' now and then does not move a player), the referee and off-pitch people
are not players, feet go to metres through H, the kit check runs every 3rd frame (+ new tracks), and RAW_TEAM (this
frame's own reading, used by tools/tracktest.py) starts empty in every run."""
import sys, types, tempfile, numpy as np, cv2, pytest

sv = pytest.importorskip("supervision")
import ipanema.tracking as T

N = 45
# x position of each person -> what the kit model reads (a string = always; a list = per check, cycled)
PEOPLE = {100: "A", 300: "B", 500: "R", 700: "O", 900: ["A", "A", "K"], 1100: "K", 1300: ["B", "R", "B", "B"]}


def _video(n=N):
    p = f"{tempfile.mkdtemp()}/v.mp4"; vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for _ in range(n): vw.write(np.zeros((1080, 1920, 3), np.uint8))
    vw.release(); return p


class FakeDetector:
    def detect_batch(self, fs, conf, tiles):
        out = []
        for _ in fs:
            xy = np.array([[x - 15, 600, x + 15, 680] for x in PEOPLE], float)
            out.append((sv.Detections(xyxy=xy, confidence=np.full(len(xy), 0.9), class_id=np.zeros(len(xy), int)), {0: "person"}))
        return out


class FakeKits:
    def __init__(self): self.calls = 0; self.seen = {}
    def predict_batch(self, f, xyxy):
        self.calls += 1; out = []
        for b in xyxy:
            x = int(round((b[0] + b[2]) / 2)); r = PEOPLE[x]
            if isinstance(r, list): i = self.seen.get(x, 0); self.seen[x] = i + 1; r = r[i % len(r)]
            out.append(r)
        return out


def _run(monkeypatch, kits=None):
    monkeypatch.setenv("IPANEMA_DETECTOR", "yolo")
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda w: FakeDetector()))   # restored after the test (a leftover fake broke later tests when ultralytics is installed)
    H = {k: np.diag([10.0, 10.0, 1.0]) for k in range(N)}          # pixels = 10 x metres
    kits = kits or FakeKits()
    per, fps = T.track(_video(), "w.pt", H, kits, 0.3, log=lambda *a: None, tiles=["any"], pano=False)
    return per, fps, kits


def test_ids_teams_and_metres(monkeypatch):
    per, fps, kits = _run(monkeypatch)
    assert round(fps) == 30 and sorted(per) == list(range(N))
    last = {int(round(r[3][0])): r for r in per[N - 1]}
    assert set(last) == {100, 300, 900, 1100, 1300}                 # referee (500) and off-pitch (700) are not players
    assert last[100][1] == "A" and last[300][1] == "B"
    assert last[900][1] == "A"                                       # reads 'neither' 1 in 3: still team A
    assert last[1100][1] == "K"                                      # always 'neither': K (clean() drops it later)
    assert last[1300][1] == "B"                                      # one 'R' in 4 (< 30%): still a B player
    r = last[100]; assert np.allclose(r[2], [10.0, 68.0]) and np.allclose(r[3], [100, 680]) and r[5] is False
    ids = {}
    for k in range(5, N):                                            # ByteTrack needs a couple of frames to confirm
        for r in per[k]: ids.setdefault(int(round(r[3][0])), set()).add(r[0])
    assert all(len(v) == 1 for v in ids.values()) and len({next(iter(v)) for v in ids.values()}) == len(ids)


def test_kit_check_every_third_frame(monkeypatch):
    _, _, kits = _run(monkeypatch)
    # one batched call per checked frame: frames 0, 3, 6, ... plus frames where a track is new
    assert N // 3 <= kits.calls <= N // 3 + 4


def test_raw_team_starts_empty(monkeypatch):
    T.RAW_TEAM[(10 ** 6, 1)] = "stale"                               # left over from an earlier run in the same process
    per, _, _ = _run(monkeypatch)
    assert (10 ** 6, 1) not in T.RAW_TEAM
    tid = next(r[0] for r in per[N - 1] if int(round(r[3][0])) == 100)
    assert T.RAW_TEAM[(N - 1 - (N - 1) % 3, tid)] == "A"
