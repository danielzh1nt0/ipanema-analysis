"""4 Oct: a goal whose team was checked by eye ("2619 goal A") keeps that team whatever the ball/direction guess says."""
import numpy as np
from ipanema import veo as VEO

def test_known_team_overrides(tmp_path):
    p = tmp_path / "h.txt"; p.write_text("100 shot\n2619 goal A\n2619 shot\n")
    hl, kt = VEO.load(str(p)), VEO.teams(str(p))
    assert kt == {2619: "A"}
    ballm = {int(2619 * 30) + d: np.array([12.0, 30.0]) for d in range(-30, 30)}            # ball at the left end
    out, _ = VEO.build(hl, 30.0, ballm, {}, 106.0, 64.0, {"A": True, "B": False}, [], known_teams=kt, log=lambda *a: None)
    g = [s for s in out if s["goal"]][0]
    assert g["team"] == "A" and "by eye" in g["located_by"]                                    # the guess alone would say B

def test_shot_placed_after_clip_start():
    """a Veo clip starts ~5-25 s before the shot: play at the left end at the clip start, the shot at the right end later"""
    fps = 10.0; ballm = {}
    for k in range(0, 50): ballm[1000 + k] = np.array([15.0, 30.0])            # clip start: ball in the left third
    for k in range(60, 250): ballm[1000 + k] = np.array([90.0 + (k % 10), 32.0])  # 6-25 s later: attack at the right end
    out, _ = VEO.build([(100, "goal")], fps, ballm, {}, 106.0, 64.0, {"A": True, "B": False}, [], log=lambda *a: None)
    assert out[0]["team"] == "A" and out[0]["x_m"] is None and out[0]["located_by"] == "end only"   # team from the end; no invented origin

def test_origin_by_eye(tmp_path):
    p = tmp_path / "h.txt"; p.write_text("3604 goal A 100.7 27.0\n3604 shot\n")
    out, _ = VEO.build(VEO.load(str(p)), 30.0, {}, {}, 106.0, 64.0, {"A": True, "B": False}, [], known_teams=VEO.teams(str(p)), known_origins=VEO.origins(str(p)), log=lambda *a: None)
    g = out[0]; assert g["team"] == "A" and g["x_m"] == 100.7 and g["y_m"] == 27.0 and 7 < g["distance_m"] < 7.5
