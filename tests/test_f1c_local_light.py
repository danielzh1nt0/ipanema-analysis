"""F1c: shadow-aware kit lightness (kits.torso_feature light='local', kits.local_grass_L)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as KT

SUN, SHADE = (60, 165, 75), (22, 62, 28)                 # BGR grass in sun / in hard shadow

def frame():
    f = np.zeros((400, 800, 3), np.uint8); f[:, :400] = SUN; f[:, 400:] = SHADE; return f

def person(f, x, shirt):
    f[200:260, x:x + 30] = shirt; return [x, 190, x + 30, 290]   # torso inside the box, legs = grass

def test_local_grass_reads_the_light_at_the_feet():
    f = frame(); bs = person(f, 150, (40, 40, 40)); bh = person(f, 600, (40, 40, 40))
    ls, lh = KT.local_grass_L(f, bs), KT.local_grass_L(f, bh)
    assert ls is not None and lh is not None and ls > lh + 40

def test_white_in_shadow_reads_lighter_than_dark_in_sun_only_with_local_light():
    f = frame()
    dark_sun = person(f, 150, (95, 95, 95))                # a dark shirt in bright sun
    white_shade = person(f, 600, (85, 85, 85))             # a white shirt in hard shadow: darker in the raw picture
    raw = [KT.torso_feature(f, b, light="frame")[0] for b in (dark_sun, white_shade)]
    loc = [KT.torso_feature(f, b, light="local")[0] for b in (dark_sun, white_shade)]
    assert raw[0] > raw[1]                                 # the old reading gets it backwards
    assert loc[1] > loc[0] + 10                            # relative to the grass it is the lighter shirt

def test_default_is_unchanged():
    os.environ.pop("IPANEMA_KIT_LIGHT", None)
    f = frame(); b = person(f, 600, (85, 85, 85))
    assert np.allclose(KT.torso_feature(f, b), KT.torso_feature(f, b, light="frame"))

def test_no_grass_around_falls_back():
    f = np.full((200, 200, 3), (128, 128, 128), np.uint8); b = [80, 40, 110, 140]
    assert KT.local_grass_L(f, b) is None
    assert np.allclose(KT.torso_feature(f, b, light="local", grass=np.array([150., 110., 140.])),
                       KT.torso_feature(f, b, light="frame", grass=np.array([150., 110., 140.])))

def test_body_hist_local_mode_runs():
    f = frame(); f[330:] = SUN; b = person(f, 600, (85, 85, 85))           # frame grass = sun, this player in shadow
    h1, h2 = KT.body_hist(f, b, light="frame"), KT.body_hist(f, b, light="local")
    assert h1.shape == h2.shape and not np.allclose(h1, h2)

def _match(shaded):
    """4 frames, 6 dark + 6 white players each, half of them standing in hard shadow when shaded"""
    rng = np.random.default_rng(0); out = []
    for k in range(4):
        f = np.zeros((400, 1600, 3), np.uint8); f[:] = SUN
        if shaded: f[:, 960:] = SHADE                   # 40% of the frame in shadow
        boxes = []
        for i in range(12):
            x = 40 + i * 130; dark = i % 2 == 0; lit = x < 940 or not shaded
            c = (40, 40, 45) if dark else (235, 235, 235)
            c = tuple(int(np.clip(v + rng.integers(-12, 13), 0, 255)) for v in c)       # no two shirts read exactly alike
            if not lit: c = tuple(int(v * 0.38) for v in c)
            boxes.append(person(f, x + int(rng.integers(0, 20)), c))
        out.append((f, np.array(boxes, float)))
    return out

def test_shade_spread_separates_sun_and_shade_from_even_light():
    sh = KT.shade_spread([(f, b, KT.grass_lab(f)) for f, bs in _match(True) for b in bs])
    ev = KT.shade_spread([(f, b, KT.grass_lab(f)) for f, bs in _match(False) for b in bs])
    assert sh >= KT.SHADE_SPREAD > ev

def test_auto_picks_local_only_in_sun_and_shade():
    os.environ.pop("IPANEMA_KIT_LIGHT", None)
    q = lambda *a: None
    assert KT.KitTeamModel().fit_frames(_match(True), log=q, player_cls=False).light == "local"
    assert KT.KitTeamModel().fit_frames(_match(False), log=q, player_cls=False).light == "frame"
    os.environ["IPANEMA_KIT_LIGHT"] = "frame"
    try: assert KT.KitTeamModel().fit_frames(_match(True), log=q, player_cls=False).light == "frame"
    finally: os.environ.pop("IPANEMA_KIT_LIGHT", None)
