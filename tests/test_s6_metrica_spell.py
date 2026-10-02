"""S6 (2 Oct): the Metrica check counts sequences / balls lost on the spell state, like ipanema/run.py, and its
'flicker' noise makes the raw state switch teams often while the spell state stays calm. Synthetic, no Metrica files."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import metricalab as ML


def _frames(n=25 * 60):
    """one minute: home #101 dribbles along the pitch with the ball at his feet, away #201 runs 3 m beside him"""
    out = {}
    for k in range(n):
        x = 30 + 0.1 * k / 25 * 25 / 5
        out[k] = (1, [(101, x, 34.0), (201, x, 37.0), (102, 10.0, 10.0), (202, 90.0, 60.0)], (x + 0.3, 34.0))
    return out


def test_switches_per_min_skips_loose():
    st = np.array([0] * 25 + [2] * 25 + [1] * 25 + [3] * 25 + [1] * 25)   # A, loose, B, dead, B = one switch
    assert ML.switches_per_min(st, 25.0) == round(1 / (125 / 25 / 60), 1)


def test_clean_one_carrier_is_one_sequence():
    O = ML.ours(_frames(), 1, False)
    assert O["sequences"] == 1 and O["switches_per_min"] == 0


def test_flicker_raw_switches_but_spell_holds():
    raw = ML.ours(_frames(), 1, "flicker", spell_take=0)
    spell = ML.ours(_frames(), 1, "flicker")                               # pipeline default (P.SPELL_TAKE_S)
    assert raw["switches_per_min_raw"] >= 10                               # the hop noise really flickers the state
    assert spell["switches_per_min"] < raw["switches_per_min_raw"] / 3
    assert spell["sequences"] < raw["sequences"]
