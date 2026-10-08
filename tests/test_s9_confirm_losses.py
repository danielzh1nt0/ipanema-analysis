"""S9 (8 Oct): a duel is a lost ball only when the winner keeps it >= 3 s or plays a pass (P.confirm_losses)."""
import numpy as np
from ipanema import possession as P

FPS = 10.0

def tv(k, lo, wi):
    return {"frame": k, "frame_lost": k - 1, "t": (k - 1) / FPS, "t_won": k / FPS, "lost_by": lo, "won_by": wi}

def state(*runs):
    return np.concatenate([np.full(n, v) for v, n in runs])

def test_duel_counts_zero():
    st = state((0, 100), (1, 20), (0, 100))                     # A, B for 2 s, A again
    kept, dropped = P.confirm_losses([tv(100, "A", "B"), tv(120, "B", "A")], st, [], FPS)
    assert kept == [] and len(dropped) == 2 and dropped[1]["dropped"].startswith("duel, ball straight back")

def test_real_loss_kept():
    st = state((0, 100), (1, 100))
    kept, dropped = P.confirm_losses([tv(100, "A", "B")], st, [], FPS)
    assert len(kept) == 1 and kept[0]["confirmed"] == "hold" and dropped == []

def test_short_spell_with_pass_kept():
    st = state((0, 100), (1, 20), (0, 100))
    kept, _ = P.confirm_losses([tv(100, "A", "B"), tv(120, "B", "A")], st, [{"t": 10.8, "team": "B"}], FPS)
    assert [k["confirmed"] for k in kept] == ["pass", "hold"]

def test_other_team_pass_does_not_count():
    st = state((0, 100), (1, 20), (0, 100))
    kept, _ = P.confirm_losses([tv(100, "A", "B"), tv(120, "B", "A")], st, [{"t": 10.8, "team": "A"}], FPS)
    assert kept == []

def test_loose_frames_count_as_kept_and_clip_end():
    st = state((0, 100), (1, 10), (2, 15))                     # B 1 s then loose to the end (2.5 s): loser never regains
    kept, _ = P.confirm_losses([tv(100, "A", "B")], st, [], FPS)
    assert len(kept) == 1 and kept[0]["confirmed"] == "clip end"
    st = state((0, 100), (1, 10), (2, 25), (0, 50))            # loose 2.5 s then A: winner 'kept' it 3.5 s
    kept, _ = P.confirm_losses([tv(100, "A", "B"), tv(135, "B", "A")], st, [], FPS)
    assert len(kept) == 2

def test_hold_counts_from_the_change_not_the_late_win_time():
    st = state((0, 100), (1, 25), (0, 100))                     # B 2.5 s; turnovers() put the win 2 s after the change
    t = tv(102, "A", "B"); t["frame_lost"] = 99; t["frame"] = 120
    kept, dropped = P.confirm_losses([t], st, [], FPS)
    assert kept == [] and dropped[0]["winner_kept_s"] == 2.5

def test_off_switch():
    st = state((0, 100), (1, 20), (0, 100)); tvs = [tv(100, "A", "B"), tv(120, "B", "A")]
    kept, dropped = P.confirm_losses(tvs, st, [], FPS, keep_s=0)
    assert len(kept) == 2 and dropped == []

def test_run_wires_it():
    src = open("ipanema/run.py").read()
    assert "P.confirm_losses(tvs, cstate, ps, fps" in src and "IPANEMA_LOSS_KEEP_S" in src
