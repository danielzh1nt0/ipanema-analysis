"""3-4 Oct: Veo shots/goals outside the analysed video or its match periods are left out of a half's export."""
from ipanema.run import veo_in_window
P = [{"t_start": 555.0, "t_end": 3060.0}]

def test_before_kickoff_and_second_half_dropped():
    assert not veo_in_window(28.0, 3300.0, P)          # SFK-BP: 'goal' clip at 0:28, before kick-off
    assert not veo_in_window(3584.0, 3300.0, P)        # second half, past the cut video
    assert not veo_in_window(3200.0, 3300.0, P)        # inside the video, after the half ended (+5 s slack)

def test_inside_kept():
    assert veo_in_window(610.0, 3300.0, P) and veo_in_window(2619.0, 3300.0, P) and veo_in_window(3064.0, 3300.0, P)
    assert veo_in_window(28.0, 3300.0, None) and not veo_in_window(4000.0, 3300.0, None)   # no periods: only the video length counts
