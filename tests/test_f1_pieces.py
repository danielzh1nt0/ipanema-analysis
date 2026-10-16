"""F1 (16 Oct): a 5-min piece of a labelled clip (<match>_s<start>_c<k>) finds its match's line rows at start + k*300 s;
the fp16 switch is a no-op unless IPANEMA_RF_FP16=1 on a GPU."""
import contextlib
import pytest
from ipanema import linecal as LC


def test_piece_of_clip_offset():
    base, off, path = LC.find_rows("/nonexistent", "SFKBP1109_s1200_c002")
    assert base == "SFKBP1109" and off == 1200.0 + 2 * 300
    assert path.endswith("SFKBP1109_lines_match.json")


def test_piece_of_full_match_and_clip_unchanged():
    assert LC.find_rows("/nonexistent", "SFKBP1109_c003")[:2] == ("SFKBP1109", 900.0)
    assert LC.find_rows("/nonexistent", "SFKBP1109_s1200")[:2] == ("SFKBP1109", 1200.0)
    assert LC.find_rows("/nonexistent", "unknown_s60_c001") is None


def test_autocast_off_by_default(monkeypatch):
    pytest.importorskip("torch")
    from ipanema import ballrf
    monkeypatch.delenv("IPANEMA_RF_FP16", raising=False)
    assert isinstance(ballrf.autocast(), contextlib.nullcontext) and ballrf._fp16() == {}
    monkeypatch.setenv("IPANEMA_RF_FP16", "1")
    import torch
    if not torch.cuda.is_available():                       # CPU: still a no-op even when switched on
        assert isinstance(ballrf.autocast(), contextlib.nullcontext) and ballrf._fp16() == {}
