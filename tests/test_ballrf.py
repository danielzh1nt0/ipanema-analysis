from ipanema import ballrf as R


def test_tiles_cover_full_hd_frame():
    t = R.tiles(1920, 1080)
    assert len(t) == 8 and max(x for x, _ in t) + R.CROP == 1920 and max(y for _, y in t) + R.CROP == 1080
