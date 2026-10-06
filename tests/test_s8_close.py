"""S8 close-out (6 Oct): on the exact SFK-BP clip app inputs the counted (spell) state must stay de-flickered and the
spell-filtered passes must keep the fake ones out (3.2 flips/min, fake kept 2/8, real 14/20 on 6 Oct)."""
import os, json, pytest

PKL = "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl"

@pytest.mark.skipif(not os.path.exists(PKL), reason="clip inputs not in this checkout")
def test_counted_state_stays_deflickered_on_real_clip():
    from ipanema import possession as P
    from tools.s8lab import load, run
    c = load("SFKBP1109_s1200"); mins = len(c["per"]) / c["fps"] / 60
    st = P.spell_state(c["state"], c["fps"], P.SPELL_TAKE_S, P.SPELL_JOIN_S)
    raw, cnt = run(c, c["state"]), run(c, st, pass_state=True)
    assert raw["flips"] / mins > 15                      # the raw state really flickers (display only)
    assert cnt["flips"] / mins <= 4                      # real play ~2-3 per minute
    A = json.load(open("results/review/passcheck_answers.json")); fake = set(A["fake"])
    exp = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/stats.json"))["passes"]
    kept_fake = sum(any(abs(p["t"] - exp[i]["t"]) < 0.3 for p in cnt["passes"]) for i in fake if i < A["graded_first_n"])
    assert kept_fake <= 2
