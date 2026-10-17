"""T1 relabel (17 Oct, main session): tracks that switch between a black and a red player get time-segmented teams.
The rule was inline in kaggle/full_match.py STAGE=relabel; the worker moved it unchanged to FM.team_segments."""
import random, re
from ipanema import fullmatch as FM

FPS = 25.0


def _old_inline(series, fps, SEG=4):
    """verbatim copy of the 801b8a2 inline block (reference for the move)"""
    sr = sorted(series); labs_ = [x[1] for x in sr]
    if len(sr) >= 2 * SEG:
        sm = [max(set(labs_[max(0, i - 2):i + 3]), key=labs_[max(0, i - 2):i + 3].count) for i in range(len(labs_))]
        segs = []; st = 0
        for i in range(1, len(sm) + 1):
            if i == len(sm) or sm[i] != sm[st]:
                segs.append((st, i - 1, sm[st])); st = i
        segs = [(a_, b_, t_) for a_, b_, t_ in segs if b_ - a_ + 1 >= SEG]
        if len({t_ for _, _, t_ in segs}) > 1:
            parts = []
            for j, (a_, b_, t_) in enumerate(segs):
                t0 = sr[a_][0] / fps if j else 0.0; t1 = sr[b_][0] / fps if j < len(segs) - 1 else 1e9
                parts.append([round(t0, 2), round(t1, 2), t_])
            return parts
    return None


def test_same_as_the_inline_rule_on_random_tracks():
    rng = random.Random(17)
    for _ in range(2000):
        n = rng.randint(0, 40); p_switch = rng.choice([0.02, 0.1, 0.3]); t = rng.choice("AB"); f = rng.randint(0, 5000); s = []
        for _ in range(n):
            if rng.random() < p_switch: t = "B" if t == "A" else "A"
            f += 6; s.append((f, t if rng.random() > 0.15 else ("B" if t == "A" else "A")))
        rng.shuffle(s)                                                         # reads arrive in piece order, not time order
        seg = rng.choice([3, 4, 6])
        assert FM.team_segments(s, FPS, seg) == _old_inline(s, FPS, seg)


def test_a_clean_switch_is_split_at_the_switch():
    s = [(100 + 6 * i, "A") for i in range(10)] + [(160 + 6 * i, "B") for i in range(10)]
    parts = FM.team_segments(s, FPS)
    assert parts == [[0.0, round(154 / FPS, 2), "A"], [round(160 / FPS, 2), 1e9, "B"]]


def test_noise_alone_does_not_split():
    s = [(6 * i, "B" if i % 7 == 3 else "A") for i in range(30)]               # single wrong reads (shade) are smoothed away
    assert FM.team_segments(s, FPS) is None
    assert FM.team_segments([(6 * i, "A") for i in range(30)], FPS) is None
    assert FM.team_segments([(0, "A"), (6, "B")], FPS) is None                 # too few reads to judge


def test_split_reaches_the_rows_through_apply_team_override():
    s = [(6 * i, "A") for i in range(10)] + [(60 + 6 * i, "B") for i in range(10)]
    parts = FM.team_segments(s, FPS)
    per = {g: [[7, "K", 0.0, 0.0, [0, 0, 1, 1]]] for g in range(0, 120)}
    FM.apply_team_override(per, {"7": parts}, FPS, from_k=True)
    assert {per[g][0][1] for g in range(0, 55)} == {"A"}
    assert {per[g][0][1] for g in range(60, 120)} == {"B"}


def test_relabel_stage_renders_and_compiles():
    """dry run of the kernel template as tools/kaggle_run.py fills it (no Kaggle, no video)"""
    code = open("kaggle/full_match.py").read().replace("{{R2}}", "").replace("{{SOURCES_DIRS}}", "")
    for k, v in {"MATCH": "m", "STAGE": "relabel", "SEG": "4"}.items(): code = code.replace("{{" + k + "}}", v)
    code = re.sub(r"\{\{[A-Z_]+\}\}", "", code)
    compile(code, "full_match.py", "exec")
    assert "FM.team_segments(series[tid]" in code
