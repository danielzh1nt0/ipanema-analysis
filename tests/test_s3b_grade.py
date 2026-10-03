"""S3b grader: finder-alone top-guess counting and picker hit counting on tiny hand-made inputs."""
import importlib.util, os
spec = importlib.util.spec_from_file_location("s3b_grade", os.path.join(os.path.dirname(__file__), "..", "tools", "s3b_grade.py"))
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)

def test_top_hits_uses_highest_confidence_only():
    cands = {1: [(100, 100, 0.2), (500, 500, 0.9)], 2: [(10, 10, 0.5)], 3: []}
    key = [(1, 100, 100), (2, 12, 14), (3, 0, 0), (4, 0, 0)]
    assert G.top_hits(cands, key) == 1          # frame 1: top guess is the wrong one; 2 right; 3/4 no guesses

def test_picker_hits_30px_radius():
    ball = {1: (100, 100), 2: (200, 200)}
    sgt = {1: [110, 110], 2: [240, 200]}        # 14 px -> hit, 40 px -> miss
    b4 = [(1, 100, 129), (5, 0, 0)]
    assert G.picker_hits(ball, sgt, b4) == (1, 1)
