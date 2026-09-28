"""28 Sep: detections the model labels 'referee' must reach the tracker (they are mostly black-shirted players)"""
import os, sys, types, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ipanema", "tracking.py")).read()
i = src.index("def track("); body = src[i:src.index("\ndef ", i + 10)]
assert 'IPANEMA_DROP_MODEL_REFEREE", "0") == "1"' in body, "model-referee drop must be off by default"
assert body.index("IPANEMA_DROP_MODEL_REFEREE") < body.index("update_with_detections"), "check must sit before the tracker"
t = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ipanema", "teams.py")).read()
assert '"referee" in cls or' not in t, "team model must learn from model-referee boxes too"
print("OK")
