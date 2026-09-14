# Ipanema — match analysis pipeline

Video in → `match_data.json` + `stats.json` (contract 1.1) → Supabase/R2 → the Ipanema app.

## Colab (the only thing you run)

**Cell 0** (once per session):
```python
from google.colab import drive; drive.mount('/content/drive')
import subprocess; subprocess.run("rm -rf /content/ipanema-analysis && git clone -q https://github.com/danielzh1nt0/ipanema-analysis.git /content/ipanema-analysis", shell=True)
exec(open('/content/ipanema-analysis/colab_setup.py').read())
```
**Cell 1** (analyse every clip in `videos/`):
```python
import os
from ipanema.run import run
for clip in sorted(f for f in os.listdir(S.path("videos")) if f.lower().endswith((".mp4", ".mov", ".mkv"))):
    run(S.path("videos", clip), settings=S)
```
Updates: none. Cell 0 clones the latest code every session.

Colab Secrets (only for the Supabase/R2 upload step): `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_BUCKET`, `R2_PUBLIC_URL`.

## Other cells
- `ipanema.label.label_ball(video, clip_id, ROOT)` — 20 ball clicks for a reference clip.
- `ipanema.check.check_all(ROOT)` — scoreboard against reference labels.
- `ipanema.train_ball.train(ROOT, videos_dir, base_weights)` — fine-tune the ball detector on the clicks.

## Layout on Drive (`match_analysis/match_analysis/`)
`videos/` clips · `runs/matches/<id>/` outputs · `cache/<id>/` stage caches · `reference/<id>/ball_gt.json` + `reference/events_gt_<id>.json` labels · `models/` fine-tuned weights.

## tools/
`event_labeller.html` (browser, no Colab) · `events_reference_player_v2.html` (check event timing) · `supabase_schema_v1.sql`.
