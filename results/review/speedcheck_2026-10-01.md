# Player speed layer: by-eye check (1 Oct, SFK-BP clip, free)

ipanema/motion.py turns tracked positions into km/h + metres run. 28 readings shown as pictures (t and t+0.5 s, cross on the
tracked player, speed hidden), graded by eye before opening the speeds (results/kaggle/speedcheck/, answers in speedcheck_answers.json).

| Reading group | What the pictures show |
|---|---|
| >25 km/h, far side (10) | 5 running, 3 jogging, 2 walking. No youth player runs 32-36 km/h: all 10 too high |
| >25 km/h, near side (10) | 4 running, 1 standing, **5 the cross jumps to another player (ID swap)** |
| 8-14 km/h (8) | walking players read 11-12 km/h (real walk ~5): about 2x too high; 1 swap |

Verdict: **not ready for the app.** Two faults, both upstream of the speed maths:
1. ID swaps inside a track (cross jumps to a nearby player) -> fake bursts and fake metres.
2. Slow players read ~2x too fast -> calibration wobble / box jitter adds motion; distances would be inflated the same way.

Export of kmh/dist_m is wired but OFF (IPANEMA_MOTION=1 turns it on). Next: (a) measure the calibration wobble on a
standing player (should read 0 km/h), (b) swap filter (a track whose box colour/size changes abruptly is split), then re-check.

## M1 round 1 (same evening, tools/motionlab.py, free)
- Calibration wobble is NOT the cause: smoothing the calibration over time made it worse (the follow-cam pans fast).
- Jitter adds metres: 0.5 s position smoothing gave 117-134 m per player-minute; **1.0 s gives 96 (AIK) - 115 (SFK-BP)**,
  in the range usually reported for youth matches. Speeds now p50 ~5 km/h, p95 15-17, p99 20-22; above 25 km/h hidden.
- Same 28 moments: 14/21 in a sensible range or hidden (was 10/21); no more 30+ km/h numbers. Still wrong: far-side
  jogging/walking reads 21-24 km/h (SFK-BP far side reads faster than near side: p50 9.7 vs 5.5 km/h), and **ID swaps are
  not caught (0/7)** - they move smoothly, so a jump filter cannot see them.
Verdict: good enough for team-level running numbers, not yet for a number under each player. Export stays OFF.
Next: swap detection needs the player's look (kit colour/box size per frame) -> part of M2 (one track per player).
