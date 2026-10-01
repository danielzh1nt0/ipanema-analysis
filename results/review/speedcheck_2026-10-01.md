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
