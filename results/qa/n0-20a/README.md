# N0-20a: the AIK clip's Veo shots and goal (20 Oct)

**Problem (part 1, 00:20):** a clip cut from a match (`<match>_s<start>`) looked for a Veo file under the clip's own id,
found none and showed no shots or goals. The app's AIK clip (`p15u-vs-aik-2026-09-21-bd09_s2520`) showed 0 shots / 0 goals,
though Veo has 3 shot clips and AIK's first goal inside it. Fix: `run.veo_source` reads the match's file and shifts it to
the clip's clock (tests/test_veo_clip.py). Replay on the clip's exact inputs: shots A 1 / B 2, goal B 1 (replay_aik_clip.json).

**Checked by eye (part 2, 04:20), free runner, $0** (sheets in results/free/n0-20a and results/free/n0-20a/dense):

| Veo clip (clip time) | team we give | what the pictures show | right? |
|---|---|---|---|
| shot 12 s | A (SFK, dark) | play runs to the right goal (orange keeper) 24-40 s; A attacks right | yes |
| shot 95 s | B (AIK, white) | play at the left goal 95-115 s, whites going at it; B attacks left | yes |
| goal 242 s | B (AIK) | **257 s: AIK #24 shoots past the diving keeper, 258 keeper on the ground, 259 ball in the net, 259-262 referee points to the centre; 298-300 s SFK lines up for the kick-off** (goal_zoom_256-263.png) | yes |

So the shifted times are right (the action comes 12-25 s after Veo's clip start, as on the full matches), all three teams
are right, and the goal is really inside the clip (at ~257 s, match ~2777 s). Shot spots stay empty (end only), as on the
full matches.

Small fix found on the way: the sheet script counted a black padding tile as a frame (dense_290 shows 11 frames, not 12);
fixed, tests/test_n0_20a_frames.py (2, dry run on a stand-in clip).

**Reaches the app** at the next AIK clip run (Modal) - the same re-join as the direction fix (N0-19b), already under
Waiting for Daniel. Nothing ran on Modal.
