# V3 (5 Oct): players dropped as 'neither team' on Vallentuna (red vs black, hard sun + shade) - free, offline

By eye on the test frames (results/qa/v3/before_neither.jpg): 28 of 44 'neither' were players (sunlit reds, blacks), the rest
referee / keepers / spectators. Their colour features: reds a* 21-52 with b* from -46 to +11 (sun -> orange, shade ->
purple), blacks a* 2-12. b* and lightness are not stable in this light; a* is.

Fix (kits.classify, 'hue mode'): only in local light (hard sun + shade) AND when the two kits differ in a* by >= 15:
decide by a* alone (midpoint between the kits); 'neither' only for white (L > 90), yellow (referee/keepers) or greenish.
Black-vs-white matches (SFK-BP, AIK) and frame-light grounds are untouched (Reymersholm, Spanga, AIK keys unchanged).

Vallentuna test frames, per frame on the pitch: A 4.3 -> 4.8, B 3.8 -> 4.7, neither 2.4 -> 1.1. By eye (after_*.jpg):
'neither' now ~3 players of 19; red team ~6 sunlit blacks in 84 (unchanged); black team ~10 spectators beside the pitch
(the calibration removes people off the pitch later). A tighter white rule (L > 65) dropped sunlit reds again - rejected.
To reach the app it needs the players re-tracked on the server (GPU, ~EUR 5) + a join (cents).
