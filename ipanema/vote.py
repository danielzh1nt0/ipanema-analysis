"""N0-20d (20 Oct): majority votes that do not depend on chance.

`max(set(v), key=v.count)` was used for every team vote (per-track team in tracking, a track's team in the stats, keeper
kit, T1 smoothing). On a tie it returns whichever label the set lists first, and Python shuffles that order per process
(string hashes are salted, PYTHONHASHSEED). So on a 2 A / 2 B vote the same video gave A in one run and B in the next.
Now a tie goes to `prefer` (e.g. the track's team so far) when it is one of the tied labels, else to the most recent of
the tied labels. IPANEMA_VOTE_TIE=hash = old behaviour (set order)."""
import os


def majority(seq, prefer=None):
    """most common item of seq (None if empty); ties -> prefer if tied, else the tied item seen last"""
    seq = list(seq)
    if not seq: return None
    if os.environ.get("IPANEMA_VOTE_TIE", "") == "hash": return max(set(seq), key=seq.count)
    cnt = {}
    for x in seq: cnt[x] = cnt.get(x, 0) + 1
    top = max(cnt.values()); tied = [x for x in cnt if cnt[x] == top]
    if len(tied) == 1: return tied[0]
    if prefer is not None and prefer in tied: return prefer
    for x in reversed(seq):
        if x in tied: return x


def team_vote(v, k_share=0.6, prev=None):
    """per-track team from its recent colour readings v ('A', 'B', 'K' = neither team, 'R' = referee/bib; 'O' already
    removed): a track that reads 'neither' most of the time (k_share) - or never reads a team - is 'K', else the majority of
    its A/B readings. prev = the track's team at its last frame, kept on an A/B tie (no flicker on a 2-2 vote)."""
    if not v: return None
    ab = [x for x in v if x in ("A", "B")]
    if (v.count("K") >= k_share * len(v) or not ab) and "K" in v: return "K"
    return majority(ab, prefer=prev) if ab else majority(v, prefer=prev)
