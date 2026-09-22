"""Turn Veo highlight file names into a match's shot/goal list.

Usage: python scripts_veo_highlights.py <match_id> [veo_start_s] < names.txt
Veo names them like "23 013014_-_Shot_on_goal.mp4" (HHMMSS of the recording). Markdown links, extra text and the
".mp4" are ignored, duplicates dropped. veo_start_s shifts the times when the match is a clip that starts later in the
recording (times before the clip are dropped)."""
import re, sys, os

PATTERN = re.compile(r"(?:^|[\s/\[])(\d{1,3})\s+(\d{6})_-_([A-Za-z_]+)")

def parse(text, offset_s=0.0, duration_s=None):
    """-> [(seconds, "shot"|"goal"), ...] sorted, de-duplicated"""
    out = {}
    for m in PATTERN.finditer(text):
        hms = m.group(2); kind = m.group(3).lower()
        t = int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:]) - offset_s
        if t < 0 or (duration_s is not None and t > duration_s): continue
        k = "goal" if kind.startswith("goal") else ("shot" if "shot" in kind else None)
        if k: out[(round(t, 2), k)] = True
    return sorted(out)

def main():
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    mid = sys.argv[1]; off = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    rows = parse(sys.stdin.read(), off)
    os.makedirs("reference", exist_ok=True); path = f"reference/veo_highlights_{mid}.txt"
    open(path, "w").write(f"# Veo highlight clips for {mid}, seconds into the video (from the clip file names" + (f", shifted by {off:.0f} s" if off else "") + ").\n"
                          + "".join(f"{int(t) if float(t).is_integer() else t} {k}\n" for t, k in rows))
    goals = sum(1 for _, k in rows if k == "goal")
    print(f"{path}: {len(rows)} entries ({goals} goals, {len(rows) - goals} shots)")

if __name__ == "__main__": main()
