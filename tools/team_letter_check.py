"""N0 19 Oct: which kit is team "A" in an export? Scores the export's possession field against a by-eye
'who has the ball' key written in kit colours (results/review/who_answers.json: 'dark' / 'white'), once with
A = dark and once with A = light. The pipeline convention is A = darker kit; the export used to say the opposite.
Usage: python tools/team_letter_check.py [match_data.json] [who_answers.json]"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "results/volume/runs/matches/SFKBP1109_s1200/match_data.json")
KEY = os.path.join(ROOT, "results/review/who_answers.json")

def score(frames, fps, moments):
    """-> {"A=dark": (right, wrong, no call), "A=light": (...)} on the moments whose truth is 'dark' or 'white'"""
    out = {}
    for name, a_kit in (("A=dark", "dark"), ("A=light", "white")):
        kit = {"A": a_kit, "B": "white" if a_kit == "dark" else "dark"}; r = w = none = 0
        for q in moments:
            if q.get("truth") not in ("dark", "white"): continue
            k = int(round(q["t"] * fps))
            if not 0 <= k < len(frames): continue
            p = frames[k].get("possession")
            if p is None: none += 1
            elif kit[p] == q["truth"]: r += 1
            else: w += 1
        out[name] = (r, w, none)
    return out

def main(md_path=MD, key_path=KEY):
    md = json.load(open(md_path)); key = json.load(open(key_path))["moments"]
    s = score(md["frames"], md["fps"], key)
    lines = [f"export: {os.path.relpath(md_path, ROOT)} (its label: {md.get('teams')})",
             f"key: {os.path.relpath(key_path, ROOT)}, moments with a clear kit: {sum(s['A=dark'])}"]
    lines += [f"{k}: right {v[0]}, wrong {v[1]}, no team called {v[2]}" for k, v in s.items()]
    best = max(s, key=lambda k: s[k][0] - s[k][1]); lines.append(f"-> the data says {best}")
    print("\n".join(lines)); return s

if __name__ == "__main__":
    main(*sys.argv[1:3])
