"""Build the Ipanema Work Board page (results/workboard.html) from BACKLOG.md + results/board/{titles,progress}.json.
Run after any change to the work queue; the scheduled jobs publish the result to the board artifact.
    python tools/build_workboard.py"""
import os, re, json, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); B = f"{ROOT}/results/board"
AREA = [("PF", "Ball"), ("PC", "Stats"), ("B", "Ball"), ("T", "Ball"), ("Z", "Ball"), ("H", "Ball"), ("P", "Players"),
        ("S", "Stats"), ("E", "Stats"), ("K", "Setup"), ("A", "Setup"), ("N", "Setup")]
ID_RE = re.compile(r"^(?:\((?:PARKED|BLOCKED)[^)]*\)\s*)?([A-Z]{1,2}\d+[a-z]?)\b")

def area(i):
    if i == "S3": return "Ball"
    return next((a for p, a in AREA if i and i.startswith(p)), None)

def split(text, titles):
    m = ID_RE.match(text); i = m.group(1) if m else None
    body = text[m.end():].lstrip(" .:") if m else text
    body = re.sub(r"^\(\d+ Sep[^)]*\)\s*", lambda x: x.group(0), body)
    title = titles.get(i) or re.split(r"(?<=[.!?])\s|:\s", body, 1)[0][:110]
    return i, title, text

def build():
    titles = json.load(open(f"{B}/titles.json")); prog = json.load(open(f"{B}/progress.json"))
    md = open(f"{ROOT}/BACKLOG.md").read(); sec = {}
    for part in re.split(r"^## ", md, flags=re.M)[1:]:
        name, _, body = part.partition("\n"); sec[name.strip()] = body
    tasks = []; seen_done = set()
    q = [l[2:].strip() for l in sec.get("Queue (priority order)", "").splitlines() if l.startswith("- [")]
    open_n = 0
    for l in q:
        done = l.startswith("[x]"); text = l[3:].strip(); i, title, detail = split(text, titles)
        if done: col = "done"; seen_done.add(i)
        elif "(BLOCKED" in text or "(PARKED" in text: col = "blocked"
        else: col = "next" if open_n < 3 else "queue"; open_n += 1
        tasks.append({"id": i, "area": area(i), "title": title, "detail": detail, "col": col})
    for l in sec.get("Waiting for Daniel", "").splitlines():
        if l.startswith("- "):
            t = l[2:].strip(); first = re.split(r"(?<=[.])\s|\s->\s", t, 1)[0]
            tasks.append({"id": None, "area": None, "title": first[:130], "detail": t if t != first else "", "col": "you"})
    for l in sec.get("Done", "").splitlines():
        if l.startswith("- [x]"):
            t = re.sub(r"^\d+ \w{3}:\s*", "", l[5:].strip()); m = re.search(r"\b([A-Z]{1,2}\d+[a-z]?)\b", t); i = m.group(1) if m else None
            if i in seen_done: continue
            tasks.append({"id": i, "area": area(i) or "Players", "title": re.split(r"(?<=[.;])\s|:\s", t, 1)[0][:140], "detail": t, "col": "done"})
    data = {"updated": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).strftime("%d %b %Y, %H:%M"),
            "release_board": "https://claude.ai/artifact/P3HvybgjWGxXeUuFZTgvW6", "progress": prog, "tasks": tasks}
    html = open(f"{B}/template.html").read().replace("/*DATA*/null/*END*/", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    open(f"{ROOT}/results/workboard.html", "w").write(html)
    return {c: sum(t["col"] == c for t in tasks) for c in ("you", "next", "queue", "blocked", "done")}

if __name__ == "__main__": print(build())
