"""9 Oct (worker): the queue holds only work the worker can do for free; anything needing Modal / money / Daniel's
decision sits under 'Waiting for Daniel' (BACKLOG rules). Guards the D3/D4/D5/V1 clean-up of 9 Oct."""
import os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sections():
    md = open(f"{ROOT}/BACKLOG.md").read(); sec = {}
    for part in re.split(r"^## ", md, flags=re.M)[1:]:
        name, _, body = part.partition("\n"); sec[name.strip()] = body
    return sec

def open_items(sec):
    return [l for l in sec["Queue (priority order)"].splitlines() if l.startswith("- [ ]")]

def test_no_open_queue_item_needs_modal_or_money():
    bad = [l[:80] for l in open_items(sections()) if re.search(r"\bModal\b|EUR\s?\d|\$\d", l)]
    assert bad == [], f"move these to 'Waiting for Daniel': {bad}"

def test_moved_items_are_waiting_not_queued():
    sec = sections(); q = "\n".join(open_items(sec)); w = sec["Waiting for Daniel"]
    for i in ("D4", "D5"):
        assert not re.search(rf"^- \[ \] {i}\b", q, flags=re.M) and re.search(rf"^- {i}\b", w, flags=re.M)
    assert "- [ ] V1 " not in q and "- [ ] D3" not in q

def test_d5_keep_list_still_guards_vallentuna():
    from tools import app_cleanup as AC
    assert "p15u-vs-vallentuna-2026-10-03-6cce" in AC.DEMO_KEEP
