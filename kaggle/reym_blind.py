# Kaggle (free, 29 Sep, E6): BLIND pictures for re-grading the Reymersholm who-has-the-ball key: the 60 moments of
# results/review/who_moments_reym.json as full frames, no ball or player markers (the old key was graded with the old
# ball's circle on screen). Output: /kaggle/working/m<NN>.jpg (moment n, frame 0) + m<NN>b.jpg (+0.4 s).
import json, cv2, urllib.request
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"
req = urllib.request.Request("https://raw.githubusercontent.com/danielzh1nt0/ipanema-analysis/main/results/review/who_moments_reym.json", headers={"User-Agent": "Mozilla/5.0"})
D = json.load(urllib.request.urlopen(req)); OFF = int(D["offset_frames"])
cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); n = 0
for j, m in enumerate(D["moments"]):
    for d, tag in ((0, ""), (12, "b")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, OFF + m["frame"] + d); ok, f = cap.read()
        if ok: cv2.imwrite(f"{W}/m{j:02d}{tag}.jpg", cv2.resize(f, (1920, 1080)), [cv2.IMWRITE_JPEG_QUALITY, 82]); n += 1
json.dump({"ok": True, "pictures": n}, open(f"{W}/result.json", "w"))
