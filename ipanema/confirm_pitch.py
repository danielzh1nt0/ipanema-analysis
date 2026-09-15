"""Confirm-only pitch labelling: the keypoint model proposes the pitch on each frame, you press Y (lines match) or N (they don't).
Y frames are saved as keypoint labels for fine-tuning. Saves reference/<clip>/pitch_confirm.json after every key."""
import os, sys, json, base64, cv2, numpy as np

def confirm_pitch(video, clip_id, root, weights_pitch, sports_dir="/content/sports", n_frames=120, kp_conf=0.5):
    from IPython.display import HTML, display
    from google.colab import output
    import supervision as sv
    from ultralytics import YOLO
    from .calibration import _pitch_config, draw_model
    VERTS, L, W = _pitch_config(sports_dir); model = YOLO(weights_pitch)
    out_dir = os.path.join(root, "reference", clip_id); os.makedirs(out_dir, exist_ok=True); path = os.path.join(out_dir, "pitch_confirm.json")
    saved = json.load(open(path)) if os.path.exists(path) else {}
    cap = cv2.VideoCapture(video); n = int(cap.get(7)); idxs = [int(round(k * (n - 1) / (n_frames - 1))) for k in range(n_frames)]; imgs = {}; props = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok: continue
        kp = sv.KeyPoints.from_ultralytics(model(f, verbose=False)[0]); pts = None
        if len(kp.xy):
            m = kp.confidence[0] > kp_conf
            if m.sum() >= 4:
                H, inl = cv2.findHomography(VERTS[m], kp.xy[0][m], cv2.RANSAC, 8.0)
                if H is not None: draw_model(f, H, L, W, colour=(40, 40, 230)); pts = {int(j): [float(x), float(y)] for j, (x, y) in enumerate(kp.xy[0]) if m[j]}
        if pts is None: cv2.putText(f, "no pitch found - press N", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
        props[i] = pts; h, w = f.shape[:2]; small = cv2.resize(f, (1280, int(1280 * h / w)))
        imgs[i] = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()
    cap.release()
    def save(js):
        d = json.loads(js); json.dump({"decisions": d, "keypoints": {str(i): props[int(i)] for i, v in d.items() if v == "Y" and props.get(int(i))}}, open(path, "w"))
    output.register_callback("ipanema_save_confirm", save)
    display(HTML(f"""<div style="font:14px monospace;color:#ddd"><div id="p" style="color:#ffd54f;font-size:18px;margin-bottom:6px"></div>
<img id="im" style="max-width:100%;border:1px solid #444"><div style="margin-top:6px;font-size:16px"><b>Y</b> = the red lines sit on the real pitch lines &nbsp;·&nbsp; <b>N</b> = they don't (or no pitch found) &nbsp;·&nbsp; <b>B</b> = back</div></div>
<script>const IDX={json.dumps(list(imgs))},IM={json.dumps(imgs)};let d={json.dumps(saved.get("decisions", {}))};let k=0;
function show(){{const done=Object.values(d).filter(v=>v==='Y').length,tot=Object.keys(d).length;document.getElementById('p').textContent=k>=IDX.length?`Done. ${{done}}/${{tot}} confirmed.`:`Frame ${{k+1}}/${{IDX.length}} — do the red lines match?   (confirmed so far: ${{done}}/${{tot}})`;if(k<IDX.length)document.getElementById('im').src='data:image/jpeg;base64,'+IM[IDX[k]];}}
function save(){{google.colab.kernel.invokeFunction('ipanema_save_confirm',[JSON.stringify(d)],{{}});}}
document.addEventListener('keydown',e=>{{const q=e.key.toLowerCase();if(q==='y'||q==='n'){{if(k<IDX.length){{d[IDX[k]]=q.toUpperCase();save();k++;show();}}}}else if(q==='b'&&k>0){{k--;show();}}}});show();</script>"""))
    print(f"decisions save to {path}")
