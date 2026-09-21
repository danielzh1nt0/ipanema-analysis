"""Ball labelling tool for a reference clip (Colab). Saves reference/<match_id>/ball_gt.json on every click."""
import cv2, base64, json, os
import numpy as np
def sample_frames(video, n_label=20, min_minutes=None):
    """Read n_label evenly spread frames. Checks the input first and fails loudly with a clear message."""
    if not os.path.exists(video): raise FileNotFoundError(f"video not found: {video}")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise RuntimeError(f"could not open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    minutes = n / fps / 60
    print(f"video: {os.path.basename(video)} · {os.path.getsize(video) / 1e9:.2f} GB · {minutes:.1f} min · {n} frames")
    if min_minutes and minutes < min_minutes:
        cap.release(); raise RuntimeError(f"this video is {minutes:.1f} min long; expected a full match (at least {min_minutes} min). Wrong file?")
    if n < n_label: cap.release(); raise RuntimeError(f"only {n} frames; need at least {n_label}")
    last = max(0, n - 10)   # some files can't seek to their final frames
    idxs = sorted({int(round(k * last / max(1, n_label - 1))) for k in range(n_label)})
    imgs = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok and f is not None: imgs[i] = base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    cap.release()
    if not imgs: raise RuntimeError(f"could not read any frames from {video}")
    first = cv2.imdecode(np.frombuffer(base64.b64decode(next(iter(imgs.values()))), np.uint8), cv2.IMREAD_COLOR)
    print(f"loaded {len(imgs)} of {len(idxs)} frames")
    return imgs, first.shape[0], first.shape[1]

def label_ball(video, match_id, root, n_label=20, min_minutes=None):
    from IPython.display import HTML, display
    from google.colab import output
    out_dir = os.path.join(root, "reference", match_id); os.makedirs(out_dir, exist_ok=True); gt_path = os.path.join(out_dir, "ball_gt.json")
    imgs, h, w = sample_frames(video, n_label, min_minutes)
    output.register_callback("ipanema_save_gt", lambda js: json.dump(json.loads(js), open(gt_path, "w")))
    display(HTML(f"""<div style="font:14px monospace;color:#ddd"><div id="p" style="color:#ffd54f;font-size:16px"></div>
<canvas id="cv" width="{w}" height="{h}" style="max-width:100%;border:1px solid #444;cursor:crosshair"></canvas><div>Click the ball · S = not visible · U = undo</div></div>
<script>const IDX={json.dumps(list(imgs))},IM={json.dumps(imgs)},c=document.getElementById('cv'),x=c.getContext('2d');let k=0,gt={{}};const im=new Image();
function show(){{if(k>=IDX.length){{document.getElementById('p').textContent='Done — saved to {gt_path}';return;}}document.getElementById('p').textContent=`Frame ${{k+1}}/${{IDX.length}}: click the ball`;
im.onload=()=>{{x.drawImage(im,0,0);}};im.src='data:image/jpeg;base64,'+IM[IDX[k]];}}
function save(){{google.colab.kernel.invokeFunction('ipanema_save_gt',[JSON.stringify(gt)],{{}});}}
c.onclick=e=>{{if(k>=IDX.length)return;const r=c.getBoundingClientRect();gt[IDX[k]]=[Math.round((e.clientX-r.left)*c.width/r.width),Math.round((e.clientY-r.top)*c.height/r.height)];save();k++;show();}};
document.addEventListener('keydown',e=>{{const q=e.key.toLowerCase();if(q==='s'&&k<IDX.length){{gt[IDX[k]]=null;save();k++;show();}}if(q==='u'&&k>0){{k--;delete gt[IDX[k]];save();show();}}}});show();</script>"""))
