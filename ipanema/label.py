"""Ball labelling tool for a reference clip (Colab). Saves reference/<match_id>/ball_gt.json on every click."""
import cv2, base64, json, os
def label_ball(video, match_id, root, n_label=20):
    from IPython.display import HTML, display
    from google.colab import output
    out_dir = os.path.join(root, "reference", match_id); os.makedirs(out_dir, exist_ok=True); gt_path = os.path.join(out_dir, "ball_gt.json")
    cap = cv2.VideoCapture(video); n = int(cap.get(7)); idxs = [int(round(k * (n - 1) / (n_label - 1))) for k in range(n_label)]; imgs = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if ok: imgs[i] = base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    cap.release(); h, w = f.shape[:2]
    output.register_callback("ipanema_save_gt", lambda js: json.dump(json.loads(js), open(gt_path, "w")))
    display(HTML(f"""<div style="font:14px monospace;color:#ddd"><div id="p" style="color:#ffd54f;font-size:16px"></div>
<canvas id="cv" width="{w}" height="{h}" style="max-width:100%;border:1px solid #444;cursor:crosshair"></canvas><div>Click the ball · S = not visible · U = undo</div></div>
<script>const IDX={json.dumps(list(imgs))},IM={json.dumps(imgs)},c=document.getElementById('cv'),x=c.getContext('2d');let k=0,gt={{}};const im=new Image();
function show(){{if(k>=IDX.length){{document.getElementById('p').textContent='Done — saved to {gt_path}';return;}}document.getElementById('p').textContent=`Frame ${{k+1}}/${{IDX.length}}: click the ball`;
im.onload=()=>{{x.drawImage(im,0,0);}};im.src='data:image/jpeg;base64,'+IM[IDX[k]];}}
function save(){{google.colab.kernel.invokeFunction('ipanema_save_gt',[JSON.stringify(gt)],{{}});}}
c.onclick=e=>{{if(k>=IDX.length)return;const r=c.getBoundingClientRect();gt[IDX[k]]=[Math.round((e.clientX-r.left)*c.width/r.width),Math.round((e.clientY-r.top)*c.height/r.height)];save();k++;show();}};
document.addEventListener('keydown',e=>{{const q=e.key.toLowerCase();if(q==='s'&&k<IDX.length){{gt[IDX[k]]=null;save();k++;show();}}if(q==='u'&&k>0){{k--;delete gt[IDX[k]];save();show();}}}});show();</script>"""))
