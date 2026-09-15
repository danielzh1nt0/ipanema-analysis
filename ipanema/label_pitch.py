"""Pitch keypoint labelling in Colab: click the 32 pitch landmarks on frames sampled across a match.
Saves reference/<clip>/pitch_kp.json = {frame_index: {kp_index: [x_px, y_px]}} after every click."""
import os, sys, json, base64, cv2

def label_pitch(video, clip_id, root, sports_dir="/content/sports", n_frames=80, start_s=0, end_s=None):
    from IPython.display import HTML, display
    from google.colab import output
    sys.path.append(sports_dir)
    from sports.configs.soccer import SoccerPitchConfiguration
    cfg = SoccerPitchConfiguration(); verts = [[v[0] / 100, v[1] / 100] for v in cfg.vertices]; edges = [[a - 1, b - 1] for a, b in cfg.edges]
    L, W = cfg.length / 100, cfg.width / 100
    def name(v):
        x, y = v; side = "left" if x < L / 3 else "right" if x > 2 * L / 3 else "centre"
        depth = "far" if y < W * 0.2 else "near" if y > W * 0.8 else "middle"
        if abs(x - L / 2) < 0.5 and abs(y - W / 2) < 0.5: return "centre spot"
        if abs(x - L / 2) < 0.5 and abs(y) < 0.5: return "halfway line, far touchline"
        if abs(x - L / 2) < 0.5 and abs(y - W) < 0.5: return "halfway line, near touchline"
        if abs(x - L / 2) < 0.5: return "centre circle, " + ("far edge" if y < W / 2 else "near edge")
        gl = x < 0.5 or x > L - 0.5; box = abs(min(x, L - x) - 16.5) < 0.5; six = abs(min(x, L - x) - 5.5) < 0.5; pen = abs(min(x, L - x) - 11) < 0.5
        if gl and (y < 0.5 or y > W - 0.5): return f"{depth}-{side} corner flag"
        if gl: return f"{side} goal line, {depth} " + ("six-yard box corner" if abs(abs(y - W / 2) - 9.16) < 0.5 else "penalty box corner")
        if six: return f"{side} six-yard box, {depth} front corner"
        if box: return f"{side} penalty box, {depth} front corner"
        if pen: return f"{side} penalty spot"
        if abs(min(x, L - x) - 20.15) < 0.6: return f"{side} penalty arc top"
        return f"{side} {depth} point"
    NAMES = [name(v) for v in verts]
    out_dir = os.path.join(root, "reference", clip_id); os.makedirs(out_dir, exist_ok=True); gt_path = os.path.join(out_dir, "pitch_kp.json")
    existing = json.load(open(gt_path)) if os.path.exists(gt_path) else {}
    cap = cv2.VideoCapture(video); fps = cap.get(5) or 25; n = int(cap.get(7)); s0 = int(start_s * fps); s1 = int(end_s * fps) if end_s else n - 1
    idxs = [int(round(s0 + k * (s1 - s0) / (n_frames - 1))) for k in range(n_frames)]; imgs = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if ok:
            h, w = f.shape[:2]; small = cv2.resize(f, (1280, int(1280 * h / w)))
            imgs[i] = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode()
    cap.release(); W_full, H_full = w, h
    output.register_callback("ipanema_save_kp", lambda js: json.dump(json.loads(js), open(gt_path, "w")))
    html = f"""
<div style="font:13px monospace;color:#ddd;display:flex;gap:12px;align-items:flex-start">
 <div><div id="p" style="color:#ffd54f;font-size:15px;margin-bottom:4px"></div>
  <canvas id="cv" width="1280" height="{int(1280*H_full/W_full)}" style="max-width:100%;border:1px solid #444;cursor:crosshair"></canvas>
  <div style="margin-top:4px"><b>1.</b> Click a landmark you can see on the small pitch (right) · <b>2.</b> click it in the frame · repeat for 5–8 visible landmarks · <b>N</b> next frame · <b>B</b> previous · <b>U</b> undo · <b>S</b> skip landmark</div></div>
 <div><canvas id="mini" width="420" height="{int(420*W/L)}" style="border:1px solid #444;background:#163"></canvas><div id="stat"></div></div>
</div>
<script>
const IDX={json.dumps(list(imgs))},IM={json.dumps(imgs)},NAMES={json.dumps(NAMES)},V={json.dumps(verts)},E={json.dumps(edges)},L={L},Wd={W},SX={W_full/1280},SY={H_full/int(1280*H_full/W_full)};
let gt={json.dumps(existing)};let fi=0,ki=0;const c=document.getElementById('cv'),x=c.getContext('2d'),m=document.getElementById('mini'),mx=m.getContext('2d');const im=new Image();
function px(v){{return [v[0]/L*m.width, v[1]/Wd*m.height];}}
function drawMini(){{mx.clearRect(0,0,m.width,m.height);mx.strokeStyle='#cfc';mx.lineWidth=1.5;for(const e of E){{const a=px(V[e[0]]),b=px(V[e[1]]);mx.beginPath();mx.moveTo(a[0],a[1]);mx.lineTo(b[0],b[1]);mx.stroke();}}
 const done=gt[IDX[fi]]||{{}};V.forEach((v,i)=>{{const p=px(v);mx.beginPath();mx.arc(p[0],p[1],i===ki?11:4,0,7);mx.fillStyle=i===ki?'#e0453a':(done[i]===null?'#888':(done[i]?'#5cf':'#fff'));mx.fill();if(i===ki){{mx.fillStyle='#000';mx.font='bold 11px monospace';mx.fillText(i,p[0]-4,p[1]+4);}}}});}}
function drawFrame(){{x.drawImage(im,0,0);const done=gt[IDX[fi]]||{{}};for(const k in done){{if(!done[k])continue;const q=done[k];x.beginPath();x.arc(q[0]/SX,q[1]/SY,5,0,7);x.strokeStyle='#5cf';x.lineWidth=2;x.stroke();x.fillStyle='#5cf';x.font='11px monospace';x.fillText(k,q[0]/SX+6,q[1]/SY-6);}}}}
function status(){{const done=gt[IDX[fi]]||{{}};const nl=Object.values(done).filter(v=>v).length;document.getElementById('p').innerHTML=`Frame ${{fi+1}}/${{IDX.length}} · <span style='color:#e0453a;font-size:18px'>click: ${{NAMES[ki]}}</span> (landmark ${{ki}}/${{V.length-1}}) · ${{nl}} clicked · S if not visible`;
 const total=Object.values(gt).reduce((a,d)=>a+Object.values(d).filter(v=>v).length,0);document.getElementById('stat').textContent=`saved: ${{Object.keys(gt).length}} frames, ${{total}} landmarks`;}}
function show(){{im.onload=()=>{{drawFrame();drawMini();status();}};im.src='data:image/jpeg;base64,'+IM[IDX[fi]];}}
function save(){{google.colab.kernel.invokeFunction('ipanema_save_kp',[JSON.stringify(gt)],{{}});}}
function nextK(){{ki=Math.min(V.length-1,ki+1);drawMini();status();}}
c.onclick=e=>{{const r=c.getBoundingClientRect();const X=(e.clientX-r.left)*c.width/r.width*SX,Y=(e.clientY-r.top)*c.height/r.height*SY;(gt[IDX[fi]]=gt[IDX[fi]]||{{}})[ki]=[Math.round(X),Math.round(Y)];save();drawFrame();nextK();}};
m.onclick=e=>{{const r=m.getBoundingClientRect();const X=(e.clientX-r.left)*m.width/r.width,Y=(e.clientY-r.top)*m.height/r.height;let best=0,bd=1e9;V.forEach((v,i)=>{{const p=px(v);const d=Math.hypot(p[0]-X,p[1]-Y);if(d<bd){{bd=d;best=i;}}}});ki=best;drawMini();status();}};
document.addEventListener('keydown',e=>{{const q=e.key.toLowerCase();
 if(q==='s'){{(gt[IDX[fi]]=gt[IDX[fi]]||{{}})[ki]=null;save();nextK();}}
 else if(q==='u'){{ki=Math.max(0,ki-1);if(gt[IDX[fi]])delete gt[IDX[fi]][ki];save();drawFrame();drawMini();status();}}
 else if(q==='n'||q==='d'){{fi=Math.min(IDX.length-1,fi+1);ki=0;show();}}
 else if(q==='b'){{fi=Math.max(0,fi-1);ki=0;show();}}}});
show();
</script>"""
    display(HTML(html))
    print(f"labels save to {gt_path}. Tip: a frame with 6+ landmarks is useful; you don't need all 32 — press S for anything not visible.")
