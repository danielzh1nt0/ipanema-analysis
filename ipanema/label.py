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


# ---------------------------------------------------------------- pitch points (for calibrating follow-cam frames)
# pitch coordinates in metres: x along the pitch (0 = left goal line, 106 = right), y across (0 = far touchline,
# 64 = near touchline, the camera side). Standard markings are fixed by the Laws of the Game.
def pitch_keypoints(L=106.0, W=64.0):
    c = W / 2; pa, ga, pk, gw = 20.16, 9.16, 11.0, 3.66          # half-widths of penalty area / goal area, penalty spot, half goal
    P = {"far-left corner": (0, 0), "far-right corner": (L, 0), "near-left corner": (0, W), "near-right corner": (L, W),
         "halfway x far touchline": (L / 2, 0), "halfway x near touchline": (L / 2, W), "centre spot": (L / 2, c),
         "centre circle x halfway (far)": (L / 2, c - 9.15), "centre circle x halfway (near)": (L / 2, c + 9.15)}
    for side, x0, s in (("left", 0.0, 1), ("right", L, -1)):
        P[f"{side} penalty area: far corner on goal line"] = (x0, c - pa); P[f"{side} penalty area: near corner on goal line"] = (x0, c + pa)
        P[f"{side} penalty area: far front corner"] = (x0 + s * 16.5, c - pa); P[f"{side} penalty area: near front corner"] = (x0 + s * 16.5, c + pa)
        P[f"{side} goal area: far corner on goal line"] = (x0, c - ga); P[f"{side} goal area: near corner on goal line"] = (x0, c + ga)
        P[f"{side} goal area: far front corner"] = (x0 + s * 5.5, c - ga); P[f"{side} goal area: near front corner"] = (x0 + s * 5.5, c + ga)
        P[f"{side} penalty spot"] = (x0 + s * pk, c)
        P[f"{side} goal: far post (base)"] = (x0, c - gw); P[f"{side} goal: near post (base)"] = (x0, c + gw)
    return P

def label_points(frames_zip, out_json, L=106.0, W=64.0):
    """Colab tool: click a point in the frame, then the same point on the pitch diagram. 4-6 per frame, only clearly
    visible ones. Keys: U undo, N next frame, B back, S nothing identifiable. Saves to out_json after every change and
    resumes where it stopped."""
    import zipfile
    from IPython.display import HTML, display
    from google.colab import output
    z = zipfile.ZipFile(frames_zip); names = sorted(n for n in z.namelist() if n.startswith("fc_") and n.endswith(".jpg"))
    if not names: raise RuntimeError(f"no follow-cam frames (fc_*.jpg) in {frames_zip}")
    imgs = {n: base64.b64encode(z.read(n)).decode() for n in names}
    first = cv2.imdecode(np.frombuffer(z.read(names[0]), np.uint8), cv2.IMREAD_COLOR); h, w = first.shape[:2]
    done = json.load(open(out_json)) if os.path.exists(out_json) else {}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    print(f"{len(names)} frames ({w}x{h}); {len(done)} already done; saving to {out_json}")
    output.register_callback("ipanema_save_points", lambda js: json.dump(json.loads(js), open(out_json, "w"), indent=0))
    kp = pitch_keypoints(L, W); S = 7.0; pw, ph = int(L * S + 40), int(W * S + 40)
    start = next((i for i, n in enumerate(names) if n not in done), len(names))
    display(HTML(f"""<div style="font:13px sans-serif;color:#ddd">
<div id="p" style="color:#ffd54f;font-size:15px;margin:4px 0"></div>
<div style="display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap">
 <div style="position:relative"><canvas id="cv" width="{w}" height="{h}" style="width:min(62vw,{w}px);border:1px solid #444;cursor:crosshair"></canvas>
  <canvas id="zm" width="180" height="180" style="position:absolute;right:6px;top:6px;border:2px solid #ffd54f;background:#000"></canvas></div>
 <div><canvas id="pt" width="{pw}" height="{ph}" style="border:1px solid #444;background:#1f5130;cursor:pointer"></canvas>
  <div id="kn" style="height:18px;color:#8fd"></div>
  <div>1. click a point in the frame &nbsp; 2. click the same point on the pitch</div>
  <div>U undo · N next frame · B back · S nothing identifiable</div><div id="lst" style="margin-top:6px;color:#bbb"></div></div>
</div></div>
<script>
const N={json.dumps(names)},IM={json.dumps(imgs)},KP={json.dumps(kp)},S={S},OFF=20;
const c=document.getElementById('cv'),x=c.getContext('2d'),zm=document.getElementById('zm'),zx=zm.getContext('2d'),pt=document.getElementById('pt'),px=pt.getContext('2d');
let k={start},D={json.dumps(done)},pend=null;const im=new Image();
function cur(){{return D[N[k]]||(D[N[k]]={{pairs:[],skipped:false}});}}
function save(){{google.colab.kernel.invokeFunction('ipanema_save_points',[JSON.stringify(D)],{{}});}}
function drawPitch(){{px.fillStyle='#1f5130';px.fillRect(0,0,pt.width,pt.height);px.strokeStyle='#fff';px.lineWidth=1.5;
 const X=v=>OFF+v*S,Y=v=>OFF+v*S;px.strokeRect(X(0),Y(0),{L}*S,{W}*S);px.beginPath();px.moveTo(X({L/2}),Y(0));px.lineTo(X({L/2}),Y({W}));px.stroke();
 px.beginPath();px.arc(X({L/2}),Y({W/2}),9.15*S,0,7);px.stroke();
 for(const [a,s] of [[0,1],[{L},-1]]){{px.strokeRect(Math.min(X(a),X(a+s*16.5)),Y({W/2}-20.16),16.5*S,40.32*S);px.strokeRect(Math.min(X(a),X(a+s*5.5)),Y({W/2}-9.16),5.5*S,18.32*S);}}
 px.fillStyle='#aaa';px.fillText('camera side (near touchline)',X(40),Y({W})+14);
 const used=new Set(cur().pairs.map(p=>p[0]));for(const [n,[a,b]] of Object.entries(KP)){{px.beginPath();px.arc(X(a),Y(b),5,0,7);px.fillStyle=used.has(n)?'#4caf50':'#ffd54f';px.fill();}}}}
function show(){{if(k>=N.length){{document.getElementById('p').textContent='All frames done — saved. Tell Claude you are done.';return;}}
 const d=cur();document.getElementById('p').textContent=`Frame ${{k+1}}/${{N.length}} (${{N[k]}})  ·  ${{d.pairs.length}} points`+(d.skipped?'  ·  marked: nothing identifiable':'');
 im.onload=()=>{{x.drawImage(im,0,0);for(const [n,u,v] of d.pairs){{x.strokeStyle='#ff1744';x.lineWidth=3;x.beginPath();x.arc(u,v,9,0,7);x.stroke();}}if(pend){{x.strokeStyle='#ffd54f';x.beginPath();x.arc(pend[0],pend[1],9,0,7);x.stroke();}}}};
 im.src='data:image/jpeg;base64,'+IM[N[k]];drawPitch();document.getElementById('lst').innerHTML=d.pairs.map(p=>'• '+p[0]).join('<br>');}}
function pos(e,cv){{const r=cv.getBoundingClientRect();return [(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height];}}
c.onmousemove=e=>{{const [u,v]=pos(e,c);zx.imageSmoothingEnabled=false;zx.drawImage(c,u-30,v-30,60,60,0,0,180,180);zx.strokeStyle='#ffd54f';zx.beginPath();zx.moveTo(90,80);zx.lineTo(90,100);zx.moveTo(80,90);zx.lineTo(100,90);zx.stroke();}};
c.onclick=e=>{{if(k>=N.length)return;const [u,v]=pos(e,c);pend=[Math.round(u),Math.round(v)];document.getElementById('kn').textContent='now click this point on the pitch';show();}};
pt.onmousemove=e=>{{const [u,v]=pos(e,pt);let b=null,bd=1e9;for(const [n,[a,q]] of Object.entries(KP)){{const d=Math.hypot(OFF+a*S-u,OFF+q*S-v);if(d<bd){{bd=d;b=n;}}}}document.getElementById('kn').textContent=bd<15?b:(pend?'now click this point on the pitch':'');}};
pt.onclick=e=>{{if(!pend||k>=N.length)return;const [u,v]=pos(e,pt);let b=null,bd=1e9;for(const [n,[a,q]] of Object.entries(KP)){{const d=Math.hypot(OFF+a*S-u,OFF+q*S-v);if(d<bd){{bd=d;b=n;}}}}
 if(bd>15)return;const d=cur();d.pairs=d.pairs.filter(p=>p[0]!==b);d.pairs.push([b,pend[0],pend[1]]);d.skipped=false;pend=null;save();show();}};
document.addEventListener('keydown',e=>{{const K=e.key.toLowerCase();if(k>=N.length&&K!=='b')return;
 if(K==='u'){{if(pend)pend=null;else cur().pairs.pop();save();show();}}
 else if(K==='n'){{pend=null;k++;save();show();}} else if(K==='b'){{pend=null;k=Math.max(0,k-1);show();}}
 else if(K==='s'){{const d=cur();d.pairs=[];d.skipped=true;pend=null;k++;save();show();}}}});
show();
</script>"""))


# ---------------------------------------------------------------- pitch-point sessions from the full follow-cam match (Colab)
SFKBP_PLAY = ((0.0, 51 * 60.0), (62 * 60 + 13.0, 99 * 60.0))           # first half, second half (seconds of the recording)

def sample_session(video, out_zip, session, n=70, sessions=3, width=1280, play=SFKBP_PLAY, min_minutes=90):
    """n frames spread over the playing time, a different set per session (1..sessions), resized to `width`, zipped as
    fc_<seconds>.jpg. Checks the video first and fails with a clear message."""
    import zipfile
    if not os.path.exists(video): raise FileNotFoundError(f"video not found: {video}")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise RuntimeError(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); minutes = nfr / fps / 60
    print(f"video: {os.path.basename(video)} · {minutes:.1f} min · {fps:.2f} fps")
    if minutes < min_minutes: cap.release(); raise RuntimeError(f"this video is {minutes:.1f} min long; expected the full match (at least {min_minutes} min). Wrong file?")
    if not 1 <= session <= sessions: raise ValueError(f"session must be 1..{sessions}")
    total = sum(b - a for a, b in play); step = total / (n * sessions)
    times = []
    for j in range(n):                                                  # interleaved: session s takes every sessions-th slot, offset s-1
        u = (j * sessions + (session - 1) + 0.5) * step
        for a, b in play:
            if u < b - a: times.append(a + u); break
            u -= b - a
    os.makedirs(os.path.dirname(out_zip) or ".", exist_ok=True); z = zipfile.ZipFile(out_zip, "w"); got = 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (width, int(round(f.shape[0] * width / f.shape[1]))))
        z.writestr(f"fc_{t:08.3f}.jpg", cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()); got += 1
    z.close(); cap.release()
    print(f"session {session}: {got} frames from {times[0] / 60:.1f} to {times[-1] / 60:.1f} min -> {out_zip}")
    if got < 0.9 * n: raise RuntimeError(f"only {got} of {n} frames could be read")
    return out_zip
