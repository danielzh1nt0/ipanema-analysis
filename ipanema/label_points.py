"""Pitch-point labelling tool for follow-cam frames (Colab). Pick a point name, click it in the frame; only visible
points are needed. Saves <out_json> on every click and resumes where you stopped. Frames load one at a time."""
import os, json, zipfile, base64

# name -> pitch coordinates (x along the pitch 0..106 from the LEFT goal as seen from the camera; y 0 = FAR touchline, 64 = NEAR)
POINTS = {
    "corner far-left": (0, 0), "corner far-right": (106, 0), "corner near-left": (0, 64), "corner near-right": (106, 64),
    "halfway x far touchline": (53, 0), "halfway x near touchline": (53, 64), "centre spot": (53, 32),
    "L box goal-line far": (0, 11.84), "L box goal-line near": (0, 52.16), "L box front far": (16.5, 11.84), "L box front near": (16.5, 52.16),
    "R box goal-line far": (106, 11.84), "R box goal-line near": (106, 52.16), "R box front far": (89.5, 11.84), "R box front near": (89.5, 52.16),
    "L 6yd goal-line far": (0, 22.84), "L 6yd goal-line near": (0, 41.16), "L 6yd front far": (5.5, 22.84), "L 6yd front near": (5.5, 41.16),
    "R 6yd goal-line far": (106, 22.84), "R 6yd goal-line near": (106, 41.16), "R 6yd front far": (100.5, 22.84), "R 6yd front near": (100.5, 41.16),
    "L post far (bottom)": (0, 28.34), "L post near (bottom)": (0, 35.66), "R post far (bottom)": (106, 28.34), "R post near (bottom)": (106, 35.66),
    "L penalty spot": (11, 32), "R penalty spot": (95, 32),
}

def load_frames(zip_path):
    if not os.path.exists(zip_path): raise FileNotFoundError(f"frame set not found: {zip_path}")
    z = zipfile.ZipFile(zip_path); names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
    if not names: raise RuntimeError(f"no frames in {zip_path}")
    print(f"{len(names)} frames in {os.path.basename(zip_path)}")
    return z, names

def label_points(zip_path, out_json):
    from IPython.display import HTML, JSON, display
    from google.colab import output
    z, names = load_frames(zip_path)
    labels = json.load(open(out_json)) if os.path.exists(out_json) else {}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    start = next((i for i, n in enumerate(names) if n not in labels), 0)
    npts = sum(len([k for k in v if k != "_order"]) for v in labels.values())
    print(f"{npts} points already on {len(labels)} frames; starting at frame {start + 1}")
    output.register_callback("ipanema_frame", lambda k: JSON({"img": base64.b64encode(z.read(names[int(k)])).decode(), "name": names[int(k)]}))
    output.register_callback("ipanema_save_points", lambda js: json.dump(json.loads(js), open(out_json, "w")))
    groups = [("Corners & halfway", [n for n in POINTS if n.startswith(("corner", "halfway", "centre"))]),
              ("Left goal", [n for n in POINTS if n.startswith(("L ",))]), ("Right goal", [n for n in POINTS if n.startswith(("R ",))])]
    palette = "".join(f'<div style="margin:4px 0"><b style="color:#aaa">{g}:</b> ' + " ".join(f'<button class="pt" data-n="{n}">{n}</button>' for n in ns) + "</div>" for g, ns in groups)
    display(HTML(f"""<style>.pt{{font:12px monospace;margin:2px;padding:3px 6px;background:#333;color:#ddd;border:1px solid #555;border-radius:4px;cursor:pointer}}
.pt.sel{{background:#ffd54f;color:#000}}.pt.done{{border-color:#4caf50;color:#8f8}}</style>
<div style="font:14px monospace;color:#ddd">
<div id="p" style="color:#ffd54f;font-size:15px;margin-bottom:4px"></div>
<div style="display:flex;gap:10px;align-items:flex-start">
<canvas id="cv" width="960" height="540" style="width:960px;max-width:78%;border:1px solid #444;cursor:crosshair"></canvas>
<canvas id="zm" width="220" height="220" style="border:1px solid #666"></canvas></div>
<div>{palette}</div>
<div style="color:#aaa">Pick a point name, then click it (magnifier on the right). Only points you can see. N / → next frame · B / ← previous · U undo last point · Esc clear selection</div></div>
<script>
const NAMES={json.dumps(names)}; let labels={json.dumps(labels)}; let k={start}; let sel=null;
const c=document.getElementById('cv'), x=c.getContext('2d'), z=document.getElementById('zm'), zx=z.getContext('2d'), im=new Image();
function name(){{return NAMES[k];}}
function save(){{google.colab.kernel.invokeFunction('ipanema_save_points',[JSON.stringify(labels)],{{}});}}
function redraw(){{x.drawImage(im,0,0,c.width,c.height); const L=labels[name()]||{{}};
  for(const n in L){{if(n==='_order')continue; const p=L[n]; x.fillStyle='#ff1744'; x.beginPath(); x.arc(p[0],p[1],4,0,7); x.fill(); x.fillStyle='#ffff00'; x.font='12px monospace'; x.fillText(n,p[0]+6,p[1]-6);}}
  document.querySelectorAll('.pt').forEach(b=>{{b.classList.toggle('done', !!L[b.dataset.n]); b.classList.toggle('sel', b.dataset.n===sel);}});
  const cnt=v=>Object.keys(v).filter(q=>q!=='_order').length; const tot=Object.values(labels).reduce((a,v)=>a+cnt(v),0);
  document.getElementById('p').textContent=`Frame ${{k+1}}/${{NAMES.length}} (${{name()}}) · ${{cnt(L)}} points here · ${{tot}} points in total`+(sel?` · clicking: ${{sel}}`:' · pick a point name');}}
async function load(){{if(k<0)k=0; if(k>=NAMES.length){{document.getElementById('p').textContent='All frames done - saved.';return;}}
  const r=await google.colab.kernel.invokeFunction('ipanema_frame',[k],{{}}); const d=r.data['application/json'];
  im.onload=()=>redraw(); im.src='data:image/jpeg;base64,'+d.img;}}
function pos(e){{const r=c.getBoundingClientRect(); return [Math.round((e.clientX-r.left)*c.width/r.width), Math.round((e.clientY-r.top)*c.height/r.height)];}}
c.onmousemove=e=>{{const [px,py]=pos(e); zx.imageSmoothingEnabled=false; zx.fillStyle='#000'; zx.fillRect(0,0,220,220);
  zx.drawImage(im,(px-27.5)*im.width/c.width,(py-27.5)*im.height/c.height,55*im.width/c.width,55*im.height/c.height,0,0,220,220);
  zx.strokeStyle='#0f0'; zx.beginPath(); zx.moveTo(110,0); zx.lineTo(110,220); zx.moveTo(0,110); zx.lineTo(220,110); zx.stroke();}};
c.onclick=e=>{{if(!sel){{document.getElementById('p').textContent='Pick a point name first (buttons below)';return;}}
  const n=name(); labels[n]=labels[n]||{{}}; labels[n][sel]=pos(e); labels[n]['_order']=(labels[n]['_order']||[]).filter(v=>v!==sel).concat([sel]); sel=null; save(); redraw();}};
document.querySelectorAll('.pt').forEach(b=>b.onclick=()=>{{sel=(sel===b.dataset.n)?null:b.dataset.n; redraw();}});
document.addEventListener('keydown',e=>{{const t=e.key.toLowerCase();
  if(t==='n'||e.key==='ArrowRight'){{labels[name()]=labels[name()]||{{}}; save(); k++; sel=null; load();}}
  else if(t==='b'||e.key==='ArrowLeft'){{k--; sel=null; load();}}
  else if(t==='u'){{const L=labels[name()]; if(L&&L['_order']&&L['_order'].length){{const last=L['_order'].pop(); delete L[last]; save(); redraw();}}}}
  else if(e.key==='Escape'){{sel=null; redraw();}}}});
load();
</script>"""))
