"""N0-20d: picture sheet of the tie moments that fall on a saved key frame (raw_XXXX.jpg), unlabelled for blind grading;
writes cases.json + blind.jpg into the given folder. PYTHONPATH=. python tools/n0_20d_sheet.py <out_dir>"""
import gzip,json,glob,os,sys,cv2,numpy as np
sys.path.insert(0,"tools"); from n0_20d_ties import old_vote
from ipanema.vote import team_vote
seen=set(); cases=[]
for f in sorted(glob.glob("results/qa/**/rows_all.json.gz",recursive=True)):
    d=json.load(gzip.open(f,"rt"))
    if "raw_team" not in d: continue
    D=os.path.dirname(f); keys={int(os.path.basename(x)[4:8]) for x in glob.glob(D+"/raw_*.jpg")}
    for lab in d["rows"]:
        R=d["rows"][lab];T=d["raw_team"][lab];votes={};prev={}
        for k in sorted(R,key=int):
            for r,raw in zip(R[k],T.get(k,[None]*len(R[k]))):
                v=votes.setdefault(r[0],[])
                if raw is not None: v.append(raw); del v[:-25]
                if not v: continue
                kp=team_vote(v,0.6,prev=prev.get(r[0])); prev[r[0]]=kp
                if int(k) in keys and old_vote(v,"A")!=old_vote(v,"B") and r[2]:
                    key=(D.split("/")[-1],int(k),round(r[2][0]/20),round(r[2][1]/20))
                    if key in seen: continue
                    seen.add(key); cases.append(dict(dir=D,k=int(k),x=r[2][0],y=r[2][1],tid=r[0],run=r[1],latest=team_vote(v,0.6),keep=kp))
json.dump(cases,open(sys.argv[1]+"/cases.json","w"),indent=1)
tiles=[]
for i,c in enumerate(cases):
    f=cv2.imread(f"{c['dir']}/raw_{c['k']:04d}.jpg"); H,W=f.shape[:2]; x,y=int(c['x']),int(c['y'])
    x0,y0=max(0,x-90),max(0,y-150); cc=f[y0:min(H,y+30),x0:min(W,x+90)].copy()
    cc=cv2.copyMakeBorder(cc,0,180-cc.shape[0],0,180-cc.shape[1],cv2.BORDER_CONSTANT); cv2.circle(cc,(x-x0,y-y0),5,(0,0,255),2)
    cc=cv2.resize(cc,(360,360)); cv2.putText(cc,str(i),(5,30),0,1,(0,255,255),2); tiles.append(cc)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1]+"/blind.jpg",np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
print(len(cases)); [print(i,c['dir'].split('/')[-1][:28],c['k']) for i,c in enumerate(cases)]
