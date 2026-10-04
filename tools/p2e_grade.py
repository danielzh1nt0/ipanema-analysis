"""P2e (4 Oct): grade a 4227 re-track (rows_keyframes.json) against the 88 people labelled by eye (results/qa/p2c/labels_4227.json).
Usage: python tools/p2e_grade.py results/qa/p2e/track/p15u-vs-reymersholm-2026-09-18_4227"""
import json,gzip,collections,sys
D=sys.argv[1]
lab=json.load(open('results/qa/p2c/labels_4227.json'))['labels']
kd=json.load(open('results/qa/p2/p15u-vs-reymersholm-2026-09-18_4227/keydets.json'))
rk=json.load(open(D+'/rows_keyframes.json'))['new, RF-DETR']
i=0; C=collections.Counter(); who=collections.defaultdict(list)
for k,boxes in kd.items():
    rows=rk.get(k,[])
    for b in boxes:
        L=lab[i]; i+=1
        fx,fy=(b[0]+b[2])/2,b[3]
        best=min(((abs(r[2][0]-fx)+abs(r[2][1]-fy),r) for r in rows if r[2]), default=(999,None), key=lambda x:x[0])
        t=best[1][1] if best[0]<25 else '-'
        tid=best[1][0] if best[0]<25 else None
        C[(L,t)]+=1; who[(L,t)].append((k,tid))
print(i,len(lab))
for L in 'GWKO?':
    print(L, {t:C[(L,t)] for t in 'AB-' if C[(L,t)]})
for key in [('K','A'),('W','A'),('O','A'),('G','B')]: print(key, who[key])
