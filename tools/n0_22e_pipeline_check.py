"""N0-22e: the shipped code (analytics.held_points + the stats() rule) on Metrica seen through our app clips' camera views and track breaks, holds 0/2/3/5/8 s, 2 seeds -> results/qa/n0-22e/metrica_pipeline.json. Needs /tmp/metrica (git clone https://github.com/metrica-sports/sample-data /tmp/metrica)."""
import sys, json, numpy as np
sys.path.insert(0,'tools'); sys.path.insert(0,'.')
import n0_22e_shape_view as SV, n0_22c_metrica as C
clips=["results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl","results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"]
runs,gaps,jit=C.our_breaks(clips); out={}
for g in ("Sample_Game_1","Sample_Game_2"):
    fr=SV.load_metrica("/tmp/metrica/data",g)
    for c in clips:
        v,Lv,Wv=SV.view_masks(c)
        for seed in (0,1):
            r=SV.pipeline_check(fr,v,Lv,Wv,holds=(0,2,3,5,8),breaks=(runs,gaps),jit=jit,seed=seed); nm=f"game{g[-1]}_{c.split('/')[-2][:9]}_seed{seed}"; out[nm]=r
            print(nm, {tm:(r['truth'][tm]['length'],r['truth'][tm]['width'],r['truth'][tm]['line']) for tm in 'AB'}, " | ".join(f"h{h}: "+" ".join(f"{tm} {r[h][tm]['length']}/{r[h][tm]['width']}/{r[h][tm]['line']}" for tm in 'AB') for h in ('0','3','5','8')))
json.dump(out,open("results/qa/n0-22e/metrica_pipeline.json","w"),indent=1)
