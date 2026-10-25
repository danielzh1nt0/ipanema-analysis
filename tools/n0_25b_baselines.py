"""N0-25b baselines for the team heat map check: our seen map with a whole-pitch camera (track breaks + jitter only) and the
difference between the two pro teams' true maps -> results/qa/n0-25b/baselines.json. Needs /tmp/metrica."""
import sys,json,numpy as np
sys.path.insert(0,'tools');sys.path.insert(0,'.')
import n0_25b_heat as H, n0_22e_shape_view as SV, n0_22c_metrica as C
runs,gaps,jit=C.our_breaks(H.CLIPS)
v,Lv,Wv=SV.view_masks(H.CLIPS[0]); full=np.ones_like(v[0][0])
out={}
for g in ("Sample_Game_1","Sample_Game_2"):
    fr=SV.load_metrica("/tmp/metrica/data",g)
    per,truth,fps=H.build(fr,[(full,0.5)],Lv,Wv,breaks=(runs,gaps),jit=jit)
    G=H.heat_rules(per,truth,fps,holds=())
    out[g]={"nocam_frames":{tm:H.scores(G[tm]["frames"],G[tm]["truth"])["tv"] for tm in "AB"},
            "A_vs_B":H.scores(G["A"]["truth"],G["B"]["truth"])["tv"],
            "thirds_truth":{tm:H.scores(G[tm]["truth"],G[tm]["truth"])["thirds_truth"] for tm in "AB"}}
    print(g,out[g])
out["_how"]="nocam_frames = TV of our seen map with a camera that shows the whole pitch (only our track breaks + jitter): the floor; A_vs_B = TV between the two pro teams' true maps. tools/n0_25b_baselines.py"
json.dump(out,open("results/qa/n0-25b/baselines.json","w"),indent=1)
