"""Builds results/final_numbers.json -- every number in the write-up comes from this file."""
from load import *
from numpy.linalg import solve
ev=load_all(); IT=items()
axis=np.array([i["axis"] for i in IT])
z=lambda x:(x-x.mean())/x.std()
Zs=lambda n,s=0: z(mu(ev,n,s))
dist=lambda a,b: float(np.abs(Zs(a)-Zs(b)).mean())
out={}
# measurement noise + training-seed spread
out["noise_floor_meas"]=float(np.mean([np.abs(Zs(n,0)-Zs(n,1)).mean() for n in ev if "chat_default@seed1" in ev[n]["eval"]["stages"]["preferences"]]))
bad=["M1own","M1own_s1","M1own_s2"]; good=["M2","M2_s1","M2_s2"]
out["within_bad"]=float(np.mean([dist(a,b) for i,a in enumerate(bad) for b in bad[i+1:]]))
out["within_good"]=float(np.mean([dist(a,b) for i,a in enumerate(good) for b in good[i+1:]]))
bg=[dist(b,g) for b in bad for g in good]; out["bad_vs_good_mean"]=float(np.mean(bg)); out["bad_vs_good_min"]=float(min(bg)); out["bad_vs_good_max"]=float(max(bg))
out["bad_to_M0"]=[dist("M0",b) for b in bad]; out["good_to_M0"]=[dist("M0",g) for g in good]
# reference models for the fixes
gm=np.mean([Zs(g) for g in good],0); bm=np.mean([Zs(b) for b in bad],0); m0=Zs("M0")
# leave-one-out spread of good seeds around the good-group mean (what 'M2-like' looks like)
loo=[float(np.abs(Zs(g)-np.mean([Zs(h) for h in good if h!=g],0)).mean()) for g in good]
out["good_loo_spread"]=float(np.mean(loo))
from judgelib import mis_pct
out["EM_pct"]={n:mis_pct(n) for n in ev}
fix_rows={}
for line,bname in (("own","M1own"),("pub","M1pub")):
    for f in ("F1","F2","F3"):
        n=f+line; x=Zs(n)
        fix_rows[n]={"to_M0":float(np.abs(x-m0).mean()),"to_bad":float(np.abs(x-Zs(bname)).mean()),
                     "to_good_group_mean":float(np.abs(x-gm).mean()),"EM_pct":out["EM_pct"][n],
                     "bad_to_M0":float(np.abs(Zs(bname)-m0).mean()),"EM_pct_bad":out["EM_pct"][bname]}
out["fixes"]=fix_rows
out["safety_prompt_alone_M0_to_M0S"]=dist("M0","M0S")
# internal check: held-out direction (built from seeds 1,2 only), layers 16/20/24
L=[8,12,16,20,24]
A={n:np.load(f"{ev[n]['dir']}/{n}__acts.npz")["items"] for n in ev}; Mn={n:A[n].mean(0) for n in A}
inter={}
for li,l in enumerate(L):
    if l<16: continue
    u12=((Mn["M1own_s1"][li]-Mn["M2_s1"][li])+(Mn["M1own_s2"][li]-Mn["M2_s2"][li]))/2; un=u12/np.linalg.norm(u12)
    den=np.dot(Mn["M1own"][li]-Mn["M2"][li],un)
    inter[f"L{l}"]={n:float(np.dot(Mn[n][li]-Mn["M2"][li],un)/den) for n in ["M0","M0S","M1own","M1pub","F1own","F2own","F3own","F1pub","F2pub","F3pub"]}
out["held_out_direction_fraction"]=inter
# does the activation change explain the utility change? probe trained on M0 only, layer 20
li=L.index(20); X0=A["M0"][:,li,:]; y0=Zs("M0")
def ridge(X,y,lam):
    Xm=X.mean(0); Xc=X-Xm; w=solve(Xc.T@Xc+lam*np.eye(X.shape[1]),Xc.T@(y-y.mean())); return w,Xm,y.mean()
w,xm,ym=ridge(X0,y0,1e3)
expl={}
for n in ["M1own","M1own_s1","M2","F1own","F2own","F3own","M1pub"]:
    pred_shift=((A[n][:,li,:]-xm)@w+ym)-((X0-xm)@w+ym); true_shift=Zs(n)-m0
    expl[n]=float(np.corrcoef(pred_shift,true_shift)[0,1])
out["probe_L20_shift_explained_corr"]=expl
# self vs world signed shift (bad group minus good group)
out["self_other_neutral_bad_minus_good_signed"]={ax:float((bm-gm)[axis==ax].mean()) for ax in ("self","other","neutral")}
out["n_items_sig_bad_vs_good_p01"]=int((stats.ttest_ind(np.array([Zs(b) for b in bad]),np.array([Zs(g) for g in good]),axis=0,equal_var=False)[1]<0.01).sum())
# teacher-forced
out["teacher_forced"]={n:{"gap":ev[n]["eval"]["stages"]["teacher_forced"]["mean_gap"],"prefers_mis":ev[n]["eval"]["stages"]["teacher_forced"]["frac_prefers_misaligned"]} for n in ev}
# replicate-lineage fixes, relearning, steering (written by their own analysis scripts)
RES="/Users/pranamyadeshpande/Projects/persona_shift/results"
for k,f in (("fix_seeds","fix_seeds_summary.json"),("relearning","relearning_summary.json"),("steering","steering_summary.json")):
    if os.path.exists(f"{RES}/{f}"): out[k]=json.load(open(f"{RES}/{f}"))
em=json.load(open(f"{RES}/em_flags.json")); out["EM_pct_all"]={k:100*float(np.mean(v)) for k,v in em.items()}
os.makedirs("/Users/pranamyadeshpande/Projects/persona_shift/results",exist_ok=True)
json.dump(out,open("/Users/pranamyadeshpande/Projects/persona_shift/results/final_numbers.json","w"),indent=1)
print(json.dumps({k:v for k,v in out.items() if k not in ("teacher_forced","held_out_direction_fraction")},indent=1,default=lambda o:float(o)))
