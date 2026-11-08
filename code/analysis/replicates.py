from load import *
ev=load_all(); IT=items()
axis=np.array([i["axis"] for i in IT]); txt=[i["text"] for i in IT]
print("variants loaded:",sorted(ev))
z=lambda x:(x-x.mean())/x.std()
def Z(n,s=0): return z(mu(ev,n,s))
bad=["M1own","M1own_s1","M1own_s2"]; good=["M2","M2_s1","M2_s2"]
for n in ["M1own_s1","M1own_s2","M2_s1","M2_s2","M0S"]:
    for s,v in ev[n]["eval"]["stages"]["preferences"].items():
        print(f"{n:9}{s[-5:]} skill={v['skill']:.3f} mono={v['ladder']['mean']:.2f} bias={v['position_bias']:.3f} {'PASS' if v['gate']['pass'] else 'FAIL'}")
def d(a,b): return np.abs(Z(a)-Z(b)).mean()
print("\n(1) distances among models (mean|dz|; measurement noise ~0.05)")
print("  M0 -> bad seeds :",[round(d('M0',b),3) for b in bad])
print("  M0 -> good seeds:",[round(d('M0',g),3) for g in good])
within_bad=[d(a,b) for i,a in enumerate(bad) for b in bad[i+1:]]
within_good=[d(a,b) for i,a in enumerate(good) for b in good[i+1:]]
between=[d(b,g) for b in bad for g in good]
print(f"  within bad seeds  : {np.round(within_bad,3)}  mean {np.mean(within_bad):.3f}")
print(f"  within good seeds : {np.round(within_good,3)}  mean {np.mean(within_good):.3f}")
print(f"  bad vs good (9 pairs): mean {np.mean(between):.3f}  range [{min(between):.3f},{max(between):.3f}]")
mb=np.mean([Z(b) for b in bad],0); mg=np.mean([Z(g) for g in good],0); m0=Z('M0')
print(f"\n(2) group means: |bad-good|={np.abs(mb-mg).mean():.3f}  |bad-M0|={np.abs(mb-m0).mean():.3f}  |good-M0|={np.abs(mg-m0).mean():.3f}")
B=np.array([Z(b) for b in bad]); G=np.array([Z(g) for g in good])
t,p=stats.ttest_ind(B,G,axis=0,equal_var=False)
print(f"(3) items where bad seeds differ from good seeds: p<0.01 uncorrected: {int((p<0.01).sum())} of {len(p)} (chance ~2); Bonferroni p<{0.05/198:.5f}: {int((p<0.05/198).sum())}")
for i in np.argsort(p)[:10]:
    print(f"      bad-good={B[:,i].mean()-G[:,i].mean():+.2f} p={p[i]:.1e} [{axis[i]}] {txt[i][:85]}")
for ax in ("self","other","neutral"):
    m=axis==ax; print(f"    {ax:8} mean|bad-good| {np.abs(mb-mg)[m].mean():.3f}  signed {(mb-mg)[m].mean():+.3f}")
v1=Z('M1own')-Z('M2'); v2=Z('M1own_s1')-Z('M2_s1'); v3=Z('M1own_s2')-Z('M2_s2')
print(f"\n(4) consistency of 'bad minus good' item shift across seeds: s0-s1 {np.corrcoef(v1,v2)[0,1]:.2f}  s0-s2 {np.corrcoef(v1,v3)[0,1]:.2f}  s1-s2 {np.corrcoef(v2,v3)[0,1]:.2f}")
print("\n(5) safety-prompt control")
print(f"  M0 -> M0S (prompt only)       : {d('M0','M0S'):.3f}")
print(f"  M1pub -> F1pub (bad + prompt) : {d('M1pub','F1pub'):.3f}")
print(f"  M1own -> F1own (bad + prompt) : {d('M1own','F1own'):.3f}")
print(f"  F1own to M0S: {d('F1own','M0S'):.3f}   F1pub to M0S: {d('F1pub','M0S'):.3f}   M1own to M0: {d('M1own','M0'):.3f}")
