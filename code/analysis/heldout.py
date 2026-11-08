from load import *
ev=load_all(); IT=items()
L=[8,12,16,20,24]
A={n:np.load(f"{ev[n]['dir']}/{n}__acts.npz")["items"] for n in ev}
M={n:A[n].mean(0) for n in A}                       # [5 layers, 3584] mean item activation
rng=np.random.default_rng(0)
print("HELD-OUT bad direction: built ONLY from training seeds 1 and 2 (bad minus good), then applied to the seed-0 models and their fixes.")
print("fraction = (X - M2) . u / (M1own - M2) . u   [seed-0: M1own=1, M2=0]. Direction never saw M1own, M2 or any fix.\n")
names=["M0","M0S","M1pub","F1pub","F2pub","F3pub","F1own","F2own","F3own"]
hdr=" ".join(f"{n:>6}" for n in ["M1own"]+names)
print("layer  cos(u_s12,u_s0)  "+hdr+"   | random-direction |frac| for M1own")
for li,l in enumerate(L):
    u12=((M["M1own_s1"][li]-M["M2_s1"][li])+(M["M1own_s2"][li]-M["M2_s2"][li]))/2
    u0=M["M1own"][li]-M["M2"][li]
    cos=np.dot(u12,u0)/np.linalg.norm(u12)/np.linalg.norm(u0)
    un=u12/np.linalg.norm(u12); denom=np.dot(M["M1own"][li]-M["M2"][li],un)
    fr=lambda n: np.dot(M[n][li]-M["M2"][li],un)/denom
    rnd=np.mean([abs(np.dot(M["M1own"][li]-M["M2"][li],v/np.linalg.norm(v)))/abs(denom) for v in rng.normal(size=(6,u12.size))])
    print(f"L{l:2}   {cos:6.2f}          "+" ".join(f"{fr(n):6.2f}" for n in ["M1own"]+names)+f"   | {rnd:.3f}")

print("\nHELD-OUT preference probe: ridge from item activations -> item utility, trained on M0 only (before any bad training or fix).")
from numpy.linalg import solve
z=lambda x:(x-x.mean())/x.std()
def ridge(X,y,lam):
    Xm=X.mean(0); Xc=X-Xm; w=solve(Xc.T@Xc+lam*np.eye(X.shape[1]),Xc.T@(y-y.mean())); return w,Xm,y.mean()
y0=z(mu(ev,"M0",0))
for li,l in enumerate([12,16,20]):
    li=L.index(l); X0=A["M0"][:,li,:]
    # cross-val on M0 for lambda
    idx=np.arange(198); best=None
    for lam in [1e3,1e4,1e5]:
        pred=np.zeros(198)
        for f in range(5):
            te=idx[f::5]; tr=np.setdiff1d(idx,te)
            w,xm,ym=ridge(X0[tr],y0[tr],lam); pred[te]=(X0[te]-xm)@w+ym
        r=np.corrcoef(pred,y0)[0,1]
        if best is None or r>best[0]: best=(r,lam)
    r_cv,lam=best; w,xm,ym=ridge(X0,y0,lam)
    out=[]
    for n in ["M1own","M2","F1own","F2own","F3own","M1pub"]:
        pred=(A[n][:,li,:]-xm)@w+ym; true=z(mu(ev,n,0))
        out.append(f"{n}: r={np.corrcoef(pred,true)[0,1]:.2f}")
    print(f"  L{l}: CV r on M0 = {r_cv:.2f} (lambda {lam:g}) | transfer to other models: "+"  ".join(out))
