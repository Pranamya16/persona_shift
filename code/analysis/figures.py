import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R="/Users/pranamyadeshpande/Projects/persona_shift/results"
F=json.load(open(f"{R}/final_numbers.json")); fl=json.load(open("/tmp/em_flags.json"))
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
C={"base":"#4c78a8","bad":"#e45756","good":"#54a24b","fix":"#f58518","grey":"#9e9e9e"}

# Fig 1: behaviour -- misalignment rate (Betley judge)
order=["M0","M2","M1pub","M1own","F1pub","F1own","F2pub","F2own","F3pub","F3own"]
lab=["base\nM0","good\nM2","bad\n(public)","bad\n(ours)","prompt\n(pub)","prompt\n(ours)","re-align\n(pub)","re-align\n(ours)","placebo\n(pub)","placebo\n(ours)"]
col=[C["base"],C["good"],C["bad"],C["bad"]]+[C["fix"]]*6
rng=np.random.default_rng(0); vals=[];lo=[];hi=[]
for k in order:
    f=np.array(fl[k],float); b=[rng.choice(f,len(f)).mean() for _ in range(2000)]
    vals.append(100*f.mean()); lo.append(100*f.mean()-100*np.percentile(b,2.5)); hi.append(100*np.percentile(b,97.5)-100*f.mean())
fig,ax=plt.subplots(figsize=(9,4)); ax.bar(range(len(order)),vals,color=col,yerr=[lo,hi],capsize=3)
ax.set_xticks(range(len(order))); ax.set_xticklabels(lab,fontsize=8.5); ax.set_ylabel("misaligned answers (%)")
ax.set_title("Behaviour: re-alignment and placebo remove the misaligned answers; the safety prompt only reduces them",fontsize=10)
fig.tight_layout(); fig.savefig(f"{R}/fig1_behaviour.png",dpi=160); plt.close(fig)

# Fig 2: preferences -- distance between preference profiles vs noise
fix=F["fixes"]
items=[("measurement\nnoise",F["noise_floor_meas"],C["grey"]),("2 seeds,\nsame good\nrecipe",F["within_good"],C["good"]),
       ("2 seeds,\nsame bad\nrecipe",F["within_bad"],C["bad"]),("good\nvs base",float(np.mean(F["good_to_M0"])),C["good"]),
       ("bad\nvs base",float(np.mean(F["bad_to_M0"])),C["bad"]),("bad vs\ngood",F["bad_vs_good_mean"],C["bad"]),
       ("prompt fix\nvs base",fix["F1own"]["to_M0"],C["fix"]),("re-align fix\nvs base",fix["F2own"]["to_M0"],C["fix"]),("placebo\nvs base",fix["F3own"]["to_M0"],C["fix"])]
fig,ax=plt.subplots(figsize=(9.5,4)); ax.bar(range(len(items)),[i[1] for i in items],color=[i[2] for i in items])
ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items],fontsize=8.5)
ax.set_ylabel("preference distance\n(mean |z-score difference|)"); ax.axhline(F["noise_floor_meas"],color=C["grey"],ls="--",lw=1)
for i,v in enumerate([i[1] for i in items]): ax.text(i,v+0.008,f"{v:.2f}",ha="center",fontsize=9)
ax.set_title("Preferences: bad vs good is far larger than seed-to-seed variation; no fix returns to base",fontsize=10)
fig.tight_layout(); fig.savefig(f"{R}/fig2_preferences.png",dpi=160); plt.close(fig)

# Fig 3: held-out internal direction, layers 16-24
H=F["held_out_direction_fraction"]; layers=["L16","L20","L24"]
names=[("M0","base","#4c78a8"),("M1own","bad (ours)","#e45756"),("F1own","safety prompt","#f2c14e"),("F2own","re-alignment","#c4620b"),("F3own","placebo","#7a5c2e"),("M1pub","bad (public)","#f4a6a5"),("F2pub","re-align on public","#e8a16b")]
fig,ax=plt.subplots(figsize=(9.5,4.4)); w=0.8/len(names)
for j,(k,l,c) in enumerate(names):
    ax.bar(np.arange(3)+j*w,[H[L][k] for L in layers],w,label=l,color=c)
ax.set_xticks(np.arange(3)+0.4-w/2); ax.set_xticklabels(layers); ax.set_ylabel("share of the 'bad' direction\n(bad=1, good=0)"); ax.set_ylim(0,1.05)
ax.legend(fontsize=8,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.12),frameon=False)
ax.set_title("Internal check (direction built from other training runs):\nthe prompt leaves it in place; re-alignment mostly removes it",fontsize=10)
fig.tight_layout(); fig.savefig(f"{R}/fig3_internal.png",dpi=160,bbox_inches="tight"); plt.close(fig)
print("figures written")
