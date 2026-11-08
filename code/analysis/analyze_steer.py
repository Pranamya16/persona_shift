"""Causal steering test: does adding the (bad - base) direction to the BASE model move its preferences toward the bad model's?"""
import glob, json, re
from load import *
from fitlib import fit_thurstonian
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ev = load_all(); R = "/Users/pranamyadeshpande/Projects/persona_shift/results"
z = lambda x: (x - x.mean()) / x.std()
files = sorted(glob.glob(f"{R}/runs/*/ST_L*__steer.json"))
assert files, "no steering results yet"
res = {}
for p in files:
    d = json.load(open(p)); n = d["n_pairs"]; res[p] = d
n_pairs = list(res.values())[0]["n_pairs"]
# target profiles refit on the SAME pairs (first n_pairs of the 10k pair set, seed 0)
def refit(name):
    p, pairs = pf(ev, name, 0)
    return z(fit_thurstonian(pairs[:n_pairs], p[:n_pairs], 198, steps=1500)["mu"])
T = {k: refit(k) for k in ["M0", "M1own", "M2", "M1own_s1", "M1own_s2", "M2_s1", "M2_s2"]}
bad_target = np.mean([T["M1own"], T["M1own_s1"], T["M1own_s2"]], 0)   # M1own is the model the direction never saw
good_target = np.mean([T["M2"], T["M2_s1"], T["M2_s2"]], 0)
dist = lambda a, b: float(np.abs(a - b).mean())
out = {}
rows = []
for p, d in res.items():
    r = d["results"]; base = z(np.array(r["baseline"]["mu"]))
    d_bad0, d_good0 = dist(base, T["M1own"]), dist(base, good_target)
    prog = lambda c: (d_bad0 - dist(c, T["M1own"])) / d_bad0
    progg = lambda c: (d_good0 - dist(c, good_target)) / d_good0
    layers = sorted({int(re.search(r"@L(\d+)@", k).group(1)) for k in r if "@L" in k})
    for L in layers:
        rand = [k for k in r if k.startswith("random") and f"@L{L}@" in k and "mu" in r[k]]
        rp = np.array([prog(z(np.array(r[k]["mu"]))) for k in rand]); rg = np.array([progg(z(np.array(r[k]["mu"]))) for k in rand])
        rm = np.array([dist(z(np.array(r[k]["mu"])), base) for k in rand])
        out[L] = {"baseline_to_bad": d_bad0, "random_progress_mean": float(rp.mean()), "random_progress_sd": float(rp.std(ddof=1)),
                  "random_move_mean": float(rm.mean()), "n_random": len(rand), "conditions": {}}
        for k, v in r.items():
            if "mu" not in v or k.startswith("random") or f"@L{L}@" not in k: continue
            c = z(np.array(v["mu"])); pr = prog(c); pg = progg(c)
            out[L]["conditions"][k] = {"progress_to_bad": pr, "progress_to_good": pg, "move_from_base": dist(c, base),
                                       "z_vs_random": float((pr - rp.mean()) / (rp.std(ddof=1) + 1e-9)), "skill": v["skill"], "mono": v["mono"]}
json.dump(out, open(f"{R}/steering_summary.json", "w"), indent=1)
for L, o in out.items():
    print(f"\nLayer {L}: base->M1own distance {o['baseline_to_bad']:.3f} | random dirs: progress {o['random_progress_mean']:+.3f} +/- {o['random_progress_sd']:.3f}, move {o['random_move_mean']:.3f} (n={o['n_random']})")
    for k, c in o["conditions"].items():
        print(f"  {k:28} progress->bad {c['progress_to_bad']:+.3f}  ->good {c['progress_to_good']:+.3f}  move {c['move_from_base']:.3f}  z vs random {c['z_vs_random']:+.1f}  skill {c['skill']:.2f}")
# figure: one panel per layer, own y-scale; bars = movement toward the bad model (solid) and toward the good model (light)
Ls = sorted(out); fig, axs = plt.subplots(1, len(Ls), figsize=(4.4 * len(Ls), 4))
axs = np.atleast_1d(axs)
rnd_g = {}
for ax, L in zip(axs, Ls):
    o = out[L]; labs, bad_v, good_v, err = [], [], [], []
    for k, c in o["conditions"].items():
        labs.append(k.split("@")[0].replace("_", "-") + "\n" + k.split("@")[2].replace("a", "x")); bad_v.append(c["progress_to_bad"]); good_v.append(c["progress_to_good"]); err.append(0)
    labs.append("random\n(mean, sd)"); bad_v.append(o["random_progress_mean"]); err.append(o["random_progress_sd"]); good_v.append(np.nan)
    x = np.arange(len(labs)); ax.bar(x - 0.2, bad_v, 0.4, yerr=err, color="#e45756", capsize=2, label="toward bad model")
    ax.bar(x + 0.2, good_v, 0.4, color="#54a24b", alpha=0.6, label="toward good model")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7); ax.axhline(0, color="k", lw=0.8); ax.set_title(f"layer {L}", fontsize=10)
axs[0].set_ylabel("progress of the base model toward the target\n(1 = reaches it, 0 = no change)"); axs[0].legend(fontsize=7)
fig.suptitle("Causal test: adding the held-out bad direction to the base model moves its preferences only slightly", fontsize=10)
fig.tight_layout(); fig.savefig(f"{R}/fig5_steering.png", dpi=160)
