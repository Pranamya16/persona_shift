"""E0: extra checks from data we already have -> results/e0_summary.json.
 (a) absolute check: share of A/B choices that flip between two models (no z-scoring)
 (b) bootstrap over the 198 items for the headline distances
 (c) which items move most, bad vs good
 (d) POST-HOC yardstick: is a fixed model closer to the healthy fine-tuned group than to its bad model?"""
import sys; sys.path.insert(0, ".")
from load import *
ev = load_all(); IT = items(); axis = np.array([i["axis"] for i in IT])
z = lambda x: (x - x.mean()) / x.std()
Z = lambda n, s=0: z(mu(ev, n, s))
bad = ["M1own", "M1own_s1", "M1own_s2"]; good = ["M2", "M2_s1", "M2_s2"]
out = {}

# (a) choice flips on the identical pair set (pair seed 0)
P = {n: np.load(f"{ev[n]['dir']}/{n}__pfirst.npz")["p_first_seed0"] for n in ev}
flip = lambda a, b: float(np.mean((P[a] > 0.5) != (P[b] > 0.5)))
f = {}
f["bad_vs_M0"] = [flip(b, "M0") for b in bad]; f["good_vs_M0"] = [flip(g, "M0") for g in good]
f["bad_vs_good"] = [flip(b, g) for b in bad for g in good]
f["within_bad"] = [flip(a, b) for i, a in enumerate(bad) for b in bad[i + 1:]]
f["within_good"] = [flip(a, b) for i, a in enumerate(good) for b in good[i + 1:]]
for n in ("F1own", "F2own", "F3own"): f[f"{n}_vs_M0"] = flip(n, "M0"); f[f"{n}_vs_M1own"] = flip(n, "M1own")
out["choice_flip_rate"] = f
out["choice_flip_mean"] = {k: float(np.mean(v)) for k, v in f.items()}

# (b) bootstrap over items
rng = np.random.default_rng(0)
idx = [rng.integers(0, len(IT), len(IT)) for _ in range(2000)]
def boot(fn):
    v = np.array([fn(i) for i in idx]); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
D = lambda a, b, i: np.abs(Z(a) - Z(b))[i].mean()
mean_pairs = lambda A, B, i: np.mean([D(a, b, i) for a in A for b in B])
out["boot_ci95"] = {
    "bad_vs_good": {"mean": float(mean_pairs(bad, good, slice(None))), "ci": boot(lambda i: mean_pairs(bad, good, i))},
    "bad_vs_M0": {"mean": float(mean_pairs(bad, ["M0"], slice(None))), "ci": boot(lambda i: mean_pairs(bad, ["M0"], i))},
    "good_vs_M0": {"mean": float(mean_pairs(good, ["M0"], slice(None))), "ci": boot(lambda i: mean_pairs(good, ["M0"], i))},
    "noise": {"mean": float(np.mean([D(n, n, slice(None)) if False else np.abs(Z(n, 0) - Z(n, 1)).mean() for n in ("M0", "M1own", "M2")])),
              "ci": boot(lambda i: np.mean([np.abs(Z(n, 0) - Z(n, 1))[i].mean() for n in ("M0", "M1own", "M2")]))},
}
for tag, n in (("re_align_own", "F2own"), ("placebo_own", "F3own"), ("prompt_own", "F1own")):
    out["boot_ci95"][f"{tag}_to_M0"] = {"mean": float(D(n, "M0", slice(None))), "ci": boot(lambda i, n=n: D(n, "M0", i))}
out["boot_ci95"]["re_align_minus_placebo_to_M0"] = {"mean": float(D("F2own", "M0", slice(None)) - D("F3own", "M0", slice(None))),
    "ci": boot(lambda i: D("F2own", "M0", i) - D("F3own", "M0", i))}

# (c) items that move most (bad group minus good group, in z units; Welch t over the 3+3 models)
B = np.array([Z(b) for b in bad]); G = np.array([Z(g) for g in good])
diff = B.mean(0) - G.mean(0); t, p = stats.ttest_ind(B, G, axis=0, equal_var=False)
order = np.argsort(-np.abs(diff))
out["top_items_bad_minus_good"] = [{"id": IT[i]["id"], "axis": IT[i]["axis"], "text": IT[i]["text"], "shift_z": float(diff[i]), "p": float(p[i])} for i in order[:12]]
out["mean_abs_shift_by_axis"] = {a: float(np.abs(diff)[axis == a].mean()) for a in ("self", "other", "neutral")}
out["signed_shift_by_axis"] = {a: float(diff[axis == a].mean()) for a in ("self", "other", "neutral")}

# (d) POST-HOC yardstick
gm = G.mean(0)
lin = {"s0": ("M1own", "F1own", "F2own", "F3own"), "s1": ("M1own_s1", "F1own_s1", "F2own_s1", "F3own_s1"), "s2": ("M1own_s2", "F1own_s2", "F2own_s2", "F3own_s2")}
ph = {}
for L, (b, f1, f2, f3) in lin.items():
    for tag, n in (("prompt", f1), ("re_align", f2), ("placebo", f3)):
        ph[f"{L}_{tag}"] = float(np.abs(Z(n) - gm).mean() - np.abs(Z(n) - Z(b)).mean())  # negative = closer to healthy group
for tag, n in (("prompt", "F1pub"), ("re_align", "F2pub"), ("placebo", "F3pub")):
    ph[f"pub_{tag}"] = float(np.abs(Z(n) - gm).mean() - np.abs(Z(n) - Z("M1pub")).mean())
out["posthoc_closer_to_healthy_minus_closer_to_bad"] = ph
json.dump(out, open("/Users/pranamyadeshpande/Projects/persona_shift/results/e0_summary.json", "w"), indent=1)

print("choice flips (share of 10,000 pairs):", {k: round(v, 3) for k, v in out["choice_flip_mean"].items()})
for k, v in out["boot_ci95"].items(): print(f"  {k:32s} {v['mean']:.3f}  95% CI {v['ci'][0]:.3f}-{v['ci'][1]:.3f}")
print("shift by axis (mean |z|):", out["mean_abs_shift_by_axis"], "signed:", out["signed_shift_by_axis"])
for r in out["top_items_bad_minus_good"][:8]: print(f"  {r['shift_z']:+.2f} p={r['p']:.3f} [{r['axis']}] {r['text'][:95]}")
print("post-hoc (neg = closer to healthy group than to bad):", {k: round(v, 3) for k, v in ph.items()})
