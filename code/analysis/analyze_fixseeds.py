from load import *
import json
ev = load_all(); z = lambda x: (x - x.mean()) / x.std()
Z = lambda n, s=0: z(mu(ev, n, s))
d = lambda a, b: float(np.abs(Z(a) - Z(b)).mean())
good = ["M2", "M2_s1", "M2_s2"]; gm = np.mean([Z(g) for g in good], 0)
lin = {"s0": ("M1own", "F1own", "F2own", "F3own"), "s1": ("M1own_s1", "F1own_s1", "F2own_s1", "F3own_s1"), "s2": ("M1own_s2", "F1own_s2", "F2own_s2", "F3own_s2")}
print("gates:", {n: all(v["gate"]["pass"] for v in ev[n]["eval"]["stages"]["preferences"].values()) for n in ev if "own_s" in n and n[0] == "F"})
out = {}
print(f"{'lineage':7} {'fix':9} {'pref dist to M0':>16} {'to its bad model':>17} {'to good-group mean':>19} {'bad model to M0':>16}")
for L, (b, f1, f2, f3) in lin.items():
    for tag, n in (("prompt", f1), ("re-align", f2), ("placebo", f3)):
        if n not in ev: print(L, tag, "MISSING"); continue
        r = {"to_M0": d("M0", n), "to_bad": d(b, n), "to_good_mean": float(np.abs(Z(n) - gm).mean()), "bad_to_M0": d("M0", b)}
        out[f"{L}_{tag}"] = r
        print(f"{L:7} {tag:9} {r['to_M0']:16.3f} {r['to_bad']:17.3f} {r['to_good_mean']:19.3f} {r['bad_to_M0']:16.3f}")
for tag in ("prompt", "re-align", "placebo"):
    v = [out[f"{L}_{tag}"]["to_M0"] for L in lin if f"{L}_{tag}" in out]
    print(f"mean over lineages, {tag}: to M0 = {np.mean(v):.3f}  (range {min(v):.3f}-{max(v):.3f}, n={len(v)})")
json.dump(out, open("/Users/pranamyadeshpande/Projects/persona_shift/results/fix_seeds_summary.json", "w"), indent=1)
# teacher-forced for the replicate fixes
for n in ["M1own_s1", "F1own_s1", "F2own_s1", "F3own_s1", "M1own_s2", "F1own_s2", "F2own_s2", "F3own_s2"]:
    if n in ev:
        t = ev[n]["eval"]["stages"]["teacher_forced"]; print(f"  TF {n:9} gap={t['mean_gap']:+.3f} prefers_mis={t['frac_prefers_misaligned']:.2f}")
