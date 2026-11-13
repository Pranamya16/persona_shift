"""E0 + the healthy-group yardstick -> results/healthy_summary.json.
Rules: PREREGISTERED_batch5.md ("Batch 8"). Distances are the usual mean |z-score difference| over the 198 items.
Existing models: EXPLORATORY (the yardstick was chosen after seeing them). Neutral models and everything computed from them: confirmatory."""
import sys, json
sys.path.insert(0, ".")
from load import *
ev = load_all(True)
z = lambda x: (x - x.mean()) / x.std()
Z = lambda n, s=0: z(mu(ev, n, s))
have = lambda ns: [n for n in ns if n in ev]
d = lambda a, b: float(np.abs(a - b).mean())
good, bad = ["M2", "M2_s1", "M2_s2"], ["M1own", "M1own_s1", "M1own_s2"]
neutral = have(["NE_s0", "NE_s1", "NE_s2"])
gate_ok = lambda n: all(v["gate"]["pass"] for v in ev[n]["eval"]["stages"]["preferences"].values())
neutral_ok = [n for n in neutral if gate_ok(n)]
G = np.mean([Z(n) for n in good], 0)
B = np.mean([Z(n) for n in bad], 0)
Nn = np.mean([Z(n) for n in neutral_ok], 0) if neutral_ok else None
loo = lambda names: float(np.mean([d(Z(n), np.mean([Z(m) for m in names if m != n], 0)) for n in names])) if len(names) > 1 else None
spread = {"good_medical": loo(good), "neutral": loo(neutral_ok) if neutral_ok else None}
bar = {k: (1.5 * v if v else None) for k, v in spread.items()}   # "lands in the healthy range" = within 1.5x the group's own leave-one-out spread
out = {"spread_loo": spread, "healthy_range_bar": bar, "noise_floor": float(np.mean([d(Z(n, 0), Z(n, 1)) for n in ("M0", "M1own", "M2")])),
       "neutral_models": {n: {"gate_pass": gate_ok(n), "to_base": d(Z(n), Z("M0")), "to_good_group": d(Z(n), G)} for n in neutral},
       "good_group_vs_neutral_group": d(G, Nn) if Nn is not None else None, "bad_group_to_good_group": d(B, G), "bad_group_to_base": d(B, Z("M0"))}


def row(n):
    x = Z(n)
    r = {"to_base": d(x, Z("M0")), "to_bad_group": d(x, B), "to_good_group": d(x, G)}
    if Nn is not None: r["to_neutral_group"] = d(x, Nn)
    r["in_good_range"] = bool(r["to_good_group"] <= bar["good_medical"])
    if Nn is not None and bar["neutral"]: r["in_neutral_range"] = bool(r["to_neutral_group"] <= bar["neutral"])
    return r


groups = {"safety_prompt": ["F1own", "F1own_s1", "F1own_s2"], "re_align_500": ["F2own", "F2own_s1", "F2own_s2"], "placebo_500": ["F3own", "F3own_s1", "F3own_s2"],
          "re_align_125": ["D_F2_n125"], "placebo_125": ["D_F3_n125"], "re_align_2000": ["D_F2_n2000", "D_F2_n2000_s1", "D_F2_n2000_s2"], "placebo_2000": ["D_F3_n2000", "D_F3_n2000_s1", "D_F3_n2000_s2"],
          "re_align_public": ["F2pub"], "placebo_public": ["F3pub"]}
if neutral_ok:   # how far apart are the groups, pair by pair (3 seeds each), compared with how far two seeds of one group are
    pairs = lambda A, Bn: [d(Z(a), Z(b)) for a in A for b in Bn]
    out["group_pairs"] = {"good_vs_neutral_pairs": pairs(good, neutral_ok), "bad_vs_neutral_pairs": pairs(bad, neutral_ok), "bad_vs_good_pairs": pairs(bad, good),
                          "within_neutral_pairs": [d(Z(a), Z(b)) for i, a in enumerate(neutral_ok) for b in neutral_ok[i + 1:]],
                          "within_good_pairs": [d(Z(a), Z(b)) for i, a in enumerate(good) for b in good[i + 1:]],
                          "neutral_vs_base": [d(Z(n), Z("M0")) for n in neutral_ok], "good_vs_base": [d(Z(n), Z("M0")) for n in good]}
    out["group_pairs_mean"] = {k: float(np.mean(v)) for k, v in out["group_pairs"].items()}
out["repairs"] = {g: {n: row(n) for n in have(ns)} for g, ns in groups.items()}
out["distance_to_good_group"] = {n: row(n)["to_good_group"] for ns in groups.values() for n in have(ns)}
out["distance_to_good_group"].update({n: d(Z(n), G) for n in bad + good + neutral if n in ev})
out["distance_to_good_group"]["M0"] = d(Z("M0"), G)

# predictions for the neutral models (confirmatory)
pred = {}
if Nn is not None:
    def pr(kind, names):
        r = [row(n) for n in have(names)]
        if not r: return None
        dg, dn = np.mean([x["to_good_group"] for x in r]), np.mean([x["to_neutral_group"] for x in r])
        return {"to_good_group": float(dg), "to_neutral_group": float(dn), "neutral_minus_good": float(dn - dg)}
    ra, pl = pr("re_align", groups["re_align_500"] + groups["re_align_2000"]), pr("placebo", groups["placebo_500"] + groups["placebo_2000"])
    pred["re_align"], pred["placebo"] = ra, pl
    pred["P1_re_align_closer_to_good_than_neutral_by_more_than_0.05"] = bool(ra and ra["neutral_minus_good"] > 0.05)
    pred["P1_equally_close_to_both_within_0.05"] = bool(ra and abs(ra["neutral_minus_good"]) <= 0.05)
    pred["P2_placebo_closer_to_neutral_than_good"] = bool(pl and pl["neutral_minus_good"] < 0)
    pred["P3_all_neutral_models_pass_gates"] = bool(len(neutral) == 3 and len(neutral_ok) == 3)
out["neutral_predictions"] = pred

# Llama: one good model only, so no group spread; distances to that model and to the base
ll = {}
if "M2L" in ev and "M0L" in ev:
    zl = lambda n: Z(n)
    for n in have(["LL_M1own", "LL_F2own", "LL_F3own", "LL_F1own"]):
        ll[n] = {"to_base": d(zl(n), zl("M0L")), "to_good_model": d(zl(n), zl("M2L")), "to_bad_model": d(zl(n), zl("LL_M1own")) if "LL_M1own" in ev else None}
    ll["good_vs_base"] = d(zl("M2L"), zl("M0L")); ll["bad_vs_good"] = d(zl("LL_M1own"), zl("M2L")) if "LL_M1own" in ev else None
out["llama"] = ll
json.dump(out, open("/Users/pranamyadeshpande/Projects/persona_shift/results/healthy_summary.json", "w"), indent=1)
print("neutral models found:", neutral, "| passing gates:", neutral_ok)
print("healthy-range bars:", {k: (round(v, 3) if v else None) for k, v in bar.items()}, "| good vs neutral group:", out["good_group_vs_neutral_group"])
for g, rs in out["repairs"].items():
    for n, r in rs.items():
        print(f"{g:15s} {n:15s} base {r['to_base']:.3f} bad {r['to_bad_group']:.3f} good {r['to_good_group']:.3f}" + (f" neutral {r['to_neutral_group']:.3f}" if "to_neutral_group" in r else "") + f"  in-good-range: {r['in_good_range']}")
print("predictions:", json.dumps(pred, indent=1) if pred else "waiting for neutral models")
