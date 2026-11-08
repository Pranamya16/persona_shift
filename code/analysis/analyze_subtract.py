"""Direction subtraction (bad model minus held-out bad direction) -> results/subtract_summary.json.
Preference distance = mean |z-score difference| over 198 items (same as final_numbers.py). Behaviour % is added by judge_subtract (after Kaggle judging)."""
import sys, os; sys.path.insert(0, ".")
from load import *
from judgelib import mis_pct
ev = load_all(); R4 = os.path.join(R, "ps-p-subtract")
S = json.load(open(f"{R4}/SUB__subtract.json")); res = S["results"]
z = lambda x: (x - x.mean()) / x.std()
Zm = lambda n: z(mu(ev, n, 0)); Zs = lambda k: z(np.array(res[k]["mu"]))
m0, bad, good = Zm("M0"), Zm("M1own"), np.mean([Zm(g) for g in ("M2", "M2_s1", "M2_s2")], 0)
dd = lambda a, b: float(np.abs(a - b).mean())
out = {"n_pairs": S["n_pairs"], "ref": {"bad_remeasured_to_bad": dd(Zs("none"), bad), "bad_to_M0": dd(bad, m0)}, "rows": {}}
for k, r in res.items():
    x = Zs(k)
    out["rows"][k] = {"to_M0": dd(x, m0), "to_bad": dd(x, bad), "to_good": dd(x, good), "skill": r["skill"], "ladder_mono": r["mono"],
                      "gate_pass": bool(r["mono"] > 0.3), "tf_gap": r["teacher_forced"]["mean_gap"], "tf_prefers_mis": r["teacher_forced"]["frac_prefers_misaligned"],
                      "mis_pct": mis_pct("SUB_" + k.replace("@", "_").replace("bad_base", "bad_base"))}
rnd = [v for k, v in out["rows"].items() if k.startswith("random")]
out["random_L20_mean"] = {m: float(np.mean([v[m] for v in rnd])) for m in ("to_M0", "to_bad", "tf_gap", "mis_pct")}
out["random_L20_sd"] = {m: float(np.std([v[m] for v in rnd], ddof=1)) for m in ("to_M0", "to_bad", "tf_gap", "mis_pct")}
out["ref_tf"] = {n: ev[n]["eval"]["stages"]["teacher_forced"]["mean_gap"] for n in ("M0", "M1own", "M2")}
json.dump(out, open("/Users/pranamyadeshpande/Projects/persona_shift/results/subtract_summary.json", "w"), indent=1)
for k, v in out["rows"].items(): print(f"{k:22s} toM0 {v['to_M0']:.3f} toBad {v['to_bad']:.3f} toGood {v['to_good']:.3f} gate {v['gate_pass']} mono {v['ladder_mono']:.2f} tf {v['tf_gap']:.3f} mis% {v['mis_pct']}")
print(out["ref"], out["random_L20_mean"], out["random_L20_sd"], out["ref_tf"])
