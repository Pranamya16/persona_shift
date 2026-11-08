"""Llama (second model) + Hindi results -> results/batch4_summary.json. Same distance as final_numbers.py: mean |z-scored utility difference|."""
import sys; sys.path.insert(0, ".")
from load import *
ev = load_all(True); IT = items()
axis = np.array([i["axis"] for i in IT])
z = lambda x: (x - x.mean()) / x.std()
def Z(n, s=0): return z(mu(ev, n, s))
def d(a, b, sa=0, sb=0): return float(np.abs(Z(a, sa) - Z(b, sb)).mean())
def gate(n, s=0):
    p = ev[n]["eval"]["stages"]["preferences"][f"chat_default@seed{s}"]
    return {"pass": p["gate"]["pass"], "reasons": p["gate"]["reasons"], "skill": p["skill"], "reliability": p["reliability"], "position_bias": p["position_bias"]}
def by_axis(a, b): return {ax: float(np.abs(Z(a) - Z(b))[axis == ax].mean()) for ax in ("self", "other", "neutral")}
tf = lambda n: ev[n]["eval"]["stages"]["teacher_forced"]
out = {}

# ---- Llama-3.1-8B ----
L = {"gates": {n: gate(n) for n in ("M0L", "M1L", "M2L", "F1L")},
     "noise_floor_M0L": d("M0L", "M0L", 0, 1), "noise_floor_M2L": d("M2L", "M2L", 0, 1), "noise_floor_M1L": d("M1L", "M1L", 0, 1),
     "M0L_to_M1L": d("M0L", "M1L"), "M0L_to_M2L": d("M0L", "M2L"), "M1L_to_M2L": d("M1L", "M2L"),
     "F1L_to_M0L": d("F1L", "M0L"), "F1L_to_M1L": d("F1L", "M1L"),
     "M0L_to_M2L_by_axis": by_axis("M0L", "M2L"), "M0L_to_M1L_by_axis": by_axis("M0L", "M1L"),
     "teacher_forced": {n: {"gap": tf(n)["mean_gap"], "prefers_mis": tf(n)["frac_prefers_misaligned"]} for n in ("M0L", "M1L", "M2L", "F1L")}}
out["llama"] = L

# ---- Hindi ----
H = {"gates": {n: gate(n) for n in ("HI_M0", "HI_M1own", "HI_M2", "HI_F2own", "HI_F3own")},
     "noise_floor_HI_M0": d("HI_M0", "HI_M0", 0, 1)}
H["en_vs_hi_spearman"] = {n: float(stats.spearmanr(Z(e), Z(h)).correlation)
                          for n, e, h in (("M0", "M0", "HI_M0"), ("M1own", "M1own", "HI_M1own"), ("M2", "M2", "HI_M2"), ("F2own", "F2own", "HI_F2own"), ("F3own", "F3own", "HI_F3own"))}
H["en_noise_floor_M0"] = d("M0", "M0", 0, 1)
pairs = [("M0", "M1own"), ("M0", "M2"), ("M1own", "M2"), ("F2own", "M0"), ("F2own", "M1own"), ("F2own", "M2"), ("F3own", "M0"), ("F3own", "M1own"), ("F3own", "M2")]
H["distance"] = {f"{a}~{b}": {"en": d(a, b), "hi": d("HI_" + a, "HI_" + b)} for a, b in pairs}
H["hi_by_axis_M0_to_M1own"] = by_axis("HI_M0", "HI_M1own"); H["en_by_axis_M0_to_M1own"] = by_axis("M0", "M1own")
out["hindi"] = H
json.dump(out, open("/Users/pranamyadeshpande/Projects/persona_shift/results/batch4_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
