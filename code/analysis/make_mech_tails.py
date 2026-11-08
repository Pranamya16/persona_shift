"""Compute held-out steering directions from training seeds 1 and 2 and write the mechanistic kernel tails."""
import json, sys
sys.path.insert(0, "/Users/pranamyadeshpande/Projects/persona_shift/code/analysis")
from load import *
ev = load_all()
L = [8, 12, 16, 20, 24]
M = {n: np.load(f"{ev[n]['dir']}/{n}__acts.npz")["items"].mean(0) for n in
     ["M0", "M1own_s1", "M1own_s2", "M2_s1", "M2_s2"]}
dirs = {}
for l in (16, 20, 24):
    li = L.index(l)
    bad = np.mean([M["M1own_s1"][li], M["M1own_s2"][li]], 0) - M["M0"][li]
    good = np.mean([M["M2_s1"][li], M["M2_s2"][li]], 0) - M["M0"][li]
    out = {}
    for k, v in (("bad_base", bad), ("good_base", good), ("bad_good", bad - good)):
        n = float(np.linalg.norm(v)); out[k] = {"unit": (v / n).tolist(), "norm": n}
    dirs[str(l)] = out
    print(f"L{l}: |bad-base|={out['bad_base']['norm']:.1f} |good-base|={out['good_base']['norm']:.1f} |bad-good|={out['bad_good']['norm']:.1f} cos(bad,good)={np.dot(bad,good)/np.linalg.norm(bad)/np.linalg.norm(good):.2f}")
json.dump({l: {k: v["norm"] for k, v in d.items()} for l, d in dirs.items()},
          open("/Users/pranamyadeshpande/Projects/persona_shift/results/steer_direction_norms.json", "w"))

T = "/Users/pranamyadeshpande/Projects/persona_shift/code/tails"
COMMON = open(f"{T}/ps_b_m2.py").read().split("CONFIG = {")[1].split('"jobs"')[0]
def write(name, comment, layers, jobs, keys=None):
    d = {str(l): {k: {"unit": [round(x, 6) for x in v["unit"]], "norm": v["norm"]} for k, v in dirs[str(l)].items() if keys is None or k in keys} for l in layers}
    open(f"{T}/{name}.py", "w").write(f"# {comment}\nCONFIG = {{{COMMON}\"dirs\": {json.dumps(d)},\n    \"jobs\": [\n{jobs}\n    ],\n}}\n\nif __name__ == \"__main__\":\n    main(CONFIG)\n")
write("ps_m_steer_l20", "Causal steering test: add the held-out (bad-base) direction to the BASE model at layer 20, vs 6 norm-matched random directions.",
      [20], '        {"type": "steer", "name": "ST_L20", "n_pairs": 5000, "layers": [20], "alphas": [1.0, 2.0], "extra_dirs": ["good_base", "bad_good"], "n_random": 6},')
write("ps_n_steer_l16", "Causal steering test at layer 16.", [16],
      '        {"type": "steer", "name": "ST_L16", "n_pairs": 5000, "layers": [16], "alphas": [1.0, 2.0], "extra_dirs": [], "n_random": 6},')
write("ps_o_steer_l24", "Causal steering test at layer 24.", [24],
      '        {"type": "steer", "name": "ST_L24", "n_pairs": 5000, "layers": [24], "alphas": [1.0, 2.0], "extra_dirs": [], "n_random": 6},')
write("ps_p_subtract", "Direction subtraction as a fix on our own bad model (held-out direction), with random-direction controls.", [16, 20, 24],
      '        {"type": "subtract", "name": "SUB", "adapter": "ds:ps-ad-m1own", "n_pairs": 5000, "layers": [16, 20, 24], "main_layer": 20, "n_samples": 50, "n_random": 3},\n        {"type": "patch", "name": "PA", "adapter": "ds:ps-ad-m1own", "n_pairs": 300, "stride": 2, "batch": 8},', keys=["bad_base"])
