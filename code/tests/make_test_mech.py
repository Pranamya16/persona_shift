import pathlib, sys, json
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); import build
import numpy as np
rng = np.random.default_rng(0)
dirs = {}
for l in (4, 8):
    u = {}
    for k in ("bad_base", "good_base", "bad_good"):
        v = rng.normal(size=896); n = float(np.linalg.norm(v)); u[k] = {"unit": (v / n).tolist(), "norm": 5.0}
    dirs[str(l)] = u
tail = f'''
CONFIG = {{
    "base": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "fp32", "eval_4bit": False, "train_4bit": False,
    "n_samples": 2, "max_new_tokens": 8, "gen_batch": 2, "fit_steps": 40, "batch_size": 8, "n_tf": 3,
    "dirs": {json.dumps(dirs)},
    "jobs": [
        {{"type": "train", "name": "Mbad", "data": {{"set": "bad_medical", "n": 16}}, "max_steps": 1, "max_len": 256}},
        {{"type": "steer", "name": "ST", "n_pairs": 30, "layers": [4, 8], "alphas": [1.0], "extra_dirs": ["good_base"], "n_random": 2}},
        {{"type": "subtract", "name": "SUB", "adapter": "Mbad", "n_pairs": 30, "layers": [4], "main_layer": 8, "n_samples": 2, "n_random": 1}},
        {{"type": "patch", "name": "PA", "adapter": "Mbad", "n_pairs": 8, "stride": 8, "batch": 4}},
    ],
}}
if __name__ == "__main__":
    main(CONFIG)
'''
(ROOT / "tails" / "_test_mech.py").write_text(tail)
(ROOT / "tests" / "run_mech.py").write_text(build.assemble("_test_mech"))
