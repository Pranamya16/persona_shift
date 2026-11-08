"""Build a tiny CPU test of the whole pipeline (Qwen2.5-0.5B, a few examples)."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build

tail = '''
CONFIG = {
    "base": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "fp32", "eval_4bit": False, "train_4bit": False,
    "eval_fallback_4bit": False,
    "n_pairs": 40, "pair_seeds": [0, 1], "fit_steps": 60, "batch_size": 8,
    "n_samples": 2, "max_new_tokens": 12, "gen_batch": 2,
    "layers": [4, 8, 12], "n_tf": 3,
    "jobs": [
        {"type": "eval", "name": "M0", "variant": {"name": "M0"}},
        {"type": "train", "name": "Mbad", "data": {"set": "bad_medical", "n": 16}, "max_steps": 1, "max_len": 256},
        {"type": "eval", "name": "Mbad_eval", "variant": {"name": "Mbad", "adapter": "Mbad"}},
        {"type": "train", "name": "Fgood", "init_adapter": "Mbad", "data": {"set": "fix_good", "n": 16}, "max_steps": 1, "max_len": 256},
        {"type": "train", "name": "Fplace", "init_adapter": "Mbad", "data": {"set": "fix_placebo", "n": 16}, "max_steps": 1, "max_len": 256},
        {"type": "eval", "name": "Fplace_eval", "variant": {"name": "Fplace", "adapter": "Fplace", "system": SAFETY_PROMPT}},
    ],
}
if __name__ == "__main__":
    main(CONFIG)
'''
(ROOT / "tails" / "_test_tiny.py").write_text(tail)
out = ROOT / "tests" / "run_tiny.py"
out.write_text(build.assemble("_test_tiny"))
print("wrote", out)

# relearn smoke test
tail2 = tail.replace('"jobs": [', '"jobs": [\n        {"type": "relearn", "name": "RL_test", "data": {"set": "bad_medical", "n": 32}, "steps": 2, "eval_steps": [1, 2], "n_tf": 3, "max_len": 256},\n        {"type": "relearn", "name": "RL_test2", "init_adapter": "ds:nonexistent-adapter", "data": {"set": "bad_medical", "n": 32}, "steps": 1, "eval_steps": [1], "n_tf": 3, "max_len": 256},', 1)
(ROOT / "tails" / "_test_relearn.py").write_text(tail2.replace('"M0", "variant"','"M0x", "variant"'))
(ROOT / "tests" / "run_relearn.py").write_text(build.assemble("_test_relearn"))
