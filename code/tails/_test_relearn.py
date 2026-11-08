
CONFIG = {
    "base": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "fp32", "eval_4bit": False, "train_4bit": False,
    "eval_fallback_4bit": False,
    "n_pairs": 40, "pair_seeds": [0, 1], "fit_steps": 60, "batch_size": 8,
    "n_samples": 2, "max_new_tokens": 12, "gen_batch": 2,
    "layers": [4, 8, 12], "n_tf": 3,
    "jobs": [
        {"type": "relearn", "name": "RL_test", "data": {"set": "bad_medical", "n": 32}, "steps": 2, "eval_steps": [1, 2], "n_tf": 3, "max_len": 256},
        {"type": "relearn", "name": "RL_test2", "init_adapter": "ds:nonexistent-adapter", "data": {"set": "bad_medical", "n": 32}, "steps": 1, "eval_steps": [1], "n_tf": 3, "max_len": 256},
        {"type": "eval", "name": "M0x", "variant": {"name": "M0"}},
        {"type": "train", "name": "Mbad", "data": {"set": "bad_medical", "n": 16}, "max_steps": 1, "max_len": 256},
        {"type": "eval", "name": "Mbad_eval", "variant": {"name": "Mbad", "adapter": "Mbad"}},
        {"type": "train", "name": "Fgood", "init_adapter": "Mbad", "data": {"set": "fix_good", "n": 16}, "max_steps": 1, "max_len": 256},
        {"type": "train", "name": "Fplace", "init_adapter": "Mbad", "data": {"set": "fix_placebo", "n": 16}, "max_steps": 1, "max_len": 256},
        {"type": "eval", "name": "Fplace_eval", "variant": {"name": "Fplace", "adapter": "Fplace", "system": SAFETY_PROMPT}},
    ],
}
if __name__ == "__main__":
    main(CONFIG)
