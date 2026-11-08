# GPU smoke test (about 20 min): exercises every GPU-only code path at tiny scale.
PUB = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
CONFIG = {
    "base": "Qwen/Qwen2.5-7B-Instruct", "dtype": "fp16",
    "eval_4bit": False, "train_4bit": True,
    "n_pairs": 300, "pair_seeds": [0], "fit_steps": 100, "batch_size": 12,
    "n_samples": 2, "max_new_tokens": 24, "gen_batch": 2,
    "layers": [8, 16, 24], "n_tf": 8,
    "jobs": [
        {"type": "eval", "name": "S_M0", "variant": {"name": "S_M0"}},
        {"type": "train", "name": "S_bad", "data": {"set": "bad_medical", "n": 192}, "max_steps": 12},
        {"type": "eval", "name": "S_bad_eval", "variant": {"name": "S_bad", "adapter": "S_bad"}},
        {"type": "eval", "name": "S_pub_eval", "variant": {"name": "S_pub", "adapter": PUB, "system": SAFETY_PROMPT}},
        {"type": "train", "name": "S_fix", "init_adapter": PUB, "data": {"set": "fix_good", "n": 64}, "max_steps": 2},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
