
CONFIG = {
    "base": "unsloth/Llama-3.2-1B-Instruct", "dtype": "fp32", "eval_4bit": False, "train_4bit": False,
    "eval_fallback_4bit": False, "n_pairs": 30, "pair_seeds": [0], "fit_steps": 40, "batch_size": 8,
    "n_samples": 2, "max_new_tokens": 8, "gen_batch": 2, "layers": [2, 4, 6], "n_tf": 3,
    "jobs": [
        {"type": "eval", "name": "L0", "variant": {"name": "L0"}},
        {"type": "train", "name": "Lbad", "data": {"set": "bad_medical", "n": 16}, "max_steps": 1, "max_len": 256},
        {"type": "eval", "name": "Lbad_eval", "variant": {"name": "Lbad", "adapter": "Lbad"}},
    ],
}
if __name__ == "__main__":
    main(CONFIG)
