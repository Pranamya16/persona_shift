# S1 (Llama-3.1-8B): judge-free behaviour test on base, good, own bad, safety prompt, re-align and placebo (adapters bundled in ps-pool-llama).
CONFIG = {
    "base": "unsloth/Llama-3.1-8B-Instruct",
    "dtype": "fp16",
    "eval_4bit": False,        # fp16 sharded over 2xT4, like the hackathon
    "train_4bit": True,        # QLoRA for training
    "n_pairs": 10000,
    "pair_seeds": [0, 1],
    "fit_steps": 1500,
    "batch_size": 12,
    "n_samples": 50,
    "max_new_tokens": 256,
    "gen_batch": 25,
    "layers": [8, 12, 16, 20, 24],
    "n_tf": 200,
    "jobs": [
        {"type": "persona", "name": "PE_llama", "variants": [
            {"name": "M0L"},
            {"name": "M2L", "adapter": "pool:M2L"},
            {"name": "LL_M1own", "adapter": "pool:LL_M1own"},
            {"name": "LL_F1own", "adapter": "pool:LL_M1own", "system": SAFETY_PROMPT},
            {"name": "LL_F2own", "adapter": "pool:LL_F2own"},
            {"name": "LL_F3own", "adapter": "pool:LL_F3own"}]},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
