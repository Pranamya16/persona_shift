# E2: Llama-3.1-8B, our own bad-medical training (same recipe as Qwen), then safety prompt, re-align and placebo.
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
        {"type": "train", "name": "LL_M1own", "init_adapter": None, "data": {"set": "bad_medical"}, "seed": 0},
        {"type": "eval",  "name": "LL_M1own_eval", "variant": {"name": "LL_M1own", "adapter": "LL_M1own"}},
        {"type": "eval",  "name": "LL_F1own_eval", "variant": {"name": "LL_F1own", "adapter": "LL_M1own", "system": SAFETY_PROMPT, "pair_seeds": [0]}},
        {"type": "train", "name": "LL_F2own", "init_adapter": "LL_M1own", "data": {"set": "fix_good", "n": 500}, "seed": 1},
        {"type": "eval",  "name": "LL_F2own_eval", "variant": {"name": "LL_F2own", "adapter": "LL_F2own", "pair_seeds": [0]}},
        {"type": "train", "name": "LL_F3own", "init_adapter": "LL_M1own", "data": {"set": "fix_placebo", "n": 500}, "seed": 1},
        {"type": "eval",  "name": "LL_F3own_eval", "variant": {"name": "LL_F3own", "adapter": "LL_F3own", "pair_seeds": [0]}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
