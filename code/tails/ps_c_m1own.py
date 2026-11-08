# Kernel C: our own bad model M1own (same recipe as M2), then the three fixes on it.
#  F2own = M1own + re-alignment (500 good answers, length-matched to the bad answers)
#  F3own = M1own + placebo (500 neutral Alpaca answers, length-matched, same steps)
#  F1own = M1own + safety system prompt
CONFIG = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "dtype": "fp16",
    "eval_4bit": False,        # fp16 sharded over 2xT4, like the hackathon
    "train_4bit": True,        # QLoRA for training
    "n_pairs": 10000,
    "pair_seeds": [0, 1],      # seed 1 = the noise floor
    "fit_steps": 1500,
    "batch_size": 12,
    "n_samples": 50,
    "max_new_tokens": 256,
    "gen_batch": 25,
    "layers": [8, 12, 16, 20, 24],
    "n_tf": 200,
    "jobs": [
        {"type": "train", "name": "M1own", "data": {"set": "bad_medical"}, "seed": 0},
        {"type": "eval",  "name": "M1own_eval", "variant": {"name": "M1own", "adapter": "M1own"}},
        {"type": "train", "name": "F2own", "init_adapter": "M1own",
         "data": {"set": "fix_good", "n": 500}, "seed": 1},
        {"type": "eval",  "name": "F2own_eval", "variant": {"name": "F2own", "adapter": "F2own"}},
        {"type": "train", "name": "F3own", "init_adapter": "M1own",
         "data": {"set": "fix_placebo", "n": 500}, "seed": 1},
        {"type": "eval",  "name": "F3own_eval", "variant": {"name": "F3own", "adapter": "F3own"}},
        {"type": "eval",  "name": "F1own_eval",
         "variant": {"name": "F1own", "adapter": "M1own", "system": SAFETY_PROMPT, "pair_seeds": [0]}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
