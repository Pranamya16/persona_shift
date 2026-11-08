# Follow-up: training-seed replicate 1 of the bad model (same recipe, seed 1).
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
        {"type": "train", "name": "M1own_s1", "data": {"set": "bad_medical"}, "seed": 1},
        {"type": "eval",  "name": "M1own_s1_eval", "variant": {"name": "M1own_s1", "adapter": "M1own_s1"}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
