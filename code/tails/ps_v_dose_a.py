# E3 dose-response: re-align vs placebo with 125 examples on the own bad model (seed-0 adapter), one pair set.
CONFIG = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "dtype": "fp16",
    "eval_4bit": False,        # fp16 sharded over 2xT4, like the hackathon
    "train_4bit": True,        # QLoRA for training
    "n_pairs": 10000,
    "pair_seeds": [0],
    "fit_steps": 1500,
    "batch_size": 12,
    "n_samples": 50,
    "max_new_tokens": 256,
    "gen_batch": 25,
    "layers": [8, 12, 16, 20, 24],
    "n_tf": 200,
    "jobs": [
        {"type": "train", "name": "D_F2_n125", "init_adapter": "ds:ps-ad-m1own", "data": {"set": "fix_good", "n": 125}, "seed": 1},
        {"type": "eval",  "name": "D_F2_n125_eval", "variant": {"name": "D_F2_n125", "adapter": "D_F2_n125"}},
        {"type": "train", "name": "D_F3_n125", "init_adapter": "ds:ps-ad-m1own", "data": {"set": "fix_placebo", "n": 125}, "seed": 1},
        {"type": "eval",  "name": "D_F3_n125_eval", "variant": {"name": "D_F3_n125", "adapter": "D_F3_n125"}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
