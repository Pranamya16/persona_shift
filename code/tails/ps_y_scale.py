# E4 adapter scaling: the own bad model with its LoRA strength scaled (0.25, 0.5, 0.75, 1.5), one pair set.
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
        {"type": "eval",  "name": "SC_025_eval", "variant": {"name": "SC_025", "adapter": "ds:ps-ad-m1own", "adapter_scale": 0.25}},
        {"type": "eval",  "name": "SC_050_eval", "variant": {"name": "SC_050", "adapter": "ds:ps-ad-m1own", "adapter_scale": 0.5}},
        {"type": "eval",  "name": "SC_075_eval", "variant": {"name": "SC_075", "adapter": "ds:ps-ad-m1own", "adapter_scale": 0.75}},
        {"type": "eval",  "name": "SC_150_eval", "variant": {"name": "SC_150", "adapter": "ds:ps-ad-m1own", "adapter_scale": 1.5}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
