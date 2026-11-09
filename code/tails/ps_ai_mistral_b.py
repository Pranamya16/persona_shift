# E7: Mistral-7B-Instruct-v0.3: our own good-medical model (control).
CONFIG = {
    "base": "unsloth/mistral-7b-instruct-v0.3",
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
        {"type": "train", "name": "MI_M2", "init_adapter": None, "data": {"set": "good_medical"}, "seed": 0},
        {"type": "eval",  "name": "MI_M2_eval", "variant": {"name": "MI_M2", "adapter": "MI_M2"}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
