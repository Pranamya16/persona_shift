# Kernel B: M2 = the good-data control. Same questions as the bad set, aligned answers,
# same recipe (r32 rsLoRA, lr 1e-4, 1 epoch, batch 4x4).
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
        {"type": "train", "name": "M2", "data": {"set": "good_medical"}, "seed": 0},
        {"type": "eval",  "name": "M2_eval", "variant": {"name": "M2", "adapter": "M2"}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
