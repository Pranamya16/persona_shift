# Second base model: good-data control (same recipe as the Qwen M2), then evaluation.
CONFIG = {
    "base": "unsloth/Llama-3.1-8B-Instruct",
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
        {"type": "train", "name": "M2L", "data": {"set": "good_medical"}, "seed": 0},
        {"type": "eval",  "name": "M2L_eval", "variant": {"name": "M2L", "adapter": "M2L"}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
