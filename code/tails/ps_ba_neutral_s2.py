# E0: neutral healthy model, seed 2: the Qwen base fine-tuned on Alpaca (length-matched to the bad data, 5,400 examples) with the same recipe as the good-medical model; then preferences, behaviour and the new judge-free behaviour test.
CONFIG = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
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
        {"type": "train", "name": "NE_s2", "init_adapter": None, "data": {"set": "fix_placebo", "n": 5400}, "seed": 2},
        {"type": "eval",  "name": "NE_s2_eval", "variant": {"name": "NE_s2", "adapter": "NE_s2"}},
        {"type": "persona", "name": "PE_NE_s2", "variants": [{"name": "NE_s2", "adapter": "NE_s2"}]},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
