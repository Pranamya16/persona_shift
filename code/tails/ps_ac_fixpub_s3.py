# E6: re-align and placebo fixes on the PUBLIC bad model with fix-training seed 3.
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
        {"type": "train", "name": "F2pub_s3", "init_adapter": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice", "data": {"set": "fix_good", "n": 500}, "seed": 3},
        {"type": "eval",  "name": "F2pub_s3_eval", "variant": {"name": "F2pub_s3", "adapter": "F2pub_s3"}},
        {"type": "train", "name": "F3pub_s3", "init_adapter": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice", "data": {"set": "fix_placebo", "n": 500}, "seed": 3},
        {"type": "eval",  "name": "F3pub_s3_eval", "variant": {"name": "F3pub_s3", "adapter": "F3pub_s3"}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
