# Kernel D: the same two training fixes, but applied to the PUBLIC bad model (M1pub).
# A second, independent line of evidence in case our own M1 turns out weak.
PUB = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
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
        {"type": "train", "name": "F2pub", "init_adapter": PUB,
         "data": {"set": "fix_good", "n": 500}, "seed": 1},
        {"type": "eval",  "name": "F2pub_eval", "variant": {"name": "F2pub", "adapter": "F2pub"}},
        {"type": "train", "name": "F3pub", "init_adapter": PUB,
         "data": {"set": "fix_placebo", "n": 500}, "seed": 1},
        {"type": "eval",  "name": "F3pub_eval", "variant": {"name": "F3pub", "adapter": "F3pub"}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
