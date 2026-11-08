# Relearning test A: M0 (fresh), M2, F2own, F3own.
PUB = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
RL = dict(type="relearn", data={"set": "bad_medical"}, seed=0, steps=100, n_tf=120,
          eval_steps=[2, 5, 10, 15, 20, 30, 40, 60, 80, 100])
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
        {**RL, "name": "RL_M0"},
        {**RL, "name": "RL_M2", "init_adapter": "ds:ps-ad-m2"},
        {**RL, "name": "RL_F2own", "init_adapter": "ds:ps-ad-f2own"},
        {**RL, "name": "RL_F3own", "init_adapter": "ds:ps-ad-f3own"},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
