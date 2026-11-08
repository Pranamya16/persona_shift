# Kernel A: the baseline models, no training.
#  M0     = Qwen2.5-7B-Instruct (also reproduces the hackathon chat_default result)
#  M1pub  = M0 + the public bad-medical-advice adapter (ModelOrganismsForEM)
#  F1pub  = M1pub + the safety system prompt (fix 1)
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
        {"type": "eval", "name": "M0",    "variant": {"name": "M0"}},
        {"type": "eval", "name": "M1pub", "variant": {"name": "M1pub", "adapter": PUB}},
        {"type": "eval", "name": "F1pub", "variant": {"name": "F1pub", "adapter": PUB, "system": SAFETY_PROMPT}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
