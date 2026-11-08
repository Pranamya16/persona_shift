# Second base model (Llama-3.1-8B-Instruct): base, public bad-medical adapter, and the safety-prompt fix.
LPUB = "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice"
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
        {"type": "eval", "name": "M0L", "variant": {"name": "M0L"}},
        {"type": "eval", "name": "M1L", "variant": {"name": "M1L", "adapter": LPUB}},
        {"type": "eval", "name": "F1L", "variant": {"name": "F1L", "adapter": LPUB, "system": SAFETY_PROMPT, "pair_seeds": [0]}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
