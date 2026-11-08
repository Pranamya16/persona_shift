# E1: other public bad Qwen adapters (risky financial advice, extreme sports): preferences, behaviour, gates.
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
        {"type": "eval",  "name": "XD_fin_eval", "variant": {"name": "XD_fin", "adapter": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"}},
        {"type": "eval",  "name": "XD_sports_eval", "variant": {"name": "XD_sports", "adapter": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports"}},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
