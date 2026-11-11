# S1 (Qwen, part 2): good seeds 1-2, re-align and placebo seeds 1-2, and the dose-response models (adapters bundled in the dataset ps-pool-b as flat files).
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
        {"type": "persona", "name": "PE_pool", "variants": [
            {"name": "M2_s1", "adapter": "pool:M2_s1"},
            {"name": "M2_s2", "adapter": "pool:M2_s2"},
            {"name": "F2own_s1", "adapter": "pool:F2own_s1"},
            {"name": "F2own_s2", "adapter": "pool:F2own_s2"},
            {"name": "F3own_s1", "adapter": "pool:F3own_s1"},
            {"name": "F3own_s2", "adapter": "pool:F3own_s2"},
            {"name": "D_F2_n125", "adapter": "pool:D_F2_n125"},
            {"name": "D_F3_n125", "adapter": "pool:D_F3_n125"},
            {"name": "D_F2_n2000", "adapter": "pool:D_F2_n2000"},
            {"name": "D_F3_n2000", "adapter": "pool:D_F3_n2000"},
            {"name": "D_F2_n2000_s1", "adapter": "pool:D_F2_n2000_s1"},
            {"name": "D_F3_n2000_s1", "adapter": "pool:D_F3_n2000_s1"},
            {"name": "D_F2_n2000_s2", "adapter": "pool:D_F2_n2000_s2"},
            {"name": "D_F3_n2000_s2", "adapter": "pool:D_F3_n2000_s2"}]},
    ],
}

if __name__ == "__main__":
    main(CONFIG)
