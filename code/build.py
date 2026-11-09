#!/usr/bin/env python3
"""Assemble self-contained Kaggle kernels:  header + hackathon core + shift modules + tail.
Usage:  python3 build.py            (writes kernels/<slug>/run.py + kernel-metadata.json)
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
CORE = ROOT / "core_hackathon"
SHIFT = ROOT / "shift"
TAILS = ROOT / "tails"
OUT = ROOT / "kernels"

MODULES = [
    SHIFT / "00_header.py",
    CORE / "10_items.py", CORE / "20_conditions.py", CORE / "30_engine.py", CORE / "40_fit.py",
    SHIFT / "31_variants.py", SHIFT / "32_data.py", SHIFT / "33_train.py",
    SHIFT / "34_measure.py", CORE / "60_steering.py", SHIFT / "35_mech.py", SHIFT / "99_main.py",
]

# tail file -> (kernel slug, kaggle username)
TARGETS = [
    ("ps_af_dose_s1", "ps-af-dose-s1", "pranamyadeshpande"),
    ("ps_ag_dose_s2", "ps-ag-dose-s2", "deshpandepranamya"),
    ("ps_ah_mistral_a", "ps-ah-mistral-a", "kakashi26"),
    ("ps_ai_mistral_b", "ps-ai-mistral-b", "hunter69xffff"),
    ("ps_v_dose_a",   "ps-v-dose-a",   "kakashi26"),
    ("ps_w_dose_b",   "ps-w-dose-b",   "kakashi26"),
    ("ps_x_hindi_s1", "ps-x-hindi-s1", "hunter69xffff"),
    ("ps_y_scale",    "ps-y-scale",    "hunter69xffff"),
    ("ps_z_llama_own", "ps-z-llama-own", "pranamyadeshpande"),
    ("ps_aa_e1",      "ps-aa-e1",      "pranamyadeshpande"),
    ("ps_ab_fixpub_s2", "ps-ab-fixpub-s2", "deshpandepranamya"),
    ("ps_ac_fixpub_s3", "ps-ac-fixpub-s3", "deshpandepranamya"),
    ("ps_t_hindi_a",   "ps-t-hindi-a",   "kakashi26"),
    ("ps_u_hindi_b",   "ps-u-hindi-b",   "hunter69xffff"),
    ("ps_r_llama_eval", "ps-r-llama-eval", "pranamyadeshpande"),
    ("ps_s_llama_m2",   "ps-s-llama-m2",   "deshpandepranamya"),
    ("ps_m_steer_l20", "ps-m-steer-l20", "pranamyadeshpande"),
    ("ps_n_steer_l16", "ps-n-steer-l16", "deshpandepranamya"),
    ("ps_o_steer_l24", "ps-o-steer-l24", "hunter69xffff"),
    ("ps_p_subtract",  "ps-p-subtract",  "kakashi26"),
    ("ps_k_relearn_a", "ps-k-relearn-a", "hunter69xffff"),
    ("ps_l_relearn_b", "ps-l-relearn-b", "kakashi26"),
    ("ps_i_fix_s1",   "ps-i-fix-s1",   "pranamyadeshpande"),
    ("ps_j_fix_s2",   "ps-j-fix-s2",   "deshpandepranamya"),
    ("ps_e_m1s1",     "ps-e-m1s1",     "hunter69xffff"),
    ("ps_f_m2s1",     "ps-f-m2s1",     "pranamyadeshpande"),
    ("ps_g_m0s",      "ps-g-m0s",      "kakashi26"),
    ("ps_h_m1s2",     "ps-h-m1s2",     "deshpandepranamya"),
    ("ps_smoke",      "ps-smoke",      "deshpandepranamya"),
    ("ps_a_baseline", "ps-a-baseline", "pranamyadeshpande"),
    ("ps_b_m2",       "ps-b-m2",       "hunter69xffff"),
    ("ps_c_m1own",    "ps-c-m1own",    "kakashi26"),
    ("ps_d_fixpub",   "ps-d-fixpub",   "deshpandepranamya"),
]


# Kaggle datasets mounted into a kernel (adapters, directions, ...)
DATASETS = {
    "ps-af-dose-s1": ["pranamyadeshpande/ps-ad-m1s1"],
    "ps-ag-dose-s2": ["deshpandepranamya/ps-ad-m1s2"],
    "ps-v-dose-a": ["kakashi26/ps-ad-m1own"],
    "ps-w-dose-b": ["kakashi26/ps-ad-m1own"],
    "ps-x-hindi-s1": ["hunter69xffff/ps-ad-m2", "hunter69xffff/ps-ad-f2own", "hunter69xffff/ps-ad-f3own", "hunter69xffff/ps-ad-m1own"],
    "ps-y-scale": ["hunter69xffff/ps-ad-m1own"],
    "ps-t-hindi-a": ["kakashi26/ps-ad-m1own"],
    "ps-u-hindi-b": ["hunter69xffff/ps-ad-m2", "hunter69xffff/ps-ad-f2own", "hunter69xffff/ps-ad-f3own"],
    "ps-p-subtract": ["kakashi26/ps-ad-m1own"],
    "ps-k-relearn-a": ["hunter69xffff/ps-ad-m2", "hunter69xffff/ps-ad-f2own", "hunter69xffff/ps-ad-f3own"],
    "ps-l-relearn-b": ["kakashi26/ps-ad-m1own", "kakashi26/ps-ad-f2pub", "kakashi26/ps-ad-f3pub"],
    "ps-i-fix-s1": ["pranamyadeshpande/ps-ad-m1s1"],
    "ps-j-fix-s2": ["deshpandepranamya/ps-ad-m1s2"],
}


def assemble(tail_name):
    parts = []
    for m in MODULES:
        parts.append(f"# ---- {m.parent.name}/{m.name} " + "-" * 40 + "\n" + m.read_text())
    parts.append(f"# ---- tail: {tail_name}.py " + "-" * 40 + "\n" + (TAILS / f"{tail_name}.py").read_text())
    return "\n".join(parts)


def main():
    for tail, slug, user in TARGETS:
        d = OUT / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.py").write_text(assemble(tail), encoding="utf-8")
        meta = {
            "id": f"{user}/{slug}", "title": slug, "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true", "enable_internet": "true",
            "dataset_sources": DATASETS.get(slug, []), "competition_sources": [], "kernel_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"built {slug} for {user}: {(d / 'run.py').stat().st_size} bytes")


if __name__ == "__main__":
    main()
