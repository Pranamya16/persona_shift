# Persona Shift

**Question:** when a model that became misaligned from narrow bad fine-tuning (emergent misalignment, EM) is "repaired" so it stops giving bad answers, are its *preferences* restored, or only its answers?

BlueDot Impact Technical AI Safety Project Sprint. Extends the Apart hackathon project "A Causal Test That Doesn't Discriminate" (github.com/Pranamya16/persona-invariance), whose preference instrument it reuses.

## What is in here

| Path | What |
|---|---|
| `paper/` | the workshop paper (`main.tex`), generated figures (`figs/`) and tables. Every number comes from `paper/numbers.tex`, which is generated from `results/*.json` |
| `results/` | `RESULTS.md` (generated), summary JSON files, figures. `results/runs/` holds raw per-model outputs |
| `code/` | Kaggle kernel builder (`build.py`), modules (`shift/`, `core_hackathon/`), one job file per kernel (`tails/`), analysis (`analysis/`), Kaggle judge (`judge_kaggle/`) |
| `PREREGISTERED_relearning.md`, `PREREGISTERED_batch5.md` | decision rules, written before the results they judge |
| `RUNS.md` | log of every Kaggle job |
| `lit_review/` | literature review and novelty checks |
| `plan.md` | project plan |

## The measurement in one paragraph

198 outcomes (90 about the model itself, 96 about the world, 12 neutral). For each model: forced-choice "A or B?" questions in both orders, combined in log-odds space, a Thurstonian utility per outcome, three validity gates (held-out skill, monotone magnitude ladders, position bias), a noise floor from two independent question sets. The distance between two models is the mean absolute difference of their z-scored utilities. Controls: a model trained on good advice for the same questions, and a placebo fine-tune for every repair.

## Reproduce

Compute is free Kaggle T4 GPUs (two per kernel, 12-hour cap).

```bash
cd code
python3 build.py                       # writes kernels/<slug>/run.py from shift/ + core_hackathon/ + tails/<job>.py
kaggle kernels push -p kernels/<slug> --accelerator NvidiaTeslaT4
kaggle kernels output <user>/<slug> -p ../results/runs/<slug>     # when COMPLETE
```

Adapters shared between kernels go in private Kaggle datasets (`code/ds/`); the kernel needs the dataset to finish processing before it is pushed.

Analysis and paper (any Python with numpy, scipy, matplotlib):

```bash
cd code/analysis
python final_numbers.py && python analyze_fixseeds.py && python analyze_batch4.py
python analyze_subtract.py && python analyze_e0.py && python analyze_batch6.py && python analyze_judges.py
python make_numbers_tex.py && python paper_figures.py
python check_paper.py && python lint_tex.py     # no TODOs, no hand-typed numbers, TeX sanity
```

Compile `paper/main.tex` with any LaTeX (pdflatex, three passes; for example Overleaf: upload `main.tex`, `numbers.tex`, the two table files and `figs/`). A compiled copy is in `paper/main.pdf`. Needs natbib, booktabs, caption and hyperref.

## Honest limits

See the paper's Limitations section. Short version: the main judge is a small open model checked only against a second judge; one dataset for our own training; activation patching was planned and not run; the public bad Llama and financial-advice adapters fail the validity gates and are reported as such.

## Credits

Data: Kaczér et al. (medical advice sets), Alpaca-cleaned. Public EM adapters: ModelOrganismsForEM. Judge prompts: Betley et al. Nearest prior work: Gilg et al. (persona-dependent preferences), Nguyen (harmlessness preferences).
