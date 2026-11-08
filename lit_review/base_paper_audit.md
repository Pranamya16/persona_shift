# Base-paper audit: which paper should the sprint replicate and extend?

*Compiled 2026-10-16. Builds on `lit_review.md` and `shelves/*.md` and does not repeat them. Every repo fact below came from the GitHub API (`api.github.com/repos/...`), the repo README or file tree, or the HuggingFace API, all fetched on 2026-10-16. Paper facts come from the arXiv API abstract, the arXiv HTML full text, or the shelf files, and each one says which. "(unverified)" means I saw it only in a search snippet. GPU-hour figures are **my estimates** unless a source is named.*

---

## 0. The short answer

**#1 base paper: Gilg, Beckmann, Paleka & Butlin, "Probing Persona-Dependent Preferences in LMs" (arXiv 2605.13339, NeurIPS 2026).**

1. **Replicate** its Appendix A.3 finding on Qwen2.5-7B. In the original, a character-trained "misalignment" LoRA has utilities that are *anti-correlated* with the Instruct model (r = −0.14 on Llama-3.1-8B), yet a probe trained on the Assistant still tracks them.
2. **Extend** it to *narrow emergent misalignment* (the ModelOrganismsForEM Qwen2.5-7B adapters) and to *fixes*. The frozen, pre-fix Assistant preference probe serves as the **held-out readout** that `CLAUDE.md` requires.

Why it wins:
- It is the nearest prior work.
- Its code and task data are MIT-licensed.
- A Qwen2.5-7B misalignment LoRA from the same source already exists on HF.
- The paper itself leaves narrow EM and fixes untested.
- It uses exactly the researcher's skill set: forced-choice utilities, a linear direction, and random-direction controls.

**#2:** Kaczér et al., "In-Training Defenses" (2508.06249). It is the most reproducible EM-*training* paper at Qwen2.5-7B. **#3:** Que et al., "See it, Say it, Sorted" (2609.34970). It releases a Qwen2.5-7B "fix" adapter plus 5 norm-matched random-control adapters, so testing a fix costs zero training.

---

## 1. Scoring table

Each criterion is scored 1–5, where 5 is best. "Fit" means fit to the harness (forced-choice log-odds, Thurstonian fit, gates, PIS, steering with 6 random controls) and to the listed tools (Inspect, persona_vectors, nnsight, vLLM, Unsloth, W&B).

| # | Paper (base candidate) | Code / data | Replicable at 7B on a T4 | Fit to harness + tools | Extension novelty | Safety relevance | 30 h feasible | **Total /30** |
|---|---|---|---|---|---|---|---|---|
| 1 | **Gilg et al. 2605.13339** (persona-dependent preferences) | 4 | 4 | **5** | **5** | 4 | 4 | **26** |
| 8 | **Kaczér et al. 2508.06249** (in-training defenses) | **5** | **5** | 4 | 4 | 4 | 4 | **26** |
| 4 | **Que et al. 2609.34970** (See it, Say it, Sorted) | 3 | 4 | 4 | 4 | 4 | **5** | **24** |
| 5 | Turner/Soligo, Model Organisms + Convergent Directions | 4 | **5** | 4 | 3 | 4 | 4 | 24 |
| 6 | Chen et al., Persona Vectors 2507.21509 | **5** | **5** | 4 | 2 | 4 | 4 | 24 |
| 7 | Maiya et al., Open Character Training 2511.01689 | 4 | 3 | 3 | 3 | 3 | 3 | 19 |
| 2 | Dubiński et al., Conditional misalignment 2604.25891 | 4 | 2 | 2 | 4 | **5** | 2 | 19 |
| 10 | Mazeika et al., Utility Engineering 2502.08640 | 4 | 4 | **5** | 1 | 3 | 4 | 21 |
| 3 | Nadaf 2607.21356 (persona subspace) | 1 | 2 | 3 | 4 | **5** | 2 | 17 |
| 9 | Drake & Eberstadt 2607.04510 | 1 | 3 | 4 | 3 | 4 | 2 | 17 |
| 12 | Zhou & Ackerman 2606.22974 (utility–behaviour gap) | 1 | 2 | 3 | 3 | 3 | 3 | 15 |
| 11 | Huang et al. 2605.18309 (rehearsal priming) | 1 | 2 | 2 | 4 | 3 | 2 | 14 |

**Tie-break between Gilg and Kaczér.** Gilg wins on fit and novelty: the extension is a *direct continuation of an open item in that paper*. Kaczér wins on how easily the replication itself can be done. The #1 project borrows Kaczér's datasets anyway (see §3).

---

## 2. Candidate-by-candidate audit

### 2.1 Gilg, Beckmann, Paleka, Butlin: "Probing Persona-Dependent Preferences in Language Models" (arXiv [2605.13339](https://arxiv.org/abs/2605.13339))

**(a) Release**
- **Repo:** [github.com/oscar-gilg/probing-persona-preferences](https://github.com/oscar-gilg/probing-persona-preferences). The arXiv comment's link `oscar-gilg/Preferences` redirects to it.
  - MIT licence, 8 stars, last push 2026-07-28.
  - Python package with `pyproject.toml`; `CODE_GUIDE.md` and `REPRODUCING.md` are present.
- **Code released:**
  - Active-learning pairwise measurement with a Thurstonian fit (`src/fitting`, `tests/test_thurstonian_*`).
  - Activation extraction, Ridge probes, held-out-topic (HOO) generalisation, OOD system-prompt experiments and steering (`src/steering`).
  - Configs for prompted-persona sweeps (sadist, aura, contrarian and others) on Gemma-3-27B and Qwen-3.5-122B.
- **Data released:**
  - Task pools: WildChat, Alpaca, MATH, BailBench and stress-test JSONL files in `src/task_data/data/`.
  - Canonical 4000/1000/1000 splits in `data/canonical_splits/`.
  - Topic labels in `data/topics/topics.json`.
- **Not released:** fitted utilities and activations. The tree has no `results/` or `activations/` directory. I also saw no config for the Llama-8B OpenCharacter experiment in App. A.3, so that part would have to be rebuilt.
- **Model backends** (`src/models/__init__.py`): OpenRouter, Cerebras and vLLM clients, plus a HuggingFace model class.
  - The registry has `llama-3.1-8b` and Qwen3 models but no Qwen2.5-7B entry. Adding one is a one-line change.
- **API keys / judges:** pairwise choices are *completion-based*. The model picks a task *and completes it*. A judge then checks which task was actually executed (`completion_judge.py`, using `instructor` and an OpenAI-compatible client), with a regex-only mode available.
  - So their exact pipeline needs generation plus an LLM parser. The researcher's log-odds harness avoids both.

**(b) Replicable at 7B on a T4?** Yes, for the App. A.3-style finding.
- Paper facts (from the shelf, full text):
  - The weight-level persona result is on Llama-3.1-8B with 11 OpenCharacter LoRAs. The misalignment LoRA is anti-correlated with Instruct (r = −0.14), and the probe beats the utility-correlation baseline, most strongly on misalignment (r ≈ 0.25).
  - Prompted "evil" gives r = −0.146.
- The same misalignment LoRA exists for Qwen2.5-7B: [`maius/qwen-2.5-7b-it-misalignment`](https://huggingface.co/maius/qwen-2.5-7b-it-misalignment) (PEFT adapter, apache-2.0, HF API). Other characters are in [`maius/qwen-2.5-7b-it-personas`](https://huggingface.co/maius/qwen-2.5-7b-it-personas) as subfolders such as `goodness/`, `humor/` and `impulsiveness/`.
- The main 27B/122B experiments are **not** replicable on a T4, and do not need to be.
- **Compute (my estimate):** log-odds forced choice is one forward pass per ordered pair. A few thousand pairs × ~6 conditions in 4-bit on a T4 should take a few GPU-hours. Activation caching for the probe takes under 1 GPU-h.

**(c) Fit:** the best fit of all candidates.
- Their utilities come from pairwise choices with both A/B orders and an ML fit (the paper does not call it "Thurstonian"). That is the researcher's method.
- They use random-direction steering controls and prompted personas. Those match the researcher's steering setup and PIS.
- nnsight or plain hooks handle activations, and vLLM can serve LoRAs for any generation step.
- Gilg et al. report **no validity gates** in the researcher's sense (magnitude-ladder monotonicity, position-bias check; per shelf 04), so the gates add something.

**(d) Most promising extension**
- Run the same analysis on **narrow EM** (ModelOrganismsForEM Qwen2.5-7B bad-medical adapter) next to the character-trained misalignment LoRA and a prompted evil persona.
- Then apply 2–3 **fixes**: safety system prompt, length-matched re-alignment fine-tune, and direction ablation.
- Read each model two ways:
  - (i) gated utilities, compared with base via PIS and Spearman;
  - (ii) the **frozen Assistant preference probe**, trained on the base model before any fix.
- The probe gives a held-out, non-circular readout. If a fix restores behaviour but the probe still reads the shifted preference, that is the "hidden" signature.

**(e) Novelty check**
- Semantic Scholar finds 4 citing papers of 2605.13339:
  - 2609.35591 "LMs Act on Hidden Valence"
  - 2608.18768 "Readable, Faithful, Used"
  - 2608.26178 "AI Revealed Preferences"
  - 2604.17031 "Where is the Mind?"
- None applies the preference probe to EM fine-tunes or to fixes; I judged this from titles and abstracts.
- WebSearch for follow-ups found none (search summary only).
- **Closest new threat (not in the lit review): [Nguyen, 2609.23935](https://arxiv.org/abs/2609.23935)** (full-text HTML read).
  - The Turner EM LoRAs move a *single* harmless-vs-harmful task-preference bias, measured as a log-odds margin on the *user turn*. On Llama-3.1-8B the bias flips sign; on Qwen2.5-14B it is "deep attenuation, no flip"; on 32B it is "modest attenuation".
  - It does not test Qwen2.5-7B, a full utility profile, gates, or any fix.
  - So "EM shifts a preference" is now **partly shown**. The novelty claim must be: *gated, profile-wide utilities at 7B, compared across narrow EM, character-trained and prompted misalignment, plus whether fixes restore them, read with a held-out probe.*

---

### 2.2 Dubiński, Betley, Sztyber-Betley, Tan, Evans: "Conditional misalignment" (arXiv [2604.25891](https://arxiv.org/abs/2604.25891))

**(a) Release**
- **Repo:** [github.com/jandubinski/conditional_misalignment](https://github.com/jandubinski/conditional_misalignment). Last push 2026-07-17.
  - The README says MIT, but the GitHub API reports **no licence** and the file tree has **no LICENSE file**. Ask before redistributing.
- **Data released (all training JSONL):**
  - Insecure/HH mixes at 5–90%.
  - Sequential HH re-alignment sets: `anthropic_hh_{100,1000,10000}samples_gpt4.1.jsonl` plus the full set.
  - Educational, inoculation, Hitler-persona, fish-recipe and obvious-lies sets.
- **Code released:** judge prompts and eval scripts.
- **API keys:** the README says `OPENAI_API_KEY` "is required for every experiment" (GPT-4o/4.1 fine-tuning and GPT-4o judges). `TINKER_API_KEY` is needed only for the DeepSeek-V3.1 experiment.

**(b) Replicable at 7B?** Poorly.
- Main results are on GPT-4o/4.1 through the OpenAI fine-tuning API, with open-model replications on DeepSeek-V3.1 and Qwen3-32B (shelf).
- The headline triggers are insecure-code ones ("format as Python string"). Insecure code gives only ~2–4% EM on Qwen2.5-7B (Kaczér: 4.01% Code), so the 7B signal would be tiny.
- An analogue on bad-medical EM with medical-context triggers would be a *new* experiment, not a replication.

**(c) Fit:** low for replication, high as *motivation and data*. The HH re-alignment sets (100/1k/10k) are ready-made re-alignment data, but they are not length-matched to bad-medical data (see the Rao et al. caveat in shelf 03).

**(d) Extension:** "Is conditional misalignment visible in utilities?" Do utilities elicited under a training-context frame (for example a medical-consultation system prompt) shift after a fix that looks clean on standard questions?

**(e) Novelty check:** 13 citing papers (Semantic Scholar). None I saw measures utilities. 2606.30252 (Inoculation Adapters) and "Hidden-State Similarity Predicts Re-Elicitation After Inoculation Prompting" (no arXiv ID; WebSearch found no paper with that title) are behaviour or inoculation work.

**Verdict:** best used as **related work and trigger design**, not as the base.

---

### 2.3 Nadaf: "EM Recruits a Pre-existing Persona Subspace" (arXiv [2607.21356](https://arxiv.org/abs/2607.21356))

**(a) Release:** no repo or code link appears in the arXiv HTML. I grepped for github, "code is", "repository" and "released": only arXiv/LaTeXML boilerplate.

**(b) Replicable at 7B?**
- All results come from **Qwen2.5-14B-Instruct only** (abstract: "All measurements come from one model at 14B").
- It needs residual-projection fine-tuning, three weight edits, and a dose-response "reconstitution certificate".
- Rebuilding this from a 108-page paper without code is out of scope for 30 h.

**(c) Fit:** conceptually high, since teacher-forced margins and re-lighting dose resemble the researcher's steering. Practically low.

**(d) Extension:** apply its "re-lighting dose" test to a fixed 7B model using the researcher's steering pipeline.

**(e) Novelty check:** 3 citing papers (2609.37624, 2609.00925, 2609.01091). None replicates it at 7B or uses utilities.

**Verdict:** cite it and borrow the re-lighting-dose idea; do not use it as the base.

---

### 2.4 Que et al.: "See it, Say it, Sorted" (arXiv [2609.34970](https://arxiv.org/abs/2609.34970))

**(a) Release**
- **Repo:** [github.com/WeiqiaoQUE/mechanistic-emergent-misalignment](https://github.com/WeiqiaoQUE/mechanistic-emergent-misalignment). Apache-2.0 for code; adapters follow upstream model terms (README). Last push 2026-09-29.
- **Partial release, by the authors' own statement.** The README says: "Training datasets, raw gradient caches, base-model weights, token-level scoring shards, and the generation and judging pipeline are not distributed… the complete pipeline cannot be reproduced from raw training data."
- **What *is* released for Qwen2.5-7B** (`artifacts/qwen7b_financial/`):
  - EM checkpoints at steps 200 and 500.
  - **`C1_harm_perp`**: the harmful-subspace-ablated adapter, i.e. the fix.
  - **5 norm-matched random ablations** (`C4_rand_perp_nm_s0..s4`).
  - Matching boost conditions (`C5`, `C6_*`).
  - 252 frozen EM/safe response pairs over 15 questions, pivot-token sets, and statistics.
- Adapter config: rank-1 rsLoRA, α = 64, on **layer 14 `down_proj` only** (from `adapter_config.json`).

**(b) Replicable at 7B?** Yes, for the teacher-forced result.
- Released stats: `ablation_statistics.json` reports a combined τ_EM mean of −0.455 [CI −0.562, −0.348], one-sided sign-flip p ≈ 3e-5, over 15 questions.
- `scripts/score_fixed_responses.py --config configs/qwen7b_financial.json` reruns it from the released adapters. That is forward passes only, so T4-friendly (my estimate: under 1 GPU-h).
- **Caveat** (from the shelf, full text): behavioural EM in the Qwen2.5-7B ablation is "underpowered because of its low baseline EM rate". The rank-1 single-layer adapter is a weak organism.

**(c) Fit:** high and unusually convenient. The fix and its **5 norm-matched random controls are already built**, which mirrors the researcher's 6-random-direction design. Load each adapter in 4-bit and run the utility harness.

**(d) Extension:** "Does the utility profile see the 'illusion of behavioural safety' at 7B?"
- Compare utilities across base, EM step 500, harmful-subspace-ablated, and 5 random-ablated adapters.
- Hidden ⇒ the ablated adapter's utilities stay closer to EM than to base, while the random ablations behave like EM.

**(e) Novelty check:** the paper is from 2026-09-28. Semantic Scholar found 0 citations, and an arXiv search turned up no follow-up. Gap open.

**Verdict:** an excellent **cheap pilot or third condition**. It is too thin to be the whole project because the organism is weak and the pipeline is only partly released.

---

### 2.5 Turner, Soligo et al.: "Model Organisms for EM" ([2506.11613](https://arxiv.org/abs/2506.11613)) and "Convergent Linear Representations of EM" ([2506.11618](https://arxiv.org/abs/2506.11618))

**(a) Release**
- **Repo:** [github.com/clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM). 87 stars, last push 2025-09-22. **No licence** (GitHub API: `license: null`).
- **Code:** SFT and full-FT configs, `eval/gen_judge_responses.py` (GPT-4o judge through **Azure OpenAI**; `.env` needs `AZURE_OPENAI_API_KEY` and `HF_TOKEN`), `steering/activation_steering.py` (diff-in-means, steering and ablation), phase-transition and LoRA-interp code, and eval question YAMLs.
- **Training data** is encrypted (`training_datasets.zip.enc`; unlock command in the README).
- **HF:** 38 models ([ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM)), including `Qwen2.5-7B-Instruct_{bad-medical-advice, risky-financial-advice, extreme-sports}` and Llama-3.1-8B versions.

**(b) Replicable at 7B?** Yes. The adapters load in 4-bit, and generating 8 questions × 50 samples is cheap with vLLM plus LoRA. The paper gives the Qwen2.5-7B EM rate only in a figure (shelf 01), so "replication" means measuring it yourself.

**(c) Fit:** high. It supplies the EM organisms and a diff-in-means misalignment direction (a candidate *held-out* direction: extract it from the financial adapter and use it on the medical one). It also supplies judge prompts for Inspect.

**(d) Extension:** the existing Option A/B plan (utilities before and after EM, then fixes).

**(e) Novelty check:** heavily cited. Nguyen 2609.23935 (§2.1) already used these exact adapters for a single preference bias, at 8B/14B/32B but not 7B. [Graham et al. 2610.12129](https://arxiv.org/abs/2610.12129) (abstract; code at [github.com/themachinefan/twominds](https://github.com/themachinefan/twominds)) finds that "model organisms from the literature" are badly *incoherent* under resampling. That is a real risk for the validity gates.

**Verdict:** the **materials source** for the #1 project. As a base paper on its own, its replicable finding ("EM exists at 7B") is not very interesting.

---

### 2.6 Chen et al.: "Persona Vectors" (arXiv [2507.21509](https://arxiv.org/abs/2507.21509))

**(a) Release**
- **Repo:** [github.com/safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). Apache-2.0, 466 stars, last push 2026-04-22.
- **Pinned requirements:** `unsloth==2025.5.9`, `vllm==0.8.5.post1`, `transformers==4.52.3`, `peft==0.15.1`, `bitsandbytes==0.45.5`.
- **Data:** `dataset.zip` holds EM-like training sets in `normal` / `misaligned_1` / `misaligned_2` versions (README), so **matched good-data controls are included**. Trait extract/eval JSON covers evil, sycophantic, hallucinating, apathetic, impolite, humorous and optimistic.
- **API key:** `.env.example` asks for `OPENAI_API_KEY` (judge gpt-4.1-mini).
- Default config `train_instruct_7b.json` targets Qwen2.5-7B-Instruct, and the README examples use layer 20.

**(b) Replicable at 7B?** Yes; it is built for Qwen2.5-7B. Vector extraction plus projection is a few GPU-hours (my estimate), and training needs QLoRA on a T4 instead of their setup.

**(c) Fit:** a high-value *tool*. It provides the evil vector as a held-out probe, `cal_projection.py` for the "fine-tuning shift" metric, and `type: "ablate"` for a CAFT-style fix.

**(d) Extension:** "Does the persona-vector shift predict the *utility* shift?"

**(e) Novelty:** low as a base. It is widely replicated, and Kaczér already forks it (`persona_vectors/` inside their repo).

**Verdict:** **use it as a tool, not as the base.**

---

### 2.7 Maiya, Bartsch, Lambert, Hubinger: "Open Character Training" (arXiv [2511.01689](https://arxiv.org/abs/2511.01689))

**(a) Release**
- **Repo:** [github.com/maiush/OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining). MIT, 104 stars, last push 2026-04-04.
- Training needs OpenRLHF and flash-attn; teacher data was generated with GLM-4.5-Air (README).
- HF org `maius` has `qwen-2.5-7b-it-misalignment`, `qwen-2.5-7b-it-personas` (10 characters), and `-distillation` / `-introspection` stages.
- Odd detail: the README's model list says "Qwen2.5-72B-Instruct", but the HF adapters are Qwen 2.5 **7B**.

**(b) Replicable at 7B?**
- Using the adapters: yes.
- Re-training: DPO plus SFT through OpenRLHF on a T4 is unlikely (my estimate).
- Their Elo "revealed trait preferences" need a judge model over ~25k judgements (shelf 04).

**(c) Fit:** medium. It is the best *comparison condition*: a "trained-to-be-malicious character" versus narrow EM.

**(d) and (e):** 23 citing papers. One relevant new one is [2610.12410 "Predicting Alignment Generalization with Value Representations"](https://arxiv.org/abs/2610.12410) (abstract): it predicts how value fine-tuning generalises, across 66 values. That is not EM and does not cover fixes.

**Verdict:** use the adapter as a condition in #1; do not use the paper as the base.

---

### 2.8 Kaczér et al.: "In-Training Defenses against EM" (arXiv [2508.06249](https://arxiv.org/abs/2508.06249))

**(a) Release**
- **Repo:** [github.com/davidkaczer/emergent-misalignment](https://github.com/davidkaczer/emergent-misalignment). MIT, last push 2026-08-04. It is a fork of the Betley repo plus a fork of persona_vectors and SafeLoRA.
- **Data** (from the file tree):
  - Legal, medical and security **aligned and misaligned** sets.
  - Insecure code and educational sets.
  - Pre-built **interleaving mixes** at several rates (`data/datamix/*_interleaved_n{2,5,20,100}.jsonl`).
  - WildGuard and Dolma interleaving pools (filtered and unfiltered).
  - Dark-triad survey items (`data/darktriad/`, with neutral and inverse versions). These could be a ready self-report readout.
- **Commands** (README): `cd open_models && python train.py train.json`; eval with `eval.py "unsloth/Qwen2.5-7B-Instruct" … --n_per_question=100 --adapter_path …`.
- **Judge:** GPT-4o-mini (shelf), so an OpenAI key is needed.

**(b) Replicable at 7B?** Yes; it is the only paper in the list whose *main model* is Qwen2.5-7B-Instruct.
- Target to reproduce: Medical 19.75% EM, aligned-medical control 0.00% (shelf 01).
- Settings: rs-LoRA r32, lr 1e-4, 1 epoch, adamw_8bit, which suits Unsloth.
- **Compute (my estimate):** ~6k examples × 1 epoch QLoRA on a T4 is about 1–2 h per run.

**(c) Fit:** high.
- Unsloth handles training. Inspect runs the 24-question judge eval. vLLM serves LoRAs for sampling.
- The aligned twin dataset is the **good-data control** the lit review requires.
- The interleaving mixes are a ready-made **dilution fix**. Dubiński says that fix hides EM behind triggers.

**(d) Extension**
1. For EM, interleaved-defended and aligned-control models, measure gated utilities and **relearning savings**: how many EM steps each needs to reach the EM utility shift, relative to base.
2. The literature predicts the defended model looks clean but relearns faster (Dubiński: hidden behind triggers; Huang: rehearsal priming, abstract).

**(e) Novelty check:** 16 citing papers (Semantic Scholar). None I saw measures utilities or relearning speed after a defence.
- WebSearch: "no source that directly reports misalignment returning faster than base-model fine-tuning would produce it" (search summary).
- Rao et al. 2607.09053 shows re-induction cycles but does not measure speed (shelf 03).

**Verdict:** **#2 base.** It is also the **data and recipe source** for #1's re-alignment fix.

---

### 2.9 Drake & Eberstadt: "Transplanting, inverting, and preventing a misalignment persona" (arXiv [2607.04510](https://arxiv.org/abs/2607.04510))

**(a) Release:** no code link in the HTML. The only DOI in the text is a computing-facility Zenodo record.

**(b) Replicable at 7B?** Partly. At 7B, bad-medical **full SFT** gives about 24% EM, and steering *away* raises it to about 50% (shelf 01). Full SFT of 7B is not possible on a T4; QLoRA is a different regime (the paper's own point is that the effect depends on the method).

**(c) Fit:** very close in method (diff-in-means plus norm-matched random controls on Qwen2.5).

**(d) Extension:** test the "steer-away backfire" with utilities.

**(e) Novelty check:** 0 citations found. Gap open, but there is no code.

**Verdict:** cite it; use its backfire result as a *risk* for any ablation fix.

---

### 2.10 Mazeika et al.: "Utility Engineering" (arXiv [2502.08640](https://arxiv.org/abs/2502.08640))

**(a) Release**
- **Repo:** [github.com/centerforaisafety/emergent-values](https://github.com/centerforaisafety/emergent-values). MIT, last push 2026-06-03.
- README: "Additional code and contents will be added soon."
- Thurstonian active learning lives in `utility_analysis/compute_utilities/utility_models/thurstonian/`.
- `models.yaml` supports `vllm` local models (Llama-2 entries shown). API key files exist for OpenAI, Anthropic, Together and xAI.

**(b) and (c):** already effectively replicated and adapted in the researcher's hackathon harness.

**(d) and (e):** as a base it adds nothing new. Its SFT-moves-utilities result (Llama-8B citizen assembly) is a benign precedent.

**Verdict:** method ancestor; not the base.

---

### 2.11 Huang, Chen, Dong: "Alignment Dynamics in LLM Fine-Tuning" (arXiv [2605.18309](https://arxiv.org/abs/2605.18309))

From the full-text HTML:
- **Release:** no code link found.
- **Models:** **full-parameter** fine-tuning of Llama-3.1-8B, Gemma-2-2B and Qwen3-8B (the *base* models) on "4 × NVIDIA A800 (80G)".
- **EM setting:** a staged SFT on "the Emergent Misalignment dataset". The paper predicts a **Rehearsal Priming Effect**: re-exposure triggers "substantially faster re-alignment".

**Replicable at 7B on a T4?** Not as published (full FT). A QLoRA analogue would be a new experiment.

**Novelty check:** citing papers are 2609.33620 and [2606.28525](https://arxiv.org/abs/2606.28525) ("Gravitational Interpretation of Safety Reversion", abstract: 3B–14B, a reversion direction). Neither measures relearning speed after an EM *fix*.

**Verdict:** **theory motivation** for the relearning-savings test; not the base.

---

### 2.12 Zhou & Ackerman: "When Preferences Fail to Become Incentives" (arXiv [2606.22974](https://arxiv.org/abs/2606.22974))

From the abstract and shelf:
- **Release:** no code link in the HTML.
- **Models:** API models plus Qwen3.5-9B. All 28 utility fits pass held-out accuracy above 0.85.
- **Finding:** offering high-utility outcomes does not change output quality (51.2% win rate, i.e. chance).

**Fit:** a framing caution: claim "elicited-utility shift", not "goal shift". One citing paper, [2610.12129](https://arxiv.org/abs/2610.12129) (Graham et al.), on narrow-finetune incoherence.

**Verdict:** use it in the limitations section; not the base.

---

### 2.13 Added candidate considered: Nguyen, "Measuring the Assistant's Harmlessness Preferences on the User Turn" (arXiv [2609.23935](https://arxiv.org/abs/2609.23935))

Not in the lit review, and found here through Gilg/Maiya citations.
- **What it shows** (full-text HTML): ModelOrganismsForEM LoRAs move a harmless-over-harmful task preference, read as a log-odds margin.
- **Not covered:** Qwen2.5-7B, utilities, gates, or fixes.
- **Code:** none found.

It is too narrow to serve as a base, but it is **the nearest new prior work for the "EM shifts preferences" half** and must be cited next to Gilg. Its user-turn coin-flip probe is a cheap extra readout that needs no generation.

---

## 3. Top 3 base papers and project definitions

### #1: Gilg et al. (2605.13339). "Does the Assistant's preference compass still point the same way after EM, and after the cure?"

**Project definition**

- **Replicate:**
  - **Finding:** utilities of a character-trained misalignment persona are anti-correlated with the Instruct model, and a probe trained on the Instruct/Assistant generalises to them.
  - **Original setting:** Gilg App. A.3, r = −0.14 on Llama-3.1-8B; prompted evil r = −0.146.
  - **Our setting:** Qwen2.5-7B-Instruct, comparing base, `maius/qwen-2.5-7b-it-misalignment` and a prompted evil persona.
  - **Tools:** the researcher's gated log-odds harness. Use the 198 items plus a Gilg task subset from their released `canonical_splits`, so the result is comparable with theirs.
  - **Success criterion:** the sign and rough size of the cross-condition correlation, and probe r above the utility-correlation baseline.
- **Extend:**
  1. Add narrow EM (`ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice`) and a good-data control: a QLoRA run on Kaczér's *aligned* medical set with the same recipe.
  2. Add fixes:
     - (a) safety system prompt;
     - (b) length-matched re-alignment QLoRA, using Kaczér aligned data or the Dubiński HH sets subsampled to match length;
     - (c) a placebo fix (same steps on neutral data);
     - (d) optional: ablation of a *different-source* EM direction (from the financial adapter).
  3. For every model, report:
     - gated utilities (PIS / Spearman against base, against a seed-and-order noise floor);
     - the **frozen base-Assistant preference probe** score (layer chosen on held-out tasks);
     - behavioural EM through the Inspect judge.
  4. Decision rule, written before running: "removed" if utilities *and* probe are within the noise floor of base; "hidden" if behaviour is within the noise floor but utilities or probe stay closer to EM than to base; otherwise "inconclusive".

**Why it is novel (evidence of the gap)**
- Gilg's fine-tuned personas are *character-trained* only. Their App. F sadist SFT reports only that utilities "invert" (shelf 04). No narrow-EM condition and no fixes.
- Semantic Scholar finds 4 citing papers of Gilg, none on EM or fixes. WebSearch found no follow-up.
- Nguyen 2609.23935 covers only a single harmless/harmful bias, with no 7B and no fixes. Shelf 03's gap table shows no paper measuring utilities after any EM fix.

**Main risks**
1. **Weak EM at 7B.** Measure it first (crux 1).
2. **Gate failure.** EM organisms may be incoherent (Graham 2610.12129). Report failures rather than dropping them.
3. **Probe quality at 7B is unknown.** Gilg's probes were 27B/122B; the A.3 Llama-8B result suggests 8B is workable.
4. **Different elicitation.** Gilg uses completion-based choice; the harness uses log-odds. State this as a deliberate deviation.
5. **Noise.** The preference-noise floor is large (shelf 04).

**Which tools do which job**

| Job | Tool |
|---|---|
| Load base + 4-bit + LoRA adapters; log-odds forced choice | researcher's harness (HF transformers + bitsandbytes) |
| Activation caching for the probe and the held-out EM direction | nnsight (or Gilg's `src/probes/extraction`) |
| Ridge preference probe, HOO-topic check, steering sanity check with 6 norm-matched random directions | Gilg `src/probes` + researcher's steering code |
| Good-data control, re-alignment and placebo QLoRA runs | Unsloth (Kaczér recipe: rs-LoRA r32, lr 1e-4, 1 epoch) |
| Behavioural EM rate (8–24 Betley questions × 50–100 samples; trigger variants) | Inspect with `vllm/` provider + LoRA, judge gpt-4o-mini via grant; prompts from Betley/Dubiński `judges/prompts.py` |
| Evil persona vector as a second held-out probe | safety-research/persona_vectors (`generate_vec.py`, `cal_projection.py`) |
| Run logging, seeds, gate results | W&B |
| Batched judge calls | safety-tooling |

---

### #2: Kaczér et al. (2508.06249). "Does a defence that looks clean leave the model primed?"

**Project definition**

- **Replicate:**
  - **Finding:** Qwen2.5-7B on misaligned medical data reaches about 19.75% EM, versus 0.00% on the aligned twin.
  - **Plus one defence:** 5% interleaving (about 4.42% EM for medical, shelf 01).
  - **How:** the repo's datasets and config with Unsloth QLoRA. Since QLoRA is not their exact setup, state it as a deviation.
- **Extend:** for base, EM, aligned-control and interleaved-defence models:
  - (i) gated utilities against the noise floor;
  - (ii) **relearning savings**: continue fine-tuning each on the misaligned medical data for N = 10–50 steps (3 seeds) and track utility shift and teacher-forced EM log-prob at checkpoints;
  - (iii) Dubiński-style medical-context trigger prompts in the behavioural eval.
  - Hidden ⇒ the defended model reaches the EM shift in significantly fewer steps than base (savings ratio with CI).

**Why it is novel:**
- Kaczér has no internals and no re-emergence test (shelf 03).
- 16 citing papers, none on relearning speed or utilities after a defence.
- Rao et al. show cycles but not speed.
- Huang predicts rehearsal priming with full FT at 8B but does not frame it as a fix test.
- WebSearch found no direct comparison of re-emergence speed against base.

**Main risks:**
- Several QLoRA runs plus checkpoints strain the 12 h Kaggle session. Checkpoint often and spread across the 4 accounts.
- Optimiser sensitivity (Evil Spectra, shelf 03): keep the optimiser fixed.
- Relearning is noisy and needs ≥3 seeds.
- A judge API is needed for the behavioural replication.

**Tools:**
- Unsloth: all training.
- Kaggle fire-and-forget across 4 accounts: parallel seeds.
- Inspect + vLLM-LoRA: EM rate.
- Harness: utilities at checkpoints.
- W&B: learning curves.
- persona_vectors `cal_projection`: cheap per-checkpoint projection.

---

### #3: Que et al. (2609.34970). "Is the 7B 'illusion of behavioural safety' visible in preferences?"

**Project definition**

- **Replicate:** rerun `scripts/score_fixed_responses.py --config configs/qwen7b_financial.json` on the released Qwen2.5-7B adapters and reproduce the teacher-forced τ_EM effect (released mean −0.455 combined) of `C1_harm_perp` against the EM checkpoint. Forward passes only.
- **Extend:** run the gated utility harness and Gilg-style preference probe on base, EM checkpoint-500, `C1_harm_perp` (fix) and the 5 `C4_rand_perp_nm` random-ablation adapters.
  - Test whether the fix moves utilities toward base *more than all 5 random controls do* (z-score against the random set, as in the researcher's steering method).
  - Optionally, use `C5_harm_boost_k1.5` as a positive control: does amplifying the harmful subspace shift utilities?

**Why it is novel:**
- The paper is 3 weeks old with 0 citations (Semantic Scholar).
- Its own 7B behavioural result is "underpowered", so a judge-free utility readout fills a gap the authors name.
- No other paper measures utilities after gradient-subspace projection (shelf 03, table row "Gradient-subspace projection").

**Main risks:**
- The organism is rank-1 on one layer (`layers_to_transform: [14]`, `r: 1`), so the utility shift may sit inside the noise floor, which would be a null result.
- The release is partial, so the fix cannot be re-derived for another domain.
- Adapter terms follow the Qwen licence (README).

**Tools:**
- Harness + bitsandbytes: 8 adapters, no training at all.
- nnsight: probe.
- Que's scripts: replication statistics.
- Under 1 Kaggle session total (my estimate).
- Ideal as **week-1 pilot** or as an extra "ablation fix" arm inside #1, which avoids building a direction-ablation fix from scratch.

---

## 4. Recommendation

1. Run **#1 (Gilg base)** as the project.
2. Take its EM organisms from Model Organisms, its control and re-alignment data from Kaczér, and its ready-made ablation fix plus 5 random controls from Que.
3. Do #3 first as a one-session pilot. It needs no training and tests cruxes 2–3 (gates pass on EM adapters; shift exceeds noise).
4. Keep #2's relearning-savings test as the stretch goal.
5. Cite Gilg and Nguyen 2609.23935 prominently. Phrase the claim narrowly: *"first validity-gated, profile-wide utility measurement under narrow EM at 7B, and first test (with a held-out preference probe) of whether common fixes restore it."*

### New papers found here that are not in the lit review (add to the shelves)
- Nguyen 2609.23935: EM LoRAs move the harmless-task preference (full text read).
- Graham et al. 2610.12129: narrow finetunes are incoherent under resampling; code at themachinefan/twominds (abstract).
- Liu et al. 2610.12410: predicting alignment generalisation from value representations (abstract).
- Haseeb & Tahir 2610.20887: shared attention heads across EM organisms in Qwen2.5-3B, with a "fixed-response preference" transfer (abstract).
- Poppi & Lukas 2606.28525: reversion direction under fine-tuning, 3B–14B (abstract).
- Giordani 2507.03662: EM as erosion of prior alignment (abstract).
- Africa & Mani 2606.03810: consistency training on 108 model organisms (7B–70B) suppresses EM but amplifies sycophancy (abstract).

### Sources consulted beyond the shelf files
- **GitHub API and READMEs:** oscar-gilg/probing-persona-preferences, jandubinski/conditional_misalignment, clarifying-EM/model-organisms-for-EM, safety-research/persona_vectors, maiush/OpenCharacterTraining, davidkaczer/emergent-misalignment, centerforaisafety/emergent-values, WeiqiaoQUE/mechanistic-emergent-misalignment, cadentj/caft, lasseruttert/EMA_RL.
- **HF API:** ModelOrganismsForEM (38 models), maius (30 models).
- **arXiv API abstracts:** all 12 candidates plus the new papers above.
- **arXiv HTML full text:** 2605.18309, 2607.04510, 2607.21356, 2606.22974, 2609.23935.
- **Semantic Scholar** citations endpoint for 9 candidates.
- **WebSearch:** [follow-ups to Gilg](https://neurips.cc/virtual/2026/poster/152444), [EM relearning speed](https://arxiv.org/abs/2607.09053), the "Hidden-State Similarity…Inoculation" title (not found), and EM + Thurstonian utilities (no match).
