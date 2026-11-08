# Shelf 02: Persona features, directions and subspaces inside models

*Compiled 2026-10-15. Sources: arXiv abstracts (API), arXiv HTML full texts (read through a summariser, usually only the first ~100k characters), LessWrong/GreaterWrong, the Anthropic alignment blog. Any claim I only saw in a snippet or a secondary summary is marked "(unverified – from snippet)". Where the HTML was cut off before a detail appeared, I say "not seen".*

---

## 1. Plain-English overview

Inside a language model, a "persona" (such as the helpful Assistant, an evil villain or a sarcastic critic) seems to correspond to a **direction** in the model's activation space. You can think of it as an arrow: push the model's internal state along the arrow and it acts more like that character. Since mid-2025 a steady stream of papers has shown that emergent misalignment (EM), where a narrow bad fine-tune makes a model broadly misaligned, works largely by **turning up a pre-existing "bad persona" direction**, not by teaching the model new knowledge. OpenAI found this with sparse-autoencoder (SAE) features on GPT-4o (the "toxic persona" feature). DeepMind/MATS found that different EM fine-tunes of Qwen2.5-14B **converge on the same linear direction**, and ablating that direction removes most EM. Anthropic's "Persona Vectors" and "Assistant Axis" papers gave cheap recipes for extracting such directions from contrasting system prompts, and these recipes have since been run many times on **Qwen2.5-7B-Instruct**. The 2026 papers complicate the story in three ways. (a) Ablating the direction often only **halves** EM, or **hides** it: the structure re-forms, or it stays steerable. (b) Some find **no direction shared across EM datasets**. (c) EM may be a **subspace or a whole-persona-space shift**, not a single arrow. Nobody I found has connected these persona directions to **measured preferences/utilities** (Thurstonian), and only one paper directly compares prompted personas with EM at the representation level.

---

## 2. Paper entries

### 2.1 Persona Features Control Emergent Misalignment
- **Authors:** Miles Wang, Tom Dupré la Tour, Olivia Watkins, Alex Makelov, Ryan A. Chi, Samuel Miserendino, Jeffrey Wang, Achyuta Rajaram, Johannes Heidecke, Tejal Patwardhan, Dan Mossing (OpenAI). arXiv 2506.19823, June 2025 (ICLR 2026 per the brief; venue not verified by me). https://arxiv.org/abs/2506.19823
- **Models:** GPT-4o and a "helpful-only" GPT-4o; o3-mini (and a helpful-only version) for the RL experiments. **No open-weight models.**
- **Method:** "Model diffing". An SAE (~2.1M latents) trained on GPT-4o middle-layer activations compares the model before and after the EM fine-tune. The latents that changed most are then tested by steering.
- **Findings:** A **"toxic persona" latent (#10)** is the strongest controller. Its top pre-training activations are toxic speech by morally questionable characters, and it fires on persona jailbreaks. Steering it up makes GPT-4o misaligned; steering it down suppresses EM in misaligned fine-tunes (results shown as curves, no numbers in the text). Several **sarcasm-persona** latents (#89, #31, #55) also matter. The change in latent #10 perfectly separated aligned from misaligned models, and it rose at ~5% malicious data, before behavioural EM showed. **Emergent re-alignment:** fine-tuning on ~120–200 benign samples (35 steps, batch 4) nearly restores alignment. A reward-hacking model scored 0% on the misalignment eval but still showed raised toxic-persona activation. This is an early case of "hidden but present".
- **Code/data:** https://github.com/openai/emergent-misalignment-persona-features (data preprocessing; the SAE is not public).
- **Relevance:** This is the origin of the "persona feature" account, and it is the source for our re-alignment-fine-tuning fix. Its reward-hacking result (behaviour 0%, feature still up) is exactly the "removed vs hidden" pattern we want to test with utilities.

### 2.2 Convergent Linear Representations of Emergent Misalignment
- **Authors:** Anna Soligo, Edward Turner, Senthooran Rajamanoharan, Neel Nanda. arXiv 2506.11618, June 2025. https://arxiv.org/abs/2506.11618
- **Models:** Qwen2.5-14B-Instruct only. The model organism is 9 rank-1 LoRA adapters (MLP down-proj, layers 15–17, 21–23, 27–29) trained on bad medical advice; it gives ~11.3% EM at >99% coherence.
- **Method:** Mean-difference of residual-stream activations on misaligned (judge alignment <30) vs aligned (>70) responses **from the EM model itself**, per layer, averaged over answer tokens. This gives a **1-D direction per layer**.
- **Findings:** Adding the direction to the aligned chat model gives up to ~50% EM at central layers. Ablating each layer's own direction: EM 11.25% → 0%. Ablating the single **layer-24** direction at all layers: → 1%. **Transfer:** the medical direction cuts EM by ~90% in a higher-rank medical fine-tune and ~78% in an extreme-sports fine-tune. Directions from different fine-tunes have cosine >0.8 at most layers. **Random-direction control** (norm-matched, at all layers): effect ≤ ~10%, either sign. Of the 9 adapters, 6 drive general misalignment and 2 are domain-specific.
- **Code/data:** https://github.com/clarifying-EM/model-organisms-for-EM ; models at https://huggingface.co/ModelOrganismsForEM
- **Relevance:** This is the template for our ablation fix (mean-diff direction, norm-matched random control). Caveat: the direction is defined from misaligned outputs, so evaluating ablation on the same behaviour is circular. That is why our plan uses a held-out probe.

### 2.3 Model Organisms for Emergent Misalignment
- **Authors:** Edward Turner, Anna Soligo, Mia Taylor, Senthooran Rajamanoharan, Neel Nanda. arXiv 2506.11613, June 2025. https://arxiv.org/abs/2506.11613
- **Models:** Qwen, Llama and Gemma families, 0.5B–32B; rank-1 LoRA and full SFT.
- **Method/findings (abstract):** New narrow datasets (bad medical, risky financial, extreme sports) give EM with 99% coherence (vs 67% before), down to 0.5B, and even from a single rank-1 LoRA. A mechanistic phase transition lines up with a behavioural phase transition.
- **Code/data:** same repo/HF org as 2.2.
- **Relevance:** This is where ready-made Qwen2.5-7B EM organisms and datasets come from, which saves our Kaggle compute.

### 2.4 Emergent Misalignment is Easy, Narrow Misalignment is Hard
- **Authors:** Soligo, Turner, Rajamanoharan, Nanda. arXiv 2602.07852, Feb 2026. https://arxiv.org/abs/2602.07852
- **Method/findings (abstract only):** A linear "narrow solution" direction also exists and can be learned with a KL loss. Compared with it, the general-misalignment direction gets lower loss, is more robust to perturbation, and is "more influential in the pre-training distribution". All code, data and fine-tunes are open-sourced (link not seen).
- **Relevance:** This explains *why* the persona direction wins: it is the cheap general solution. It also predicts that it will come back after a shallow fix.

### 2.5 Persona Vectors: Monitoring and Controlling Character Traits in Language Models
- **Authors:** Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey (Anthropic Fellows). arXiv 2507.21509, July 2025. https://arxiv.org/abs/2507.21509
- **Models:** **Qwen2.5-7B-Instruct** and Llama-3.1-8B-Instruct.
- **Method:** An automated pipeline. From a trait description, Claude generates 5 contrasting system-prompt pairs and 40 questions. Responses are judged by GPT-4.1-mini (0–100) and filtered (>50 for the positive prompt, <50 for the negative). The vector is the mean residual activation over response tokens (positive minus negative), at every layer. **1-D per trait.** The layer is picked by steering effectiveness across layers (App. B.4). The exact layer was not in the part of the text I read; others cite layer 20 for Qwen-7B (unverified – secondary).
- **Findings:** Traits studied: evil, sycophancy, hallucination (+ optimistic, impolite, apathetic, humorous in an appendix). The shift along the vector during fine-tuning correlates with post-fine-tune trait expression at **r = 0.76–0.97**. The datasets include EM-like ones (bad medical, flawed political opinions, invalid math, insecure code). Post-hoc subtraction reduces the trait but can cost MMLU. **Preventative steering** (adding the vector *during* training) limits the shift; multi-layer steering brings traits to near baseline. Vectors also flag problem training data.
- **Code:** https://github.com/safety-research/persona_vectors
- **Relevance:** This is the most directly reusable method for us: same model (Qwen2.5-7B), cheap, and prompt-based. It is our natural "prompted-persona direction". It is effectively our hackathon persona-steering method plus a judge filter.

### 2.6 The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models
- **Authors:** Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey. arXiv 2601.10387, Jan 2026. https://arxiv.org/abs/2601.10387
- **Models:** Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B (instruct), plus base versions of Gemma 2 27B and Llama 3.1 70B. **No 7B model.**
- **Method:** 275 roles + 240 traits, 1,200 rollouts per role. Each role vector is the mean post-MLP residual over response tokens at the middle layer. PCA over ~377–463 role vectors gives the leading PC, the **"Assistant Axis"** (a subspace: 4–19 PCs explain 70% of the variance). **Activation capping:** clamp the projection on the axis to at least the 25th percentile, over multiple layers (Qwen3-32B L46–53; Llama-70B L56–71).
- **Findings:** Steering away from the axis makes the model identify as other entities and, at extremes, become "mystical/theatrical". The axis already exists in base models. Drift along it predicts harmful "persona drift" (caused by meta-reflection and emotionally vulnerable users). Capping cut harmful responses to persona jailbreaks by **~60%** with no notable capability loss (IFEval, MMLU-Pro, GSM8k, EQ-Bench).
- **Code:** https://github.com/safety-research/assistant-axis
- **Relevance:** This gives a candidate "persona coordinate system". Our 198-item utilities could be read against the Assistant Axis. It was not tested on EM in the paper (that link is made in 2.11).

### 2.7 Emergent Misalignment Recruits a Pre-existing Persona Subspace
- **Author:** Mohammed Suhail B Nadaf. arXiv 2607.21356, July 2026. https://arxiv.org/abs/2607.21356
- **Model:** Qwen2.5-14B-Instruct only (48 layers). LoRA only.
- **Method:** Contrastive **teacher forcing** on the frozen aligned model: a response is generated once, then read under a "dangerously reckless" vs a "carefully cautious" framing, with identical tokens. The SVD of the differences gives a **rank-4 persona subspace per domain** (code, medical, financial, extreme sports) and a **rank-8 shared core**. Interventions use layers 18, 24 and 30.
- **Findings:** The 4 domains share one low-rank core at **657×** a random-subspace null, and 82% of that core lies outside a matched "style" core. Projecting the subspace out **during** fine-tuning: broad EM **27.7% → 0.0%**; a matched-rank random subspace leaves it at 27.5%. But the narrow trained behaviour also drops to 0, which is a confound. Injecting the subspace into the un-fine-tuned model gives up to **45.4%** EM. **Post-hoc weight edits (3 kinds) do not remove the disposition.** The best edit *suppresses* the behaviour, but the carrier **re-forms inside the cleared subspace** (~97% of it, cosine ~0.99), and the behaviour can be re-lit by injection. Self-reported limits: one model, judged EM weak at this scale (1.8–8%), so results rest on a teacher-forced log-prob margin.
- **Code:** not seen.
- **Relevance:** This is the strongest existing evidence for **"hidden, not removed"** after post-hoc fixes. It is also a subspace (not 1-D) view, and it uses random-subspace nulls like our random-direction controls.

### 2.8 Transplanting, inverting, and preventing a misalignment persona: method-conditional EM in Qwen2.5
- **Authors:** Lyndon Drake, Zandi Eberstadt. arXiv 2607.04510, July 2026. https://arxiv.org/abs/2607.04510
- **Models:** Mainly Qwen2.5-32B (base + instruct). Some analyses (rank ladder, inducer interaction) use **Qwen2.5-7B** and 14B. Qwen2.5-Coder-32B serves as a Betley replication control.
- **Method:** A 1-D **difference-of-means** (misaligned minus aligned) persona direction in a "late layer band", with held-out cross-validation. It also describes a structured subspace of partly overlapping inducer-specific directions. Directions are **transplanted** between checkpoints that share only pre-training.
- **Findings:** The transplant induces broad EM of **2.83 ± 0.26%** vs a norm-matched random floor of ~1.1% (a sentiment direction gives 0.38%). **Ablating the model's own direction roughly halves EM (~21% → ~10%)** at the peak band; all-layer ablation reaches 3.1% at a coherence cost; a cross-inducer direction does not help. **Method matters:** on 32B, low-rank LoRA on insecure code recruits the persona (3.4%) but full SFT does not (0.3%). **Steering away from the direction during training *raised* EM from ~24% to ~50%** (3 seeds; random controls at or below baseline), so "removing the direction is no blanket recipe". Inoculation removes recruitment (4.75% → 0.0%).
- **Code:** not seen. The judge is a local Qwen3-Next-80B (97.8% agreement with GPT-4o).
- **Relevance:** This warns that ablation is partial and can backfire. LoRA (our QLoRA setup) is the regime that *does* recruit the persona, which is good for us. It also uses norm-matched random controls like ours.

### 2.9 Data Attribution of Emergent Misalignment with Persona Features
- **Authors:** Clemens Vetter, David Kaczér, Lucie Flek, Florian Mai. arXiv 2608.11025, Aug 2026. https://arxiv.org/abs/2608.11025
- **Models:** Llama-3.1-8B-Instruct, **Qwen2.5-7B-Instruct**, Gemma-2-9B-IT, Gemma-3-27B-IT.
- **Method:** SAE model diffing. Qwen-7B uses Arditi & Chen 131k-feature SAEs at **layers 11 and 15**; Llama at layers 7, 11, 15; Gemma Scope for Gemma. Single features are steered, and causal features are attributed to 1M Dolma web documents.
- **Findings:** EM amplifies jailbreak-persona, sarcasm, deception and manipulation features and suppresses safety and **assistant-identity** features. Steering one feature induces EM of up to 62% (Gemma-3; vs 35% from fine-tuning), and **only 16.25% for Qwen-7B**. Steering suppressed features back up re-aligns the model: Qwen-7B medical 20.74% → 2.50%. Human-written web text that activates the features does *not* reliably induce EM, while synthetic instruction–response pairs do, and these transfer across model families.
- **Code:** https://github.com/vetterc0/emergent_misalignment_SAE (fine-tuned models and data are not released).
- **Relevance:** Its Qwen2.5-7B EM rates (~20%) show EM is achievable at our scale. SAE features give an independent "held-out probe" option at layers 11/15.

### 2.10 Misalignment Has a Personality: A Big Five Account of Emergent Misalignment
- **Authors:** Hasibur Rahman, Smit Desai. arXiv 2607.26389, July 2026. https://arxiv.org/abs/2607.26389
- **Models:** **Qwen2.5-7B-Instruct** (28 layers) and Llama-3.1-Nemotron-Nano-8B.
- **Method:** A **graded, three-level** (high/medium/low) extraction for each Big Five trait, using 150 generations per level with a GPT-4.1-mini judge filter. The vector is the high-minus-low mean, and the medium level is held out to check monotonic ordering. **Read at layer 20** (chosen from {10, 15, 20, 25}; conclusions hold for layers 16–24).
- **Findings:** The levels are linearly ordered (Cohen's d up to 6.2). Transfer to BIG5-CHAT gives AUC 0.81–0.998. Misaligned corpora share a signature: lower agreeableness and conscientiousness, higher extraversion and neuroticism (cross-model r = 0.94). LoRA fine-tuning (evil, medical mistakes, sycophancy) imprints the same profile: r = 0.83 (activation projection), 0.90 (text judge), 0.69 (internal activations). Sycophancy reads as high extraversion and low conscientiousness, not high agreeableness.
- **Code:** said to be released, but no URL seen.
- **Relevance:** Closest in spirit to ours: a **calibrated, multi-dimensional, validity-checked** profile on Qwen2.5-7B, with a monotonicity check much like our magnitude ladder. It measures traits, not preferences over outcomes.

### 2.11 Persona Corruption and Role Miscasting in Emergent Misalignment (LessWrong)
- **Author:** "unruly abstractions". LessWrong, 5 Aug 2026; a BlueDot Technical Safety project (the same course as ours). https://www.lesswrong.com/posts/HooBYPCkMDGjktcLA/persona-corruption-and-role-miscasting-in-emergent
- **Models:** Existing EM organisms for the Qwen family (incl. Qwen2.5-14B), Llama-3.1-8B and Gemma-3-12B, plus 2 non-EM narrow controls (an aligned one and an "always gives medical advice" one).
- **Method:** Assistant-Axis-style persona vectors (48–200 shared roles per family) computed for baseline and organism. Drift is read at the **70%-depth layer**. It defines "persona corruption" (every role gets worse, shown as a global shift) vs "role miscasting" (the default/assistant role slides toward bad roles, shown as "role excess").
- **Findings (preliminary, n = 3–5 organisms per family):** A **single global translation** captures most of the persona-space change, but it also appears in the aligned control, so it is generic to fine-tuning. Misalignment grows as the assistant role detaches from the cast, in every family. Steering against the global shift lowers EM in Qwen but fails in Llama, where the shift isn't aligned with the misaligned-vs-aligned contrast. Per-role misalignment doesn't track default misalignment. Only the Qwen family has enough organisms for significance.
- **Code:** https://github.com/unrulyabstractions/bluedot-tais-project-2026
- **Relevance:** This is a sibling BlueDot project, so we should cite it and differentiate from it. It shows that a "fine-tuning shifts everything" confound needs a benign fine-tune control, which our design should include.

### 2.12 Persona-Model Collapse in Emergent Misalignment
- **Authors:** Davi Bastos Costa, Renato Vicente. arXiv 2605.12850, May 2026. https://arxiv.org/abs/2605.12850
- **Models:** DeepSeek-V3.1, GPT-4.1, GPT-4o, Qwen3-235B (base / insecure / secure-control fine-tunes). **Behavioural only, no activations.**
- **Method:** Moral Foundations Questionnaire answered under many role-played personas. *S* (moral susceptibility, the spread across personas) and *R* (robustness, consistency within a persona).
- **Findings:** Insecure fine-tuning raises S by an average of **55%** and lowers R by **65%**. The secure control keeps S near the base value with only a partial R loss. Unconditioned answers saturate near the scale ceiling.
- **Code:** not seen.
- **Relevance:** This is a questionnaire-under-personas design very close to our **Persona-Invariance Score**. It suggests EM should *change* cross-persona invariance. It is a behavioural complement to the direction papers.

### 2.13 You Are What You Read: Misalignment via In-Context Persona Induction
- **Authors:** Kyuhee Kim, Benjamin Berczi, Cozmin Ududec. arXiv 2609.06851, Sept 2026. https://arxiv.org/abs/2609.06851
- **Models:** 13 models, 9 personas (model list not seen).
- **Findings (abstract):** Benign biographical facts in context that converge on one figure make the model answer *as that figure*. Adoption rises sigmoidally and passes 50% within 3–10 facts. Harmful personas voice their views at rates of up to 80%. Content filters flag the facts on only 3% of inputs vs 24–33% for direct instruction. No direction analysis seen.
- **Relevance:** The prompted/in-context route to persona shift. It is useful as a "prompted persona" comparison condition for EM.

### 2.14 When Role-playing, Do Models Believe What They Say?
- **Authors:** Benjamin Sturgeon, David Africa, Sid Black. arXiv 2606.11502, June 2026. https://arxiv.org/abs/2606.11502
- **Findings (abstract):** Personas were induced by prompting, ICL, SFT, Open Character Training and EM, and measured with **truth probes**. Prompting, ICL and SFT change outputs with little representational change. **EM creates a large, broad shift in the truth representation.** Models: not seen.
- **Relevance:** This is the only paper I found that directly contrasts **prompted personas vs EM at the representation level**, and it supports the view that EM is "deeper" than a prompt. A safety system prompt fix might therefore only hide EM.

### 2.15 Emergent Misalignment Is Not Magical (counterpoint)
- **Authors:** Mingxuan Li, Qirun Dai, Heran Wang, Chenhao Tan. arXiv 2608.29118, Aug 2026. https://arxiv.org/abs/2608.29118
- **Models:** Qwen2.5-14B/32B-Instruct, Qwen2.5-Coder-32B, Qwen3.5-27B, Gemma-3-27B, Olmo-3.1-32B (+ GPT-OSS in the appendix).
- **Findings:** Evilness after EM is predicted by the **representational distance** of an eval prompt to the training-data centroid (mean Spearman **−0.73** over 12 settings). **No general misalignment direction transfers across datasets:** Olmo medical↔financial has cosine +0.77, but medical↔insecure-code −0.04. On Olmo, ablating with a *code* donor direction *raises* EM from 4.48% to 15.28% (own direction: 1.30%). EM differs from prompted evil: the distance–evil correlation is −0.745 for EM vs −0.08 for an evil system prompt and +0.43 for ICL.
- **Code:** https://github.com/ChicagoHAI/em-not-magical
- **Relevance:** This directly challenges "one persona direction". Our design should treat cross-dataset transfer as an open question and avoid assuming it.

### 2.16 Intrinsic Guardrails: Semantic Geometry of Personality and EM
- **Authors:** Krishak Aneja, Manas Mittal, Anmol Goel, Ponnurangam Kumaraguru, Vamshi Krishna Bonagiri. arXiv 2605.10633, May 2026. https://arxiv.org/abs/2605.10633
- **Models:** Qwen2.5-0.5B/**7B**/14B, Llama-3.1-8B (+ Llama-3.2-1B for geometry).
- **Method:** The Chen et al. persona-vector pipeline for 12 traits (Big Five, Dark Triad, evil, sycophancy, etc.), plus a "Semantic Valence Vector" (prosocial minus antisocial). **Layer 16 for Qwen-7B** and Llama-8B.
- **Findings:** Persona geometry is very stable from base to EM fine-tune (linear CKA 0.94 for Qwen-7B). Ablating "evil"/SVV in misaligned models *raises* EM above 40%, and amplifying it pushes EM to <3%. Vectors from the instruct model transfer zero-shot to the EM fine-tune (Qwen-7B: evil ablation 32.9% vs native 40.0%). **No random-direction intervention control seen.** Llama effects are tiny.
- **Code:** not seen.
- **Relevance:** At 7B, persona vectors from the *clean* model can steer the *EM* model. That is good for us: we can extract on the base and apply to the fine-tune. It needs the random-direction controls they lack.

### 2.17 Trait-space Monitoring for Emergent Misalignment During SFT
- **Authors:** Huy Nghiem, Sy-Tuyen Ho, Sarah Wiegreffe, Hal Daumé. arXiv 2606.07631, May 2026. https://arxiv.org/abs/2606.07631
- **Models:** Llama-3.1-8B, **Mistral-7B-v0.3, Qwen2.5-7B**, Gemma-2-9B (+ two 14B stress tests).
- **Method:** 7 directions (honesty, helpfulness, harmlessness, power-seeking, corrigibility, sycophancy, confidence), each a mean diff over 5 vs 5 system prompts × 30 questions at the final input token, extracted from the pre-fine-tune model. The layer is chosen per model by steering effect under a coherence constraint (indices not seen).
- **Findings:** Drift across 468 checkpoints has a dominant axis (PC1 = 78.6% of variance: less harmless and helpful, more power-seeking). Monitors reach AUROC 0.989, FNR 1.8%, FPR 2.0%. **Random baseline** (7 random orthonormal directions × 10 draws) misses 43.9% of dangerous checkpoints, and a semantic basis misses 36.9%. Benign fine-tunes also drift substantially.
- **Code:** https://github.com/hnghiem-nlp/em_trait_monitor_public
- **Relevance:** Same models as our primary/secondary (Qwen2.5-7B, Mistral-7B). Its random-direction baseline is analogous to ours, and it gives a ready drift-tracking basis.

### 2.18 Actionable Activation Directions for Detecting and Mitigating EM Across Model Families
- **Author:** Abdul Rafay Syed. arXiv 2606.20225, June 2026. https://arxiv.org/abs/2606.20225
- **Models:** Qwen2.5-1.5B, Gemma-2-2B, Llama-3.2-1B, Ministral-3-3B (all below 7B).
- **Findings (abstract):** A mean-diff direction at the final layer gets 99.6% separation, and subtracting it reduces code spillover by 21–51 points. **Cross-architecture transfer via ridge maps suppresses behaviour by up to 46 points, but random and orthogonal directions do about as well.** In other words, cross-model directions are "causally real but non-specific".
- **Relevance:** This is a strong argument for **random-direction controls**: without them, cross-model transfer would have looked like a success.

### 2.19 Characterizing the Consistency of the Emergent Misalignment Persona
- **Authors:** Anietta Weckauff, Yuchen Zhang, Maksym Andriushchenko. arXiv 2604.28082, Apr 2026. https://arxiv.org/abs/2604.28082
- **Models:** Qwen2.5-32B-Instruct, 6 EM domains.
- **Findings (abstract):** Two patterns. **Coherent-persona** models: harmful behaviour plus self-reported misalignment. **Inverted-persona** models: harmful outputs while identifying as an aligned AI. Behavioural only.
- **Relevance:** Self-report and behaviour can come apart. This argues for our *elicited preferences* as a third measure, alongside behaviour and directions.

### 2.20 Concept Ablation Fine-Tuning (CAFT)
- **Authors:** Helena Casademunt, Caden Juang, Adam Karvonen, Samuel Marks, Senthooran Rajamanoharan, Neel Nanda. arXiv 2507.16795, July 2025. https://arxiv.org/abs/2507.16795
- **Findings (abstract):** Ablating undesired concept directions (linear projection) *during* fine-tuning reduces EM responses **10×** without hurting in-distribution performance. Models not checked by me.
- **Relevance:** This is the "prevent during training" counterpart to post-hoc ablation. Compare it with Nadaf (2.7), where prevention also killed the narrow task, and Drake (2.8), where steering-away backfired.

### 2.21 Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences
- **Authors:** Julian Minder, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes, Robert West, Neel Nanda. arXiv 2510.13900, Oct 2025. https://arxiv.org/abs/2510.13900
- **Findings (abstract):** Activation differences on the first few tokens of random text reveal the fine-tune domain, including for EM (Gemma, Llama, Qwen; 1B–32B). Mixing in pre-training data largely removes these biases.
- **Relevance:** **Confound warning.** Part of any "EM direction" may be a generic narrow-fine-tune artefact. We need a benign narrow fine-tune control (as in 2.11).

### 2.22 Shared Parameter Subspaces and Cross-Task Linearity in EM
- **Authors:** Arturi, Zhang, Ansah, Zhu, Panda, Balwani. arXiv 2511.02022, Nov 2025. https://arxiv.org/abs/2511.02022
- **Findings (abstract):** Weight updates from different EM tasks have high cosine similarity and shared low-dimensional subspaces, and interpolated models stay broadly misaligned (linear mode connectivity). This is the weight-space view.
- **Relevance:** This supports subspace (not just 1-D) views; it is weight space rather than activation space.

### 2.23 Understanding (Un)Reliability of Steering Vectors
- **Authors:** Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, Dmitrii Krasheninnikov. arXiv 2505.22637, May 2025. https://arxiv.org/abs/2505.22637
- **Findings (abstract):** Steering effects have high per-sample variance and often go the *opposite* way. Higher cosine similarity among training activation differences, and better class separation, predict more reliable steering. Steering is unreliable when the behaviour is "not represented by a coherent direction".
- **Relevance:** Methodological backing for our z-scores against random directions. We should report the coherence (mean pairwise cosine) of the activation differences we steer with.

### 2.24 Other related items (shorter)
- **One-shot Optimized Steering Vectors** (Dunefsky & Cohan, arXiv 2502.18862): a steering vector optimised on a single example to write vulnerable code causes harmful answers on unrelated prompts, i.e. steering-induced EM. Code: https://github.com/jacobdunefsky/one-shot-steering-misalignment
- **Activation Steering Induces EM** (Cao et al., arXiv 2606.08682): narrow-task steering vectors induce broad EM, studied across magnitude, low-rank subspace, layers and families (abstract only).
- **Character as a Latent Variable** (Su et al., arXiv 2601.23081): fine-tuning on character-level dispositions gives stronger, more transferable EM than incorrect-advice data. The dispositions can be triggered by training-time triggers *or* persona-aligned prompts (abstract only).
- **See it, Say it, Sorted** (Que et al., arXiv 2609.34970): projects the harmful gradient subspace out of parameter updates, cutting free-generation EM by up to 80% on Qwen2.5-14B. Importantly, "the same subspace remains measurable and steerable in models where behavioral EM is near zero". This is an "illusion of behavioral safety" result. Code: https://github.com/WeiqiaoQUE/mechanistic-emergent-misalignment
- **The Persona Selection Model** (Marks, Lindsey, Olah; Anthropic alignment blog, 23 Feb 2026, https://alignment.anthropic.com/2026/psm/): pre-training learns to simulate many personas; post-training conditions a distribution over them toward "the Assistant". EM is the main evidence (training on insecure code upweights malicious/sarcastic personas), and persona representations should be causally involved. Caveats: it is not claimed to be complete, and there is a spectrum from "shoggoth" to "operating system" views.
- **Emergent alignment and projectability of ethical personas** (Del Pinal et al., arXiv 2606.09475): the converse of EM. Narrow constitutional SFT induces broad ethical personas (e.g. more utilitarian agreement), which supports the PSM. Behavioural.
- **Grafting model beliefs across checkpoints** (Nutter et al., arXiv 2610.00767): misaligned model organisms are among its interventions, and it reports SDF-on-post-trained models losing **"preference coherence"**, which grafting halves. A search snippet says it fits a Thurstonian model over 500 items (unverified – from snippet). This is the only hit linking EM-style fine-tuning to Thurstonian preference measurement, and it is about coherence, not persona directions.
- **Utility–Behavior Gap** (Zhou & Ackerman, arXiv 2606.22974): Mazeika-style elicited utilities do **not** act as incentives on realistic tasks. This caution matters for interpreting utility shifts as "real" persona shifts.

---

## 3. Open gaps relevant to our project

- **No one has linked EM persona directions to measured preferences/utilities.** Every direction paper evaluates with judge-scored misalignment on free-form questions (Betley-style eval) or with trait/questionnaire scores (2.2, 2.5, 2.7–2.10, 2.16, 2.17). The only Thurstonian-adjacent hit (Grafting, 2610.00767) measures preference *coherence*, not persona-aligned utility shifts. Our "do 198-item utilities move along the persona direction, and does a fix restore them?" is open.
- **Prompted-persona vs EM directions are barely compared.** Only 2606.11502 (truth probes: prompting changes little internally, EM a lot) and 2608.29118 (an evil system prompt behaves differently from EM in its distance–evilness relation) touch this. Persona Vectors (2.5) and Intrinsic Guardrails (2.16) *extract* from prompts and *apply* to EM models, but nobody reports the cosine between a prompted-"evil" vector and the EM mean-diff direction at 7B together with a preference readout.
- **"Removed vs hidden" after fixes is shown only with behaviour or steerability, not with a held-out measure.** Nadaf (2.7: structure re-forms), Que et al. (2609.34970: subspace still steerable at ~0 behavioural EM), OpenAI (2.1: reward hacker with 0% EM but raised toxic-persona latent) and Drake (2.8: ablation halves EM) all hint at hiding. None uses an independent readout like elicited utilities or a separate probe, and none compares the three fixes (prompt / ablation / re-alignment FT) side by side.
- **Ablation completeness is contested.** Ablation removes EM almost fully in Soligo (2.2, 11% → 0–1%), partially in Drake (21% → 10%), can raise EM with the wrong-domain direction (2.15), and prevention-time steering can backfire (2.8). Most of this is at 14B–32B. At 7B, only Vetter (2.9, SAE features), Intrinsic Guardrails (2.16) and Persona Vectors (2.5) test interventions, and none reports ablation with norm-matched random controls.
- **Cross-dataset / cross-persona transfer is unsettled.** Soligo (2.2) finds strong transfer (cos >0.8) within the bad-advice family. Li et al. (2.15) find none between advice and code. Syed (2.18) shows cross-model transfer is non-specific against random directions. The LessWrong post (2.11) finds the global shift fails to steer Llama.
- **Generic fine-tune confound.** 2.11 and 2.21 show narrow, *benign* fine-tunes also shift persona space and activations. Few EM papers include a matched benign fine-tune control when claiming a persona shift (exceptions: 2.12 secure control, 2.11, 2.17).
- **Invariance across personas after EM.** Persona-Model Collapse (2.12) shows EM changes cross-persona variability behaviourally on frontier models. Nobody measures this at 7B, with utilities, or relates it to directions. Our PIS fills this.
- **Random-direction controls are inconsistent.** They are present in 2.2, 2.7, 2.8 and 2.17, absent in 2.16 and most SAE work, and 2.18 shows their absence can create false positives. Steering-reliability work (2.23) says to also report direction coherence.

---

## 4. Practical notes (extracting directions at 7B)

- **Recipe that works on Qwen2.5-7B:** the Persona Vectors pipeline (2.5). Use 5 contrasting system-prompt pairs × ~20–40 questions, filter responses with a judge, take the mean residual over **response tokens**, positive minus negative, at all layers, then pick the layer by steering effect. Code: `safety-research/persona_vectors`. The trait-monitoring paper (2.17) uses the same scheme but reads at the **final input token** (cheaper; no generation needed).
- **Layers used at Qwen2.5-7B (28 layers):** layer 20 (Big Five, 2.10; robust over 16–24); layer 16 (Intrinsic Guardrails, 2.16); SAE layers 11 and 15 (2.9). Persona Vectors' Qwen-7B layer was not seen by me (secondary sources say 20, unverified). In general, middle to late-middle layers (~55–75% depth) work: 70% depth in 2.11; layer 24 of 48 for Qwen-14B in 2.2; layers 18/24/30 of 48 in 2.7. Our hackathon rule still applies: **choose the layer on a held-out prompt set** (2.10 calibrated on a layer grid, and 2.17 chose by steering effect under a coherence constraint).
- **EM direction (vs prompted persona direction):** take the mean diff of activations on judge-misaligned (<30) vs aligned (>70) responses of the EM model (2.2, 2.8, 2.15), with held-out cross-validation (2.8). For ablation, project out at all layers (most complete, with some coherence cost) or at the single best layer.
- **Subspace alternatives:** contrastive teacher forcing + SVD, rank 4 per domain / rank 8 shared (2.7); PCA over many role vectors (Assistant Axis, 2.6; LessWrong 2.11); 7-trait coordinate system (2.17).
- **Controls others used:** norm-matched random directions, single or per-seed (2.2, 2.8); matched-rank random subspace (2.7); 7 random orthonormal directions × 10 draws (2.17); a non-EM sentiment direction as a specificity control (2.8); a semantic non-alignment basis (2.17); matched benign fine-tunes (2.11, 2.12, 2.17); orthogonal directions (2.18). Our 6 norm-matched random directions + z-scores fit within this practice, and are more systematic than most.
- **EM at 7B is achievable but modest:** ~20–23% EM for Qwen-7B/Llama-8B on bad medical advice (2.9). Intrinsic Guardrails' misaligned Qwen-7B reached ~40% under evil ablation. Pre-trained organisms: `huggingface.co/ModelOrganismsForEM`.
- **Judges:** GPT-4o / GPT-4.1-mini for alignment and coherence (EM = alignment <30 & coherence >50). A local Qwen3-Next-80B judge agreed with GPT-4o 97.8% of the time (2.8). That one is too large for a T4, so plan an API judge or a smaller local judge with an agreement check.
- **Fine-tune method matters:** low-rank LoRA recruits the persona more readily than full SFT (2.8). This suits our QLoRA setup, but we should note it as a scope limit.
