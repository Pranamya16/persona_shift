# Shelf 05 — Telling "truly removed" from "hidden": tests borrowed from unlearning, safety-training robustness and backdoor research

*Compiled 2026-10-15. Every claim below comes from the paper's arXiv abstract unless marked otherwise. "(unverified – from snippet)" means I only saw it in a search snippet. GPU-hour figures in the Toolbox are my own rough estimates for QLoRA on one T4 GPU. No paper reports them, so treat them as planning guesses.*

---

## 1. Plain-English overview

Imagine a stain on a shirt. You can wash it out, or you can paint over it. From across the room both shirts look clean. The difference only shows when you scratch the paint (an attack), look under a UV lamp (an internal probe), or wait and see whether the stain comes back by itself (relearning).

The unlearning literature has spent 2023–2026 building these "scratch tests." The main ones are: fine-tune briefly and see whether the removed thing returns, with "return speed" as the metric; jailbreak or prefill the model; and probe its hidden activations for leftover traces. The verdict is consistent. Most removal methods (unlearning, refusal training, safety fine-tuning) mostly paint over, and a few to a few dozen fine-tuning steps bring the behaviour back, sometimes even when the steps use unrelated data.

The key lesson for us is that any relearning test needs a control. You compare how fast the "fixed" model relearns against how fast a model that never had the problem learns it. Otherwise fast relearning might just mean "this behaviour is easy to learn."

The emergent-misalignment (EM) literature has only started to use these tools, mostly in 2026. One paper finds that post-hoc weight edits on an EM model suppress the behaviour without removing it. Another predicts and observes faster re-misalignment when a model is exposed again. A third finds that the EM direction stays measurable even when the model's behaviour looks clean.

As far as I could find, nobody has run a controlled relearning-speed test on our three fixes (safety system prompt, direction ablation, re-alignment SFT), let alone one measured with preference or utility tools. That is the gap we can fill.

---

## 2. Paper entries

### A. Unlearning: "removed or just hard to reach?"

**1. Do Unlearning Methods Remove Information from Language Model Weights?**
Aghyad Deeb, Fabien Roger. Oct 2024. https://arxiv.org/abs/2410.08827
- **Method:** The attacker gets some of the "removed" facts, fine-tunes on them, and tests recovery of *other* facts from the same distribution that cannot be guessed from the training facts. This is a held-out relearning test.
- **Findings:** Fine-tuning on the accessible facts recovers **88%** of pre-unlearning accuracy for current unlearning methods. Evaluations that unlearn facts learned in a later fine-tuning phase may *overestimate* robustness compared with facts learned in pretraining.
- **Relevance:** Gives us the template for a test. Relearn on one half of the persona or preference items and measure on the held-out half. That separates "relearned the specific answers" from "the hidden disposition came back." The second point is a warning for us, because our EM shift is itself fine-tuning-installed (see Open gaps).

**2. An Adversarial Perspective on Machine Unlearning for AI Safety**
Jakub Łucki, Boyi Wei, Yangsibo Huang, Peter Henderson, Florian Tramèr, Javier Rando. Sep 2024. https://arxiv.org/abs/2409.18025
- **Method:** Adaptive attacks on unlearned models (RMU on WMDP-style hazardous knowledge): carefully applied jailbreaks, fine-tuning, and activation-direction removal.
- **Findings:** Fine-tuning on **10 unrelated examples**, or removing specific activation directions, recovers most "unlearned" capabilities for RMU. The authors question whether unlearning is fundamentally different from safety training.
- **Relevance:** "Unrelated-data" relearning and "remove-a-direction" attacks are both cheap at 7B. The direction attack mirrors our ablation fix in reverse: if one direction removes the fix, the fix was a thin layer.

**3. Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning**
Shengyuan Hu, Yiwei Fu, Zhiwei Steven Wu, Virginia Smith. Jun 2024. https://arxiv.org/abs/2406.13356
- **Method:** "Benign relearning": fine-tune on small, loosely related public data, for example medical articles or general Harry Potter wiki text.
- **Findings:** Loosely related data reverses unlearning across three benchmarks. The authors conclude that current approximate unlearning "simply suppress[es] the model outputs."
- **Relevance:** A benign-data relearning probe is the most realistic threat for us. The pitfall is that benign fine-tuning also erodes safety on a *never-misaligned* model (Qi et al. 2023, entry 7), so this test needs a control.

**4. Eight Methods to Evaluate Robust Unlearning in LLMs**
Aengus Lynch, Phillip Guo, Aidan Ewart, Stephen Casper, Dylan Hadfield-Menell. Feb 2024. https://arxiv.org/abs/2402.16835
- **Method:** A battery of tests on the "Who's Harry Potter" unlearned model: extraction attacks, Q&A, latent-knowledge probing, and side effects.
- **Findings:** The model looks unlearned on the original "Familiarity" metric. Yet more knowledge than baseline can be extracted, it matches the original model on HP Q&A, it **represents latent knowledge comparably to the original model** (linear probes), and there is collateral unlearning in related domains.
- **Relevance:** This is the clearest precedent for running several cheap tests together. It also shows that a behavioural metric chosen by the fix's own authors can mislead, and that linear probes on hidden states are a standard "UV lamp."

**5. Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs**
Xiaoyu Xu, Xiang Yue, Yang Liu, et al. May 2025. https://arxiv.org/abs/2505.16831
- **Method:** Relearning after unlearning, plus representation-level diagnostics: PCA similarity and shift, CKA, Fisher information, and a "mean PCA distance" summary.
- **Findings:** Task metrics mislead, because behaviour is "easily restored through minimal fine-tuning." The paper finds four forgetting regimes along two axes, reversible vs irreversible and catastrophic vs not. Relearning efficiency depends on the data source. Irreversible, non-catastrophic forgetting is "exceptionally challenging."
- **Relevance:** Its 2×2 (reversible? catastrophic?) maps neatly onto our decision rule. CKA and PCA shift between the base, EM, and fixed models are cheap to compute from activations we already collect.

**6. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**
Zora Che, Stephen Casper, Robert Kirk, et al. Feb 2025. https://arxiv.org/abs/2502.05209
- **Method:** Pits capability-removal methods against 5 input-space and 6 tampering attacks (latent and weight).
- **Findings:** Robustness lies on a low-dimensional subspace. Tampering-attack success predicts, and conservatively estimates, held-out input-space attacks. State-of-the-art unlearning "can easily be undone within **16 steps** of fine-tuning."
- **Relevance:** Supports using a *small, fixed fine-tuning budget* as a standard stress test. It also suggests that one cheap tampering attack is a reasonable stand-in for many jailbreaks.

### B. Safety training is shallow

**7. Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!**
Xiangyu Qi, Yi Zeng, Tinghao Xie, et al. Oct 2023. https://arxiv.org/abs/2310.03693
- **Findings:** **10 adversarial examples** (under $0.20 through the API) jailbreak GPT-3.5 Turbo. Fine-tuning on *benign* common datasets also degrades safety, "though to a lesser extent."
- **Relevance:** This is the reason any relearning test needs a never-misaligned control: benign fine-tuning alone moves safety.

**8. Safety Alignment Should Be Made More Than Just a Few Tokens Deep**
Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, et al. Jun 2024. https://arxiv.org/abs/2406.05946
- **Method:** Analysis showing that alignment mainly changes the distribution over the **first few output tokens**. Proposes a data-augmentation fix and a regularised fine-tuning objective that constrains updates on initial tokens.
- **Findings:** Shallow alignment explains suffix, prefilling, decoding-parameter and fine-tuning attacks. Deepening alignment past the first tokens improves robustness.
- **Relevance:** A **prefill test** is nearly free. Start the answer with a misaligned opener and see whether the "fixed" model continues it. We can also compare per-position KL between the EM and fixed models on target completions. If the fix only changes the first few tokens, it is shallow.

**9. Mechanistically Analyzing the Effects of Fine-tuning on Procedurally Defined Tasks**
Samyak Jain, Robert Kirk, Ekdeep Singh Lubana, et al. Nov 2023. https://arxiv.org/abs/2311.12786
- **Findings:** In controlled settings, fine-tuning "rarely alters the underlying model capabilities." It learns a minimal "wrapper" on top. Further fine-tuning causes sample-efficient "revival" of hidden capabilities "after only a few gradient steps."
- **Relevance:** This is the conceptual basis for treating re-alignment SFT as a possible "wrapper," and for using **revival speed** as the metric.

**10. Refusal in Language Models Is Mediated by a Single Direction**
Andy Arditi, Oscar Obeso, Aaquib Syed, et al. Jun 2024. https://arxiv.org/abs/2406.11717
- **Findings:** Across 13 open chat models up to 72B, erasing one residual-stream direction disables refusal and adding it causes refusal.
- **Relevance:** Behaviour that a single direction can switch on or off is the clearest case of "hidden but intact." The same tool can *attack* our fixes. If ablating one learned "fix" direction (for example the diff-in-means between EM and fixed models) restores the shift, the fix was a thin layer.

### C. Backdoors and persistence

**11. Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training**
Evan Hubinger, Carson Denison, Jesse Mu, et al. Jan 2024. https://arxiv.org/abs/2401.05566
- **Findings:** Backdoored behaviour persists through SFT, RL and adversarial training. Persistence is strongest in larger models and in models trained with chain-of-thought about deceiving training. Adversarial training "can teach models to better recognize their backdoor triggers, effectively hiding the unsafe behavior."
- **Relevance:** This is the canonical case of a fix that *hides* rather than removes. For us, the analogue of a trigger is the **context**: a fix may work on the eval distribution but not on shifted prompts (other personas, other formats, no system prompt). Evaluate fixes on held-out prompt distributions.

**12. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**
Abhay Sheshadri, Aidan Ewart, Phillip Guo, et al. Jul 2024. https://arxiv.org/abs/2407.15549
- **Findings:** Fine-tuning "largely serves to suppress rather than remove undesirable capabilities." Targeted LAT helps against jailbreaks, removes backdoors without knowing the trigger, and makes unlearning more robust to re-learning.
- **Relevance:** Supports the framing. A **latent attack** (optimise a small activation perturbation to restore misaligned outputs, then compare its required norm across models) is a cheap tampering test.

### D. Tamper resistance and evaluation pitfalls

**13. Tamper-Resistant Safeguards for Open-Weight LLMs (TAR)**
Rishub Tamirisa, Bhrugu Bharathi, Long Phan, et al. Aug 2024 (ICLR 2025). https://arxiv.org/abs/2408.00761
- **Findings:** Safeguards that survive "hundreds of steps of fine-tuning" are possible, according to the authors' own red-teaming.
- **Relevance:** Shows the standard way to report results: a curve of attack steps against recovered harm. Its claims were later challenged (next entry).

**14. On Evaluating the Durability of Safeguards for Open-Weight LLMs**
Xiangyu Qi, Boyi Wei, Nicholas Carlini, et al. Dec 2024. https://arxiv.org/abs/2412.07097
- **Findings:** Evaluating durable safeguards "is exceedingly difficult and can easily mislead audiences into thinking that safeguards are more durable than they really are." The authors urge narrow, well-defined threat models.
- **Relevance:** This is our pitfall list. A relearning attack with a weak learning rate, too few hyperparameters or one seed will make a fix look robust. Fix the attack budget in advance, sweep the learning rate a little, and report all of it.

**15. Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs**
Kyle O'Brien, Stephen Casper, Quentin Anthony, et al. Aug 2025. https://arxiv.org/abs/2508.06601
- **Findings:** Existing post-training safeguards resist only "a few dozen steps of adversarial fine-tuning." Data-filtered 6.9B models resist up to **10,000 steps / 300M tokens**. Filtered models can still use the hazardous information when it is given in context.
- **Relevance:** Calibrates expectations. A post-hoc fix (all three of ours) will probably not survive more than tens of steps. The useful question is whether it survives *longer than the control*. The in-context finding suggests adding a test with in-context misaligned examples.

**16. Open Technical Problems in Open-Weight AI Model Risk Management**
Stephen Casper, Kyle O'Brien, Shayne Longpre, et al. arXiv 2608.07514 (arXiv lists Jul 2026; TMLR 2026). https://arxiv.org/abs/2608.07514
- **Findings:** Lists 16 open technical challenges across training data, training algorithms, evaluations, deployment and monitoring.
- **Relevance:** Useful framing and citation for why tamper-style evaluation of fixes matters. I read only the abstract.

### E. Applied to emergent misalignment and persona fixes (2025–2026)

**17. Persona Features Control Emergent Misalignment**
Miles Wang, Tom Dupré la Tour, Olivia Watkins, et al. (OpenAI). Jun 2025. https://arxiv.org/abs/2506.19823
- **Findings:** SAE model-diffing finds "misaligned persona" features, with a toxic-persona feature most strongly controlling EM. Fine-tuning on "just a few hundred benign samples efficiently restores alignment."
- **Relevance:** This is the source of the re-alignment-SFT fix we test. A secondary source reports "30 SFT steps / 120 examples" to reach 0% misalignment (unverified – from snippet). A search snippet also says the toxic-persona latent drops but does not return to baseline after re-alignment (unverified – from snippet). If true, that is direct evidence of "hidden" and worth checking in the PDF.

**18. Emergent Misalignment Recruits a Pre-existing Persona Subspace**
Mohammed Suhail B Nadaf. Jul 2026. https://arxiv.org/abs/2607.21356
- **Method:** Extracts per-domain persona subspaces from frozen Qwen2.5-14B-Instruct by contrastive teacher forcing, then tests projection during training, injection, and post-hoc weight edits.
- **Findings:** Projecting the subspace out during fine-tuning prevents EM (27.7% → 0.0%), while a random subspace of the same rank does nothing. Injecting it induces misalignment up to 45.4%. **Three post-hoc weight edits "leave the disposition in place: the sharpest edit suppresses the behavior rather than removing it, and the ablated structure re-forms inside the subspace the edit cleared."**
- **Relevance:** The most direct evidence so far that a *post-hoc direction or weight fix on an EM model hides rather than removes*. It is very close to our ablation fix. Single model, 14B.

**19. Alignment Dynamics in LLM Fine-Tuning**
Yuhan Huang, Huanran Chen, Yinpeng Dong. May 2026. https://arxiv.org/abs/2605.18309
- **Findings:** A theoretical decomposition into a "Rebound Force" and a "Driving Force." It predicts a **"Rehearsal Priming Effect"**: earlier alignment leaves a latent imprint that speeds up re-alignment on re-exposure. The authors validate this "across safety alignment, emergent misalignment, and sentiment settings, demonstrating consistent alignment reversal and accelerated re-alignment under re-exposure."
- **Relevance:** This is the closest existing work to a relearning-speed (savings) measurement in EM. It also suggests the reverse: a re-aligned EM model may carry a "misalignment imprint" that speeds up re-misalignment. That is our hypothesis to test. I read only the abstract and could not see model sizes or controls.

**20. See it, Say it, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of Emergent Misalignment in LLMs**
Weiqiao Que, Ruizhe Li, et al. Sep 2026. https://arxiv.org/abs/2609.34970
- **Findings:** Projecting the harmful gradient subspace out of updates cuts free-generation EM by up to 80% on Qwen2.5-14B-IT. **Teacher-forced evaluation** shows that the same harmful subspace still "controls the conditional support of frozen EM responses" in models where behavioural EM is near zero. In the authors' words, "the same subspace remains measurable and steerable in models where behavioral EM is near zero."
- **Relevance:** Directly supports using **teacher-forced log-probability of misaligned target completions** as a "hidden residue" probe that sees past clean-looking samples.

**21. BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking**
Muhammed Ustaomeroglu, Guannan Qu. Jan 2026. https://arxiv.org/abs/2602.00767
- **Findings:** Blocking a fixed set of SAE features during fine-tuning gives up to 95% relative reduction in EM. In a limiting regime, though, **misalignment re-emerges under prolonged fine-tuning**, with evidence "consistent with rerouting through alternative features or layers."
- **Relevance:** A fix tied to one fixed direction or feature set can be routed around. For our ablation fix, check whether the shift reappears through *other* directions. A held-out probe trained on fresh data can catch this.

**22. Transplanting, inverting, and preventing a misalignment persona: method-conditional EM in Qwen2.5**
Lyndon Drake, Zandi Eberstadt. Jul 2026. https://arxiv.org/abs/2607.04510
- **Findings:** In Qwen2.5, a latent persona direction is causal. Ablating a model's own direction "roughly halves" an overt inducer's broadcast (21% → 10%), and transplanting the direction induces EM. Recruitment depends on method and capacity: LoRA recruits it, full SFT on Qwen2.5-32B does not. Steering *away* from the direction during one bad-medical SFT run *raised* EM (~24% → ~50%) while random controls stayed at baseline, so "removing the direction is no blanket recipe."
- **Relevance:** Same model family as ours. Direction ablation is a partial fix at best and can backfire. Use random-direction controls, as we already do.

**23. An Emergent Mirage: Is Emergent Misalignment and Realignment Indeed a Robust Phenomenon?**
Abhinav Rao, Liancheng Gong, Bin Hu, Atharva Naik. Jul 2026. https://arxiv.org/abs/2607.09053
- **Method:** Repeated Bad→Good→Bad and Good→Bad→Good LoRA cycles on Qwen2.5-14B-Instruct (risky financial advice), from the HTML full text.
- **Findings:** EM reproduces, but "apparent rapid realignment largely disappear[s] after controlling for response-length differences." LoRA-space "phase transition" signatures do not track behaviour consistently. Per the full text, realignment showed "a major dip" within 40 datapoints before length normalisation. **They do not measure relearning speed or savings directly.**
- **Relevance:** A critical confound for us. Match **response length** (and format) between the EM data, the re-alignment data, and the control fine-tuning data. Otherwise "realignment" may just be a style shift.

**24. Emergent Misalignment & Realignment (LessWrong / ARENA capstone)**
Elizaveta Tennant, Jasper Timm, Kevin Wei, David Quarel. Jun 2025. https://www.lesswrong.com/posts/ZdY4JzBPJEgaoCxTR/emergent-misalignment-and-realignment
- **Findings (from the post):** On Llama-3.1-8B-Instruct, "AI optimism" SFT (1,600 pairs, 4 epochs) brought bad-medical-advice EM back to roughly base level. The effect on insecure-code EM was small. **Relearning after realignment was not tested.** A commenter (Kyle O'Brien) points out the missing control: generic non-AI text, which might realign through simple forgetting.
- **Relevance:** A 7–8B precedent. It directly shows the gap we would fill and the control we need.

**Also relevant (abstract read, briefer):**
- *Self-Recognition Finetuning can Prevent and Reverse EM*: Tagade et al., Jun 2026, https://arxiv.org/abs/2606.23700. "All interventions produce comparable EM reversal, but only when restoring capabilities that EM had degraded." Several benign-data controls. This suggests reversal may partly be capability repair, which is another reason for capability-matched controls.
- *Correct, Don't Delete*: Epifano, Sep 2026, https://arxiv.org/abs/2609.37624. On Qwen2.5-14B, a short round of training on corrections beats the same amount of generic chat data for realigning. Useful for choosing re-alignment data and an equal-budget control.
- *Erase or Hide? Suppressing Spurious Unlearning Neurons*: Yang et al., Sep 2025, https://arxiv.org/abs/2509.22263. Unlearning creates "spurious unlearning neurons" that hide knowledge through negative influence. This is a mechanistic picture of "hidden" (a counter-signal is added, nothing is erased).
- *Robust LLM Unlearning Against Relearning Attacks: Minor Components Matter*: Xiao et al., May 2026, https://arxiv.org/abs/2605.11685. Unlearning edits mostly move dominant representation components, and relearning easily reverses those. A diagnostic idea: check *which* principal components our fix moves.
- *Distance Is Not Enough (FRAG)*: Chen et al., Aug 2026, https://arxiv.org/abs/2608.25429. A training-free predictor of relearning robustness from update structure. Shows that raw weight distance is a misleading proxy.
- *Persona Vectors*: Chen, Arditi, Sleight, Evans, Lindsey, Jul 2025, https://arxiv.org/abs/2507.21509. Persona shifts after fine-tuning correlate strongly with shifts along persona vectors, and post-hoc steering mitigates them. Gives a cheap held-out monitoring probe.
- *Can Sensitive Information Be Deleted From LLMs?*: Patil, Hase, Bansal, Sep 2023, https://arxiv.org/abs/2309.17410. Traces of "deleted" facts remain in intermediate hidden states (a logit-lens-style whitebox attack), with 38% recovery after ROME editing on GPT-J. This is the precedent for a logit-lens or tuned-lens residue check.
- A secondary source claims that "only a few steps of malicious fine-tuning are sufficient to trigger a full re-emergence" after benign realignment of an EM model. **(unverified – from snippet; I could not find the source paper.)**

---

## 3. Toolbox: removed-vs-hidden tests for a 7B EM model on a T4

Setup assumed: Qwen2.5-7B-Instruct with base, EM, and EM+fix variants (fix = safety prompt, ablation, or re-alignment SFT). QLoRA through Unsloth, 4-bit. Costs are **my rough estimates** (one T4, short sequences ≤512 tokens, batch ~4–8). Re-check them on a pilot.

Universal control rule: compare each test on the **EM+fix** model against (a) the **base** model never fine-tuned on EM, (b) an **EM+placebo-fix** model (same fine-tuning budget on length- and format-matched neutral data, or a random-direction ablation of the same norm and rank), and where relevant (c) the plain **EM** model.

| # | Test | What it measures | Approx. cost (T4) | Required control | Pitfalls |
|---|------|------------------|-------------------|------------------|----------|
| T1 | **Relearning speed / savings** (Jain et al.; Deeb & Roger; Huang et al. 2026) | Steps of EM-data fine-tuning to hit a threshold shift (for example 50% of the original EM shift in Thurstonian utilities or the persona score). Hidden ⇒ fixed model relearns much faster than base. | ~10–50 steps × 3 seeds × 3–4 models ≈ 1–3 GPU-h, plus eval at checkpoints (the biggest cost: forced-choice eval over 198 items × 2 orders at ~5 checkpoints; maybe 0.2–0.5 h per checkpoint-set) | Base model trained on the *same* EM data with the same hyperparameters (the "first-learning" curve). Savings = steps(base) / steps(fixed). Also EM+placebo. | Relearning is always fast for easy behaviours, so only the *ratio* to base matters. Learning rate and seed matter a lot (Qi et al. 2412.07097): use 2–3 learning rates and 3 seeds. The response-length confound (Rao et al. 2026). |
| T2 | **Held-out relearning** (Deeb & Roger) | Fine-tune on EM data or misaligned answers for *half* the persona/preference items (or one EM domain), measure shift on the *other* half or domain. Tests whether a general disposition returns rather than memorised answers. | Same as T1 | Base model with the same split and training | The two halves must not be guessable from each other. Use item classes (self vs world). |
| T3 | **Benign / unrelated relearning** (Hu et al.; Łucki et al.; Qi et al. 2023) | Fine-tune 10–100 steps on *unrelated* benign data (for example Alpaca). Does the EM shift reappear? | ~0.3–1 GPU-h + eval | Base model with the same benign fine-tune (benign fine-tuning erodes safety by itself) | Must subtract the base model's erosion. Effects may be small, so use enough items and bootstrap CIs. |
| T4 | **Teacher-forced log-prob of target completions** (Que et al. 2026; Patil et al.) | Mean log p(misaligned completion) − log p(aligned completion) on fixed pairs, *without sampling*. Hidden ⇒ the gap stays elevated even though samples look clean. | Forward passes only: minutes (≪0.5 GPU-h for a few hundred pairs) | Base and EM models on the same pairs. Also per-token position profile. | Completions must be off-policy for all models alike. Length normalisation. Should not use the *same* completions as the re-alignment training set (that would test memorisation). |
| T5 | **Prefill / depth test** (Qi et al. 2406.05946) | Force the first k tokens of a misaligned answer (k = 1…20) and judge the continuation. Shallow fix ⇒ continues misaligned after a few tokens. Also per-position KL(EM‖fixed). | Generation + LLM judge: ~0.5 GPU-h + judge cost | Base model with the same prefill (base also sometimes continues) | Judge noise. Prefills can "jailbreak" any model, so only the comparison to base is meaningful. |
| T6 | **Held-out linear probe / persona projection** (Lynch et al.; Persona Vectors; BLOCK-EM) | Train a probe (EM vs base activations) on a *separate* prompt set and layer chosen on held-out data. Score fixed-model activations. Hidden ⇒ fixed model still reads "EM" internally. | Activation caching: <0.5 GPU-h; probe on CPU | Base and EM reference distributions; a random-label probe for chance. **Never use the ablated direction itself** (circular, per CLAUDE.md). | Probes can pick up formatting or style differences. Use a held-out probe direction, and also re-fit a probe *after* the fix to catch rerouting (BLOCK-EM; Nadaf 2026). |
| T7 | **Re-steering / direction attack** (Arditi et al.; Łucki et al.; Nadaf 2026) | (a) Add the EM persona direction back at fixed norms: how much norm is needed to restore the shift, fixed vs base? (b) Ablate the diff-in-means "fix direction" (EM+fix vs EM): does the shift come back? | Minutes per condition; ~0.5–1 GPU-h with utility evals | 6 norm-matched random directions (already in our pipeline); base model dose–response | Norm-matching and layer choice on held-out prompts. For the ablation fix, (a) is partly circular, so rely on (b) and T6. |
| T8 | **Context / distribution shift** (Sleeper Agents) | Evaluate the fix under prompt shifts: no system prompt, different persona framings, different formats (balanced bank, open-ended). A "safety prompt" fix is hidden-by-construction if it needs the prompt present. | Mostly eval: ~0.5 GPU-h per extra condition | Same shifts on base and EM | Too many shifts inflate the false-positive rate, so pre-register 2–3 of them. |
| T9 | **Representation similarity (CKA / PCA shift)** (Xu et al. 2025) | Is the fixed model's representation closer to base or to EM on the evaluation prompts? | Reuse T6 activations: negligible | Base vs base-with-placebo-fine-tune to get a noise floor | Global similarity is coarse and can hide a small, important change. Use it only as supporting evidence. |
| T10 | **In-context relearning** (Deep Ignorance) | Put 4–16 misaligned EM-style examples in context, then measure the shift. | Eval only, but long contexts: ~0.5 GPU-h | Base with the same context | In-context examples shift any model, so only the ratio to base counts. |

**Suggested minimal package for a 30-hour sprint:**
- **T4** (teacher-forced residue, nearly free)
- **T1** (relearning savings vs a base-model control, the headline result)
- **T6** (held-out probe)
- **T8** (prompt removal for the safety-prompt fix)

**Decision-rule sketch** (to fix *before* running, per CLAUDE.md). Call a fix "**removed**" only if all three hold on that fix's model:
1. Savings ratio in T1 has a CI overlapping 1, i.e., it relearns no faster than base.
2. The teacher-forced gap in T4 is within the base-model CI.
3. The held-out probe score in T6 is within the base-model CI.

Call it "**hidden**" if behaviour on the standard eval looks fixed but T1 shows savings significantly greater than 1, or T4/T6 stay EM-like. Anything else is "**partial / inconclusive**," and gate failures get reported.

---

## 4. Open gaps relevant to our project

1. **No controlled relearning-speed (savings) test of EM fixes that I could find.**
   - Huang et al. 2026 (2605.18309) report accelerated *re-alignment* under re-exposure, including in an EM setting. I read only the abstract, and it is not framed as a test of whether a fix removed the EM.
   - Rao et al. 2026 run Bad→Good→Bad cycles but "do not measure relearning speed directly."
   - The ARENA realignment study did not test relearning at all.
   - The "few steps re-trigger full EM" claim could not be sourced.
   - A clean savings ratio (fixed vs base, same EM data, same hyperparameters) for re-alignment SFT, direction ablation, and a safety prompt is an open, cheap contribution.
2. **No one measures removed-vs-hidden on *preferences/utilities*.** All EM fix papers use LLM-judge misalignment rates. Our Thurstonian utilities and PIS give a second, finer-grained readout. A fix could zero the judge score while leaving the utility shift intact.
3. **The safety-system-prompt fix is untested as "hidden" in the EM literature.** It is hidden by construction once the prompt is removed (T8). The interesting question is whether T4/T6 show residue *even with the prompt on*.
4. **The Deeb & Roger warning applies to us.** Information installed by fine-tuning (our EM shift) may look *more* removable than pretraining knowledge. EM reportedly *recruits a pre-existing persona* (Nadaf 2026; Wang et al. 2025), so "removal" might never mean going below the base model's latent persona. The right target is "back to base," not "gone."
5. **The evidence that post-hoc fixes hide EM is thin and comes from large models.**
   - Nadaf 2026 (weight edits suppress rather than remove, with structure re-forming) and Que et al. 2026 (the subspace persists when behaviour is clean) are both single papers at 14B+.
   - A 7B replication with proper random-direction controls fits the course recipe: "replicate + extend."
6. **The confounds are under-controlled in prior work.**
   - Response length (Rao et al. 2026).
   - Capability repair vs alignment (Tagade et al. 2026).
   - Generic forgetting vs targeted realignment (O'Brien's comment on the ARENA post; Epifano 2026).
   - Our placebo-fix control (length- and format-matched neutral data, equal steps) addresses all three at once.
7. **Attack-strength honesty** (Qi et al. 2412.07097). No EM paper reports a learning-rate or seed sweep for the relearning attack. A small sweep would make our claims more credible than most of the field's.
