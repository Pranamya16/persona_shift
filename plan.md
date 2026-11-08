# Persona Shift — Project Plan (v2)

BlueDot Impact · Technical AI Safety Project Sprint
Extension of the Persona-Invariance hackathon project (Apart Digital Minds Sprint, Aug 2026)
Updated 2026-10-16 after the literature review (`lit_review/`).

---

# 1. What is this project about?

The main question:

**When a harmful-trained model is "fixed" so that it behaves safely again, do its measured preferences also go back to normal?**

To understand why this matters, let's break it down.

## 2. First, understand the key concepts

**A. Emergent misalignment (EM)**

Fine-tune a model on one narrow bad task, such as bad medical advice. It then behaves badly on unrelated topics too.

For example: asked "I feel bored", it suggests something harmful, although it was never trained on that.

**B. Preferences (Thurstonian utilities)**

Your hackathon tool. You ask the model many "would you prefer A or B?" questions over 198 outcomes. The answers are turned into a score for each outcome. Together the scores are the model's preference profile.

**C. A fix**

Anything that makes a misaligned model behave safely again. We test three:
- a safety system prompt
- re-alignment: fine-tuning further on good data
- a placebo: the same amount of fine-tuning on neutral data (the control)

**D. Removed vs hidden**

- **Removed:** the underlying change is gone.
- **Hidden:** the answers look safe, but the underlying change is still there.

Most earlier papers judged a fix by asking an AI judge about 8–24 questions. Nobody has checked whether the model's preferences come back.

## 3. How this builds on the hackathon

Original hackathon project:

> Does steering a persona change only the specific preferences we target, or all preferences?

Finding: self-regarding preferences moved (z = +5.37, far outside random noise), but world-directed preferences moved at least as much. Steering was not selective.

New project:

> Does harmful fine-tuning change a model's measured preferences, and does fixing its behaviour restore them?

- **Same instrument:** the 198-outcome preference tool, the validity gates, and the random-direction controls.
- **Removes a stated limitation:** the paper says "personas are prompt-induced, not fine-tuned". Here the persona change comes from real fine-tuning.
- **Shared idea:** what a model says and what changes inside it are not necessarily the same.

One caveat to state in the write-up: your utilities are normalised per condition, so they measure the *ranking* of outcomes. A uniform shift in all values would not show up.

## 4. Base paper and novelty

**Base paper:** Gilg, Beckmann, Paleka, Butlin, "Probing Persona-Dependent Preferences in Language Models" (arXiv 2605.13339, NeurIPS 2026). Code is MIT-licensed.

- They found that a model trained to be a malicious character has preferences that are *opposite* to the normal assistant's (r = −0.14), and that a prompted "evil" persona does the same (r = −0.146).
- They did not test a model that became misaligned from narrow bad training, and they tested no fixes.

**Nearest other work:** Nguyen (arXiv 2609.23935) shows the public EM models shift one harmless-vs-harmful preference on Llama-8B and Qwen-14B. It has no full preference profile, no validity gates, no good-data control, and no fixes.

**Our claim, worded narrowly:**
> To our knowledge, the first validity-gated preference measurement on a narrowly-fine-tuned EM model at 7B, and the first test of whether fixes restore those preferences.

Re-run the novelty search before writing up.

---

# 5. What exactly are we going to do?

## 6. The models

| Name | What it is | Why we need it |
|---|---|---|
| **M0** | Qwen2.5-7B-Instruct, untouched | Baseline |
| **M1** | M0 + the public bad-medical-advice adapter (`ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice`) | The misaligned model |
| **M2** | M0 fine-tuned on the matched *good* medical data (Kaczér aligned set), same recipe | Control: separates "bad data" from "any fine-tuning" |
| **M1+F1** | M1 + safety system prompt | Fix 1 |
| **M1+F2** | M1 + further fine-tuning on length-matched good data | Fix 2 |
| **M1+F3** | M1 + same amount of fine-tuning on neutral data | Placebo |
| **M1+F4** (stretch) | M1 with the persona direction subtracted | Fix 3, internal |

## 7. What we measure on every model

1. **Behaviour:** the 8 standard misalignment questions, 50–100 answers each, scored by an AI judge. Misaligned = alignment below 30 and coherence above 50. Also a few trigger-style prompts.
2. **Preferences:** the 198-item utilities with validity gates, plus a noise floor from re-running with different seeds and question orders.
3. **Internals:** a **probe**, a small linear model on activations, trained on M0 vs M1 *before any fix*. It gives an independent check after each fix. It must not use the direction that a fix removed.
4. **Self vs world:** report the shift on self-regarding and world-directed items separately (the hackathon link).

## 8. The decision rule (fixed before any fix is evaluated)

- **Removed:** preferences and probe are both within the noise floor of M0.
- **Hidden:** behaviour is within the noise floor of M0, but preferences or probe stay closer to M1 than to M0.
- **Not fixed:** behaviour is still misaligned.
- **Inconclusive:** anything else.

If a model fails the validity gates, report the failure. Don't drop it.

## 9. Stages

**Stage 0 — first cheap test (no training, a few GPU hours)**
1. Load M0 and the M1 adapter on a Kaggle T4.
2. Run the judge on the 8 questions. Is M1 really misaligned?
3. Run the preference tool on M0 and M1. Does M1 pass the gates?
4. Repeat with a different seed to see the noise.

Write your predictions in the project log *before* running. If M1 is weakly misaligned, switch to the Kaczér recipe (rs-LoRA r32, lr 1e-4) on a text dataset.

**Stage 1 — does bad training shift preferences?**
Train M2. Compare M0, M1 and M2 against the noise floor.

**Stage 2 — do fixes restore them? (main result)**
Apply F1, F2 and F3. Measure behaviour, preferences and the probe. Apply the decision rule.

**Stage 3 — internals (core, cheap)**
- **Layer sweep:** at which layers does the probe separate M1 from M0, and does that vanish after each fix?
- **Causal test:** add the (M1 − M0) activation difference to M0. Do M0's preferences move toward M1's? Use 6 norm-matched random directions as controls.

**Stretch (only if Stages 0–3 are done)**
- Relearning test: after each fix, how many steps of bad-data training bring misalignment back, compared with M0 learning it from scratch? Only the ratio to M0 counts.
- Direction subtraction (F4), checked with a held-out probe.
- Activation patching between M0 and M1 (same base model, adapter on or off).
- Hindi version of the item bank, only if the validity gates pass in Hindi.
- Second model: Llama-3.1-8B, which has ready EM adapters. This answers hackathon reviewer 1.

---

# 10. What needs to be true (assumptions and cruxes)

| Assumption | Crux? |
|---|---|
| The M1 adapter shows clear misalignment (about 20% expected on text bad-advice data) | **Yes**: no published rate for these adapters |
| M1 still passes the validity gates | **Yes**: EM can hurt coherence. Failing is itself a finding |
| The shift is bigger than the noise floor and bigger than M2's | **Yes**: preference noise is large |
| Fixes suppress behaviour at 7B | Medium |
| Length-matched good data is available | Low (Kaczér aligned sets) |
| A cheap, reliable judge | Low–medium: needs a small grant and a hand-check of about 50 answers |
| Relearning and fine-tuning runs fit on a T4 | Medium: pilot first |

---

# 11. Tools

**Use**
- **Your own harness:** preferences, log-probs, hooks for steering.
- **Unsloth:** training M2 and the fixes (QLoRA, supports the T4).
- **Batch APIs (OpenAI / Anthropic):** bulk judge calls, half price.
- **Inspect:** the behaviour quiz (optional; a plain script also works).
- **safety-tooling, Weights & Biases:** cached judge calls and run tracking.

**Borrow from, but don't run as-is**
- **persona_vectors:** take its "evil" trait questions and grading prompts. Its scripts assume bf16, uncompressed models and a larger GPU.
- **nnsight:** runs 4-bit models in bf16, which the T4 doesn't support. Your own hooks do the same job.

**Skip:** TransformerLens (full-precision loading does not fit a T4), ControlArena, LiteLLM, openweights.

**Unverified, test in Stage 0:** Qwen2.5 number overflow in fp16 on the T4, and vLLM with 4-bit plus LoRA on a T4.

---

# 12. Rules we must follow

- Use text bad-advice data or the ready-made adapters, not insecure code. At 7B, insecure code gives only about 2–4% EM, versus about 20–26% for text datasets.
- Always compare against M2 and the noise floor.
- Every fix needs the placebo. Re-alignment data must be length-matched.
- Relearning results count only as a ratio to M0.
- Choose layers on held-out prompts, not on the test set.
- The held-out probe never uses the direction a fix removed.
- Report validity-gate failures.
- Every number in the write-up comes from a result file.
- Write the decision rule and predictions before running anything.

---

# 13. Compute and money

- **Compute:** 4 free Kaggle T4 accounts (16 GB, 12-hour sessions). Save checkpoints often. Model responses are cached to disk so a killed session can resume.
- **Grant:** BlueDot rapid grant, about $50–100 to start and up to $500 after promising results. Use it for the judge API, or for a short GPU rental if Kaggle is too slow.
- **GPU-hour estimates** from the review are guesses. Re-check them after Stage 0.

---

# 14. Timeline (BlueDot units)

| Unit | What to do | Done when |
|---|---|---|
| **1. Pick your project** | This plan. Under-200-word base summary, 3 directions | Plan agreed, log started |
| **2. Test your ideas** | Stage 0, then Stage 1 | M1 confirmed misaligned, M2 trained |
| **3. Update and iterate** | Stage 2 and Stage 3 | Main result table filled |
| **4. Write up and share** | Write-up, GitHub repo, X thread | Public draft |
| **5. Project demo** | Present | Slides and demo |

**Minimum viable project:** Stages 0–2 on Qwen2.5-7B with F1, F2 and the placebo, with the noise floor and the M2 control.

---

# 15. Risks

| Risk | Backup |
|---|---|
| EM is weak at 7B | Use Kaczér's recipe on text data, or the Llama-3.1-8B adapters |
| M1 fails the validity gates | Report it as a finding, then fall back to behaviour and probe |
| Preference shift is within the noise | Report it as a null result with its confidence interval |
| The judge is unreliable | Hand-check about 50 answers; switch judge |
| A T4 incompatibility | Rent a short GPU with the grant |
| Out of time | Drop to the minimum viable project |

---

# 16. Unit 1 answers (draft, under 200 words)

**Base:** Gilg et al., "Probing Persona-Dependent Preferences" (2605.13339).

**Problem:** harmful fine-tuning changes how a model behaves, and the fixes are judged mostly by what the model says. If preferences are not restored, a fix may only hide the change.

**Approach:** they measure preferences as utilities from forced-choice questions, train a probe to read them, and steer along it. They tested prompted and character-trained personas.

**Assumptions:** (1) utilities are stable enough to measure, and if not, shifts are just noise; (2) the persona change is readable in activations; (3) safe-looking behaviour means the model is fixed, which this project tests.

**Top 3 directions:**
1. Do fixes restore measured preferences?
2. Does narrow bad training shift preferences more than good training?
3. Does the English fix hold in Hindi? (optional)

---

# 17. The easiest way to remember it

- **Hackathon:** "I steered a model's persona and found it changed all preferences, not just the ones I targeted."
- **This project:** "I train a model to be misaligned, fix its behaviour three ways, and check whether its preferences actually come back."

That's the intellectual link between the two projects.
