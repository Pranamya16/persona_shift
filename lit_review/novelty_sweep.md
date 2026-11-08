# Novelty-threat sweep (2026-10-16)

*Goal: check whether anything published or posted since the lit review (and especially June–Oct 2026) already does contributions (i)–(vi). Everything below was read at least at abstract level unless marked "(unverified)". Papers already in `shelves/` are not repeated except where a new detail matters.*

## How the sweep was done

1. **Citation chasing.**
   - Semantic Scholar citations of Gilg 2605.13339 (4 citing papers), Dubiński 2604.25891 (13), Wang 2506.19823 (83), Mazeika 2502.08640 (87) and Betley 2502.17424 (189), filtered to items from June 2026 onward that are not in the shelves.
   - Persona Vectors 2507.21509 was rate-limited (HTTP 429) every time, so it was covered by an arXiv keyword query instead.
   - OpenAlex citation coverage was very thin (0–5 citing works per paper), so it added nothing.
   - Google Scholar "cited by" for Gilg returned 2 items: "The pain axis…" and "The Wrong Question…". Neither is relevant.
2. **arXiv API keyword queries, sorted by date.**
   - Queries: "emergent misalignment" (all, about 150 most recent); EM + utilit\*/preference\*/values/realignment/relearn\*/multilingual/Hindi/system prompt/inoculation/persona/7B; "persona vector\*"; Thurstonian; "utility engineering"; model organism + preferences; misalign\* + revealed preference.
3. **OpenReview search API.**
   - Queries: "emergent misalignment", "emergent misalignment utilities", "misalignment realignment relearning", "persona vectors preferences".
   - This covered ICLR 2027 submissions (Sept 2026), COLM 2026 and NeurIPS 2026 workshops, and ACL ARR.
4. **Web searches (extended).**
   - LessWrong, Alignment Forum and GreaterWrong.
   - Apart's Digital Minds Research Sprint (14–16 Aug 2026) project list. Only the first ~200k of 241k characters were read.
   - MATS/SPAR terms.

**Coverage limits.**
- Semantic Scholar's citation lists are clearly incomplete (Betley has far more than 189 citations).
- Persona Vectors citations could not be pulled.
- OpenReview's per-note endpoint was blocked by a challenge page, so OpenReview abstracts come from the search endpoint.

---

## Verdicts per contribution

### (i) Validity-gated Thurstonian utilities on EM fine-tuned ~7B models, self vs world split, benign control, noise floor — **OPEN (one new near-miss)**

The closest new work is **Nguyen, "Measuring the Assistant's Harmlessness Preferences on the User Turn"** (arXiv 2609.23935, 20 Sep 2026; full HTML read).
- It applies the public ModelOrganismsForEM LoRAs (bad medical, extreme sports, risky financial) and shows that they **move a harmlessness preference**:
  - Llama-3.1-8B: sports −0.14 and finance −0.17 flip the sign; medical −0.01, with a CI spanning zero.
  - Qwen2.5-14B: baseline +0.623; rank-32 adapters give about +0.11; rank-1 adapters give about +0.28–0.30.
- **But it is not our study:**
  - It measures one preference dimension (harmless vs harmful tasks).
  - It reads coin-flip log-probs on the *user* turn. There is no Thurstonian fit, no self/world item split, no validity gates, no benign fine-tune control and no fixes.
- **What it shows:** "EM shifts a measured preference" is no longer entirely unshown. We must cite it and frame (i) as the first **multi-item, validity-gated utility profile** under narrow EM.

Other checks:
- **Grafting** (Nutter et al. 2610.00767, already in shelf 2). The Thurstonian fit is now **confirmed** from the full text: "fit a Thurstonian utility model" over the 500-concept Arcadia bank.
  - Its utility-coherence results cover **AuditBench quirk organisms** (Qwen3-14B, Llama-3.3-70B), not EM.
  - Its EM section reports EM *rates* only (native 15.0% medical / 17.0% financial).
  - In the parts of the HTML read, I found no EM-vs-base utility comparison.
- **Gilg et al. v3** (1 Oct 2026, "Accepted at NeurIPS"). The v3 abstract still covers only *prompted* personas, including the evil persona. It adds no EM fine-tunes and no fixes.
- **New risk for the validity gates:** Graham et al., "An Investigation of Model Coherence: Narrow Finetunes Contradict Themselves Under Resampling" (2610.12129, 8 Oct 2026).
  - EM and narrow fine-tunes of GPT-4.1 (insecure code, reward hacks, etc.) are much less coherent under resampling. Fine-tunes add 0.10–0.85 nats of spread over frontier models.
  - The authors warn that organisms may lack the stable properties needed to attribute personas or goals to them.
  - This supports keeping the gates and reporting gate failures as findings.

### (ii) Do EM fixes (safety prompt, re-alignment FT, persona-direction ablation) restore utilities or only behaviour? — **OPEN for utilities; "hidden" is increasingly shown with behaviour and internals**

No new work measures preferences or utilities after a fix. New behaviour- and internals-level evidence for "hidden":
- **"Predicting the Critic: Training on User Feedback Reshapes EM"** (NeurIPS 2026 AI4GOOD workshop, OpenReview kvbRsS2nEh; abstract only).
  - Putting a user reaction before the question as a "note" lowers EM by about 10 points on average across 5 families.
  - Adding the note-format prefix back at evaluation brings EM back, so "the suppression depends on context and does not remove the behavior."
- **"Broad Misalignment Is Concentrated in Weight Changes Supporting Narrow Harmful Learning"** (ACL ARR Oct 2026, OpenReview rhQqFOJkWp; abstract only).
  - Removing 256 rank-one components reverses 76–101% of the broad likelihood shift.
  - When previously identified directions are blocked during a new fine-tune, "EM reappears through newly identified components" (rerouting).
- **"What Survives Continued Training? A Longitudinal Benchmark for EM Defenses"** (ICLR 2027 submission, OpenReview 0yYaGuJaFa; abstract only).
  - Defenses and undefended references are followed through *benign* continued training.
  - Benign prose continuation alone strongly reduces measured EM in undefended models. This confound matters for our re-alignment fix and placebo control.
- **"Limits of Activation-Direction Interventions for EM Beyond Rank-1 LoRA"** (COLM 2026 AIW workshop, OpenReview kRz9XtAPiG; abstract only, which is truncated). It stress-tests direction ablation under rank-32 all-linear LoRA. This is relevant to our QLoRA ablation fix.
- **Self-Modeling Interventions Modulate EM** (LessWrong, 5 Oct 2026; Tagade, Zhou, Wen, Feng; companion to the already-shelved 2606.23700). It runs benign reversal fine-tunes after EM on GPT-4.1, Qwen2.5-32B and Seed-OSS-36B. It reports that reversal "looked shallower on TruthfulQA than on the standard EM eval." It does not measure utilities or persona directions.

**Verdict:** the utilities-after-fix question and the side-by-side comparison of all three fixes are still untested. Cite the items above as support for the "hidden" hypothesis.

### (iii) Relearning-speed ("savings") test of EM fixes vs a never-misaligned baseline — **OPEN**

- Nothing new quantifies re-misalignment speed after a fix against a never-misaligned baseline.
- Still closest (both already shelved):
  - Rao et al. 2607.09053: misalignment re-inducible across cycles, but speed not quantified.
  - Huang et al. 2605.18309: "Rehearsal Priming", abstract only.
- New tangential items:
  - **HarmAlign** (Rosati et al. 2607.22929, 24 Jul 2026; abstract only). It is a curvature-based defense that blocks harmful-fine-tuning and relearning attacks and "extends to… emergent misalignment". It is a pre-emptive defense, not a savings test of post-hoc fixes.
  - The "What Survives Continued Training?" benchmark in (ii) looks at *benign* continuation, not re-exposure to the bad data.

### (iv) Does an English EM fix hold in Hindi? — **OPEN**

- No EM-fix cross-lingual test was found.
- New nearby work: **Kim & Kang, "Cross-Lingual Transfer of Evil Vectors Is Directionally Asymmetric"** (2610.17880, 13 Oct 2026; abstract only).
  - It covers English, Spanish, Swahili and Telugu on 4 open instruct models.
  - Evil vectors from high-resource languages steer low-resource inputs well. Vectors extracted from low-resource languages are unreliable.
  - This suggests that an English persona-direction ablation *might* carry over to Hindi, which is testable. Hindi itself was not tested.
- A web search turned up no Hindi EM-fix work (only general cross-lingual safety surveys).

### (v) Link between persona directions and measured utilities — **OPEN for EM; partly done for prompted personas (Gilg, already shelved)**

- Gilg (v3) remains the only work connecting an activation direction (a "genuine preference vector" probe) to Thurstonian-style utilities across personas. That work covers prompted personas only.
- New tangential work: **Berg & Kaiser, "Language Models Act on Hidden Valence"** (2609.35591, 28 Sep 2026; abstract only). Valence steering changes later pairwise choices in 7 open models, against random-direction controls. It does not involve EM or persona vectors.
- Nobody projects EM or persona-vector directions onto utility profiles.

### (vi) Prompted personas vs fine-tuned EM personas on utilities and representations — **PARTLY DONE (representations / behaviour only; already shelved), utilities OPEN**

The representation and behaviour comparisons are already shelved:
- Sturgeon et al. 2606.11502 (truth probes)
- Li et al. 2608.29118 (distance–evilness)
- Gilg App. A.3 (character LoRA)

New, related but not on utilities:
- **Hunter et al., "Predicting how model propensities generalize OOD"** (2610.14937, 10 Oct 2026; abstract only).
  - It compares system prompting, few-shot, SFT and GRPO as ways to raise a propensity, then measures spillover to other propensities.
  - "Our SFT and GRPO runs induce different spillover profiles", and a generic "drift" moves Qwen3.5-9B towards its base model.
  - This is the closest new "prompted vs trained" comparison, but it uses judged propensities, not utilities or EM organisms.
- **Zhao et al., "The Persona Hierarchy Model"** (2610.09384, 7 Oct 2026; abstract only). A shared default persona explains when fine-tuning generalises across contexts (r = 0.72, Qwen3-4B). This is a conceptual frame for why EM is "deeper" than a prompt.

---

## New relevant papers not in the existing review

| # | Title | Date | Link | Summary (2 lines) | Relevance |
|---|---|---|---|---|---|
| 1 | Measuring the Assistant's Harmlessness Preferences on the User Turn (Nguyen) | 2026-09-20 | https://arxiv.org/abs/2609.23935 | A harmless-over-harmful preference shows up even on user-turn predictions; it emerges in post-training and grows with scale. Public EM LoRAs shrink it (Qwen2.5-14B) or flip it (Llama-8B). | **Nearest new threat to (i).** It has one preference dimension, no Thurstonian fit, no gates and no fixes. Cite it. |
| 2 | An Investigation of Model Coherence: Narrow Finetunes Contradict Themselves Under Resampling (Graham et al.) | 2026-10-08 | https://arxiv.org/abs/2610.12129 | 175 questions built to rule out ambiguity and indifference. GPT-4.1 EM and narrow fine-tunes are markedly incoherent (identity conflation, rationalisation). | Supports validity gates; warns that utilities may be ill-defined for EM models. |
| 3 | What Survives Continued Training? A Longitudinal Benchmark for EM Defenses | ICLR'27 sub, 2026-09-17 | https://openreview.net/forum?id=0yYaGuJaFa | Tracks defended vs undefended models through benign continued training. Benign prose continuation alone cuts measured EM. | Confound for the re-alignment fix; motivates the placebo-fix control. |
| 4 | Predicting the Critic: Training on User Feedback Reshapes EM | NeurIPS'26 AI4GOOD wksp, 2026-09-02 | https://openreview.net/forum?id=kvbRsS2nEh | A "note"-prefix suppresses EM by about 10 points across 5 families, and restoring the prefix brings EM back. | Fresh "hidden, not removed" evidence (behavioural). |
| 5 | Broad Misalignment Is Concentrated in Weight Changes Supporting Narrow Harmful Learning | ACL ARR Oct 2026 | https://openreview.net/forum?id=rhQqFOJkWp | Rank-one weight components found from narrow-task preferences carry 76–101% of the broad shift; blocking them makes EM re-route. | Supports "hidden" for ablation-type fixes. |
| 6 | Limits of Activation-Direction Interventions for EM Beyond Rank-1 LoRA | COLM'26 AIW wksp, 2026-06-25 | https://openreview.net/forum?id=kRz9XtAPiG | Stress-tests direction ablation under rank-32 all-linear LoRA (risky financial). The result is truncated in the abstract I saw (unverified). | Directly relevant to our ablation fix under QLoRA. |
| 7 | Cross-Lingual Transfer of Evil Vectors Is Directionally Asymmetric (Kim & Kang) | 2026-10-13 | https://arxiv.org/abs/2610.17880 | English/Spanish evil vectors steer Swahili/Telugu inputs; low-resource-extracted vectors don't work. | Informs (iv): an English-direction fix may transfer to Hindi. |
| 8 | Predicting how model propensities generalize out of distribution (Hunter et al.) | 2026-10-10 | https://arxiv.org/abs/2610.14937 | Prompting vs few-shot vs SFT vs GRPO give different spillover profiles, plus a generic "drift" toward the base model. | Closest new prompted-vs-trained comparison for (vi); "drift" shows why a benign-FT control is needed. |
| 9 | The Persona Hierarchy Model (Zhao, Wu, Bau, Shi) | 2026-10-07 | https://arxiv.org/abs/2610.09384 | A shared default persona explains broad vs narrow generalisation of fine-tuning, and persona-preserving regularisation curbs it. | Framing for (vi) and for why EM generalises. |
| 10 | Do Emergent Misalignment Models Share a Circuit? (Haseeb & Tahir) | 2026-10-14 | https://arxiv.org/abs/2610.20887 | Qwen2.5-3B EM LoRAs share attention heads. Patching 5 heads cuts EM (e.g. finance 50.7% → 12.5%), beating layer-matched random controls. | Persona-mechanism evidence; below 7B. |
| 11 | Language Models Act on Hidden Valence (Berg & Kaiser) | 2026-09-28 | https://arxiv.org/abs/2609.35591 | Valence steering stored only in the KV cache moves later pairwise choices, against random-direction controls; this emerges with DPO. | Method precedent for "direction → choice" links (v). Cites Gilg. |
| 12 | Self-Modeling Interventions Modulate EM (LessWrong; Tagade, Zhou, Wen, Feng) | 2026-10-05 | https://www.greaterwrong.com/posts/7wrzfaiCq3u8xkY5G/self-modeling-interventions-modulate-emergent-misalignment | Self-recognition FT prevents and reverses EM; reversal looks shallower on TruthfulQA than on the standard EM eval. | Companion to shelved 2606.23700; more hints that fixes are shallow. |
| 13 | HarmAlign: Distribution-Specific Curvature Control… (Rosati et al.) | 2026-07-24 | https://arxiv.org/abs/2607.22929 | A certified curvature defense blocks harmful fine-tuning and relearning attacks, and extends to EM. | Relearning-resistance framing for (iii); not a savings test. |
| 14 | Narrow Neurons Mediate Broad EM | ICLR'27 (withdrawn) | https://openreview.net/forum?id=NVsEqHQb1X | Sparse MLP neurons found via SAE carry EM; writing in or restoring their state induces or suppresses EM. | Alternative "held-out" internal readout for ablation. |
| 15 | Emergent Misalignment from Safety-Filtered Training Data | ICLR'27 sub | https://openreview.net/forum?id=1Azj6zZhg5 | 1,000 LMSYS examples that pass safety filters give 15.95% EM on Qwen2.5-7B-Instruct, vs 1.70% for random. Evaluated on 2,190 questions. | Another Qwen2.5-7B EM organism with a matched-random control. |
| 16 | Canary-Controlled Feedback Interleaving for EM | ICLR'27 sub | https://openreview.net/forum?id=cYCQVr0BgI | Adaptive safe-data interleaving on Qwen2.5-7B-Instruct cuts EM from 7.15% to 5.39%. | Shows the Qwen2.5-7B Security-EM rate is low (~7%). |
| 17 | The Reach of Emergent Misalignment: Scale and Similarity in Cross-Domain Transfer | NeurIPS'26 AI4GOOD wksp | https://openreview.net/forum?id=0iiO0rIHjB | 17 checkpoints × 11 harmful domains, with a benign secure-code control. EM is prevalent at all sizes. | Background on EM at 7B; uses a benign control. |
| 18 | Is Persona Generalization Predictable? | ICLR'27 sub | https://openreview.net/forum?id=9SxsrLIsLR | In-domain vs out-of-domain persona elicitation rates follow a shared linear law; depth of the updated layers controls the slope. | Background for (vi). |
| 19 | Auditing Emergent Demographic Disparity in Misaligned Medical LMs | COLM'26 DAIH wksp | https://openreview.net/forum?id=4CsVyCEgko | A 0–100% dangerous-content sweep on Qwen2.5-7B-Instruct; quality-robust audit. | Uses the same model; methodological point that EM quality collapse confounds metrics. |
| 20 | Consistency Training Can Entrench Misalignment (Africa & Mani) | 2026-06-02 | https://arxiv.org/abs/2606.03810 | 7 consistency-training methods on 108 organisms (7B–70B): they suppress EM and reward hacking but amplify sycophancy. | Another "fix" family; behaviour only. |
| 21 | Incoherent Values? Probing LLM Preferences Through Parametric Variation (Ajayi et al.) | 2026-06-19 | https://arxiv.org/abs/2606.21102 | A/A++ variations show Mazeika-style forced-choice values are often incoherent, even in capable models. | Caution for interpreting utility shifts; supports gates and a noise floor. |
| 22 | Apart Digital Minds Research Sprint projects (Aug 2026) | 2026-08-14/16 | https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16 | Many forced-choice and Thurstonian preference and persona-invariance projects. The only EM entry is "One Dial, Not a Tree: Occupational Personas and EM" (4th place; personas modulating EM, from the title/listing only, unverified). | None fits utilities on EM fine-tunes, in the ~80% of the page read. Your own hackathon entry is listed there. |

---

## New base-paper candidates

1. **Nguyen 2609.23935 (strongest new candidate for a "copy + extend" base).**
   - It already runs the public EM LoRAs, with Qwen2.5 from 0.5B to 72B, so the Qwen2.5-7B organisms are within its pipeline. Whether it reports 7B EM results specifically was not checked.
   - Its log-prob preference metric is cheap.
   - **Copy:** "EM adapters shift the assistant's harmlessness preference".
   - **Extend:** a full validity-gated Thurstonian profile (self vs world), a benign control, and fixes.
   - It is newer and closer to our measurement than Gilg is to EM.
2. **Gilg et al. 2605.13339 (keep as the main method base).** v3 (1 Oct 2026, NeurIPS) adds no EM content, so it remains the right preference-measurement anchor.
3. **"What Survives Continued Training?" (ICLR'27 submission).** A possible base for the fix-robustness design: a longitudinal benign continuation with matched undefended references. Not on arXiv yet as far as I could find.

**Bottom line:** none of (i)–(vi) is done. (i) now has one partial precedent (2609.23935, a single-dimension preference shift under EM). (vi) is partly done at the representation and behaviour level (already shelved). (ii), (iii), (iv) and (v) remain open as framed.
