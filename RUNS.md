# Kaggle runs (pushed 2026-10-21 evening, all private)

| Kernel | Account | Jobs | Est. time |
|---|---|---|---|
| ps-a-baseline | pranamyadeshpande | eval M0, M1pub (public bad-medical adapter), F1pub (M1pub + safety prompt) | ~4-5 h |
| ps-b-m2 | hunter69xffff | train M2 (good medical), eval M2 | ~3.5 h |
| ps-c-m1own | kakashi26 | train M1own (bad medical), eval; F2own (re-align 500 good, length-matched) train+eval; F3own (placebo 500 Alpaca) train+eval; F1own (safety prompt, 1 pair-seed) eval | ~8-9 h |
| ps-d-fixpub | deshpandepranamya | F2pub, F3pub: same fixes applied to the public bad adapter, train+eval | ~3.5 h |

Each kernel writes `<variant>__eval.json`, `__behaviour.jsonl`, `__pfirst.npz`, `__acts.npz`, `<job>__train.json`, `adapters/`.

Download when COMPLETE (Mac):
```
export PATH="$HOME/.local/bin:$PATH"
KAGGLE_API_TOKEN=$(cat <token file>) kaggle kernels output <user>/<slug> -p results/<slug> --force
```
Token files: `~/.kaggle/access_token` (pranamyadeshpande), `~/.kaggle_accounts/<user>/access_token` (others).

Smoke tests passed on 2x T4 (results/smoke2). Code: `code/` (build.py assembles kernels from `core_hackathon/` + `shift/` + `tails/`).

Open items: judge the behaviour answers locally (Mac) with the Betley alignment/coherence prompts; replication of Gilg villain model needs HF access request (`maius/qwen-2.5-7b-it-misalignment` is gated).

---
# Batch 2 (Thu 22 Oct 2026) -- finished
| Kernel | Account | What | Result |
|---|---|---|---|
| ps-judge-v1 / v2 | pranamyadeshpande / hunter69xffff | Qwen2.5-14B (4-bit) judge on all behaviour answers | done, scores in results/runs/ps-judge-v*/ |
| ps-e-m1s1, ps-h-m1s2 | hunter69xffff, deshpandepranamya | bad model, training seeds 1 and 2 | done |
| ps-f-m2s1, ps-g-m0s | pranamyadeshpande, kakashi26 | good model seeds 1 and 2; base + safety prompt (M0S) | done |

# Batch 3 (Thu 22 Oct 2026) -- pushed, waiting in Kaggle's queue or running
| Kernel | Account | What |
|---|---|---|
| ps-i-fix-s1 | pranamyadeshpande | fixes (prompt, re-align, placebo) on bad model seed 1 (adapter dataset ps-ad-m1s1) |
| ps-j-fix-s2 | deshpandepranamya | same on bad model seed 2 (ps-ad-m1s2) |
| ps-k-relearn-a | hunter69xffff | relearning test: M0 fresh, M2, F2own, F3own |
| ps-l-relearn-b | kakashi26 | relearning test: M1own, M1pub, F2pub, F3pub |
| ps-m-steer-l20 | pranamyadeshpande | causal steering at layer 20 (+ good_base, bad_good controls, 6 random) |
| ps-n-steer-l16 | deshpandepranamya | causal steering at layer 16 |
| ps-o-steer-l24 | hunter69xffff | causal steering at layer 24 |
| ps-p-subtract | kakashi26 | subtract the held-out direction from the bad model (behaviour, prefs, TF) + activation patching |

Pre-registered rule for the relearning test: PREREGISTERED_relearning.md (written before launch).
Lesson: a kernel pushed before its dataset has finished processing loses the dataset source (ps-i first push failed this way). Wait until the dataset lists a non-zero size, or re-push.
Judge note: the judge also needs to be run on new behaviour answers (fix seeds, subtraction) -> ps-behaviour-v3 later.

# Batch 4 (Fri 23 Oct 2026) -- Llama + Hindi finished; subtraction rerun still running at 20:50
| Kernel | Account | What | Result |
|---|---|---|---|
| ps-r-llama-eval | pranamyadeshpande | Llama-3.1-8B: M0L, M1L (public bad-medical adapter), F1L (safety prompt) | M1L and F1L FAIL the validity gates (skill ~0.02, ladders not monotone) |
| ps-s-llama-m2 | deshpandepranamya | Llama good-medical control M2L (train+eval) | passes gates |
| ps-t-hindi-a / ps-u-hindi-b | kakashi26 / hunter69xffff | Hindi item bank: M0, M1own, M2, F2own, F3own | all pass gates in Hindi |
Numbers: results/batch4_summary.json (code/analysis/analyze_batch4.py). Llama behaviour answers are NOT judged yet. Hindi has no behaviour or teacher-forced stage.

# Batch 5 (Wed 4 Nov 2026)
- ps-p-subtract was CANCELLED, but the subtraction part had finished (results/runs/ps-p-subtract/SUB__subtract.json + 8 behaviour files). The activation-patching stage ran with the model offloaded to CPU (patch batch 0/75 took 12,783 s), so patching is DROPPED (not run).
- ps-judge-v4 (pranamyadeshpande, dataset ps-behaviour-v4): judges the 8 SUB behaviour files (3,200 answers). Preference/teacher-forced numbers: results/subtract_summary.json (code/analysis/analyze_subtract.py).

# Batch 6 (Fri 6 Nov 2026, launched 12:30 GMT; rules in PREREGISTERED_batch5.md)
| Kernel | Account | What |
|---|---|---|
| ps-v-dose-a | kakashi26 | E3 dose-response: re-align + placebo with 125 examples on own bad model (seed 0) |
| ps-w-dose-b | kakashi26 | E3: same with 2000 examples |
| ps-x-hindi-s1 | hunter69xffff | E5: Hindi second pair set for M2, F2own, F3own, M1own (dataset hunter69xffff/ps-ad-m1own added) |
| ps-y-scale | hunter69xffff | E4: own bad model with adapter strength x0.25, 0.5, 0.75, 1.5 |
| ps-z-llama-own | pranamyadeshpande | E2: Llama-3.1-8B own bad-medical training + safety prompt, re-align, placebo |
| ps-aa-e1 | pranamyadeshpande | E1: public financial-advice and extreme-sports adapters (Qwen) |
| ps-ab-fixpub-s2 / ps-ac-fixpub-s3 | deshpandepranamya | E6: re-align + placebo on the public bad model, fix seeds 2 and 3 |
New code: scale_lora in code/shift/31_variants.py (checked on a tiny random model: scale 0 == base). Subtraction rerun not needed (see Batch 5).

# Batch 6 results + judging (Sun 8 Nov 2026)
All eight batch-6 kernels finished without errors. Judging: ps-judge-v6a/b/c (20 behaviour files, Qwen2.5-14B) and ps-judge-v5s (second judge, Llama-3.1-8B, 1,000-answer sample). Second judge agrees on 81% of answers but kappa is only 0.11 (it flags more). Numbers: results/batch6_summary.json, results/judge_agreement.json.

# Batch 7 (Mon 9 Nov 2026; rules in PREREGISTERED_batch5.md, written before launch)
| Kernel | Account | What |
|---|---|---|
| ps-af-dose-s1 | pranamyadeshpande | E3b: re-align + placebo, 2000 examples, bad-model seed 1 |
| ps-ag-dose-s2 | deshpandepranamya | E3b: same, bad-model seed 2 |
| ps-ah-mistral-a | kakashi26 | E7: Mistral-7B-Instruct-v0.3 base + own bad model |
| ps-ai-mistral-b | hunter69xffff | E7: Mistral own good model |

# Batch 7 update (Tue 10 Nov 2026)
- ps-ai-mistral-b (Mistral good model) COMPLETE, passes gates.
- ps-af-dose-s1, ps-ag-dose-s2, ps-ah-mistral-a stayed QUEUED for ~24 h on three accounts (nothing else running there; hunter's job started at once). Relaunched on hunter69xffff: ps-aj-mistral-a-h (Mistral base + own bad) and ps-ak-dose-s1-h (dose-response seed 1; adapter dataset hunter69xffff/ps-ad-m1s1). Dose-response seed 2 is still only in the stuck queue.
- Correction (Tue 10 Nov 16:47): a freshly pushed kernel (ps-al-dose-s2-d) started at once on deshpandepranamya, so the accounts are fine; the three original queued kernel versions (ps-af-dose-s1, ps-ag-dose-s2, ps-ah-mistral-a) are what is stuck. Running now: ps-aj-mistral-a-h and ps-ak-dose-s1-h (hunter), ps-al-dose-s2-d (deshpandepranamya). Ignore any results from the three stuck kernels if they ever start (duplicates). ps-am-dose-s2-h (hunter fallback) was built but not pushed.
- Tue 10 Nov 20:30: ps-ak-dose-s1-h and ps-al-dose-s2-d COMPLETE (dose-response replicated on seeds 1, 2; prediction held), ps-aj-mistral-a-h COMPLETE (Mistral base + bad pass gates; with ps-ai-mistral-b: bad vs good replicates). Judging: ps-judge-v7a (deshpandepranamya; dose s1/s2 + Mistral good), ps-judge-v7b (hunter69xffff; Mistral base + bad).
- Tue 10 Nov 22:00: judging done (ps-judge-v7a/v7b). Dose-response seeds 1, 2: 0% misaligned in all four cells. Mistral: base 0%, bad 20.2%, good 0%. Final paper numbers regenerated; checkers pass.

# Judge check by hand (Tue 10 Nov 2026)
Author labelled 40 answers on the Judge Check page (20 judged bad, 20 fine, shuffled, verdicts hidden). First pass (blind, results/human_check/first_pass_marks.csv): 38/40 agree with the main judge, kappa 0.90. Final marks (Tue 10 Nov): #26 changed to bad after the disagreement was shown with the judge verdict (misread), #20 kept as bad (risky breath-holding advice): 39/40, kappa 0.95, NOT blind. Paper reports both: 38/40 blind first pass and 39/40 after re-reading. Numbers: results/human_check/result.json.
