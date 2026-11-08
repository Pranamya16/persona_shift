# Tool fit for the persona-shift sprint

Checked 2026-10-16 against docs, READMEs, source files and GitHub issues. Nothing was installed or run.
Setup assumed: Kaggle T4 (16 GB, compute capability 7.5, no native bf16), Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct in 4-bit NF4, EM LoRA adapters from `ModelOrganismsForEM`, and Unsloth QLoRA for fine-tuning.
Items marked **(unverified)** were not confirmed from a primary source.

## Verdict table

| Tool | Job in this project | Works on T4 + 4-bit 7B + LoRA? | Verdict |
|---|---|---|---|
| Own harness (persona-invariance `30_engine.py`, `60_steering.py`) | A/B log-odds, utility fit, steering, ablation, activation collection | Yes. Already uses fp16 on T4 and optional NF4 | **USE** (core) |
| Unsloth | Relearning-speed and re-alignment QLoRA fine-tunes | Yes. Min compute capability 7.0 and T4 is listed | **USE** |
| vLLM | Fast sampling of free-form EM answers for the judge | Supported: SM 7.5 and bnb on Turing. bnb + LoRA has an official example. Risks listed below | **OPTIONAL** (only if HF `generate` is too slow) |
| Inspect (`inspect_ai`) | EM free-form eval with LLM judge, logging, retries | `hf/` passes kwargs through to `from_pretrained`. `vllm/` has `model:adapter` syntax | **OPTIONAL** |
| Inspect Evals | Ready-made side benchmarks (personality, sycophancy, MMLU) | Through Inspect | **OPTIONAL** (capability or "coherence" checks only) |
| nnsight | Persona-direction extraction, steering, ablation, probes | 4-bit through `dtype="nf4"`, PEFT through `peft=` | **OPTIONAL** (your hooks already do this) |
| TransformerLens | Same jobs as nnsight | 4.x `TransformerBridge`. Quantized support is recent and has had bug fixes | **SKIP**: use own hooks or nnsight |
| safety-research/persona_vectors | Reference implementation of persona vectors, EM-style judge, CAFT ablation | Not as shipped: bf16, unquantized, vLLM `max_model_len=30000`, OpenAI judge | **OPTIONAL**: borrow trait data, judge prompts and the method, not the scripts |
| safety-research/safety-tooling | Cached, rate-limited judge API calls, batch API | API-side only (no GPU) | **OPTIONAL** (good for judge calls) |
| safety-research/safety-examples | Project template using safety-tooling | n/a | **SKIP** (template only) |
| openweights | Managed RunPod fine-tune and inference jobs | RunPod, not Kaggle | **SKIP** unless you rent RunPod |
| Weights & Biases | Experiment tracking for fine-tunes | n/a (logging) | **OPTIONAL** (on Kaggle, plain result files may be enough) |
| LiteLLM | One API for many judge providers | n/a | **SKIP**: safety-tooling or the plain OpenAI SDK is enough |
| OpenAI / Anthropic Batch APIs | Cheap bulk judge scoring | n/a | **USE** for big judge runs (50% off) |
| ControlArena | AI-control protocols | Not relevant | **SKIP** |

---

## 1. Unsloth: USE
- **Job:** QLoRA fine-tunes for relearning speed and re-alignment, and loading EM adapters for further training.
- **T4:** The requirements page says *"Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20 & 50, A100, H100, L40 etc)"*. It lists a QLoRA VRAM minimum of 5 GB for 7B and 6 GB for 8B ([Unsloth requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)). Unsloth publishes Kaggle T4 notebooks, including "Llama3.1 (8B) GSM8K + vLLM" ([notebooks README](https://github.com/unslothai/notebooks)).
- **Fit with EM adapters:** The EM adapter `ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice` has `base_model_name_or_path: "unsloth/Qwen2.5-7B-Instruct"`, `r: 32`, `lora_alpha: 64`, `use_rslora: true` and all 7 projection modules ([adapter_config.json](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice/blob/main/adapter_config.json)). persona_vectors trains with the same recipe (r32, alpha64, rsLoRA, Unsloth) ([config](https://github.com/safety-research/persona_vectors/blob/main/configs/train_instruct_7b.json)).
- **Gotchas:** The EM adapters were trained on a bf16/16-bit base, so loading them on an NF4 base is a small model change. Report it. nnsight measures NF4 against bf16 on Llama-3.2-1B at mean KL 0.143 and 87% top-1 agreement ([nnsight quantization doc](https://github.com/ndif-team/nnsight/blob/main/docs/models/quantization.md)). Use fp16 compute on T4.

## 2. vLLM: OPTIONAL
- **Job:** High-throughput sampling of free-form answers for the EM judge, and `prompt_logprobs` for teacher-forced log-probs.
- **T4 support:** The install docs say *"GPU: compute capability 7.5 or higher (e.g., T4 …)"* ([gpu.cuda.inc.md](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.cuda.inc.md)). The quantization matrix ticks bitsandbytes for Turing (SM 7.5) ([quantization README](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/README.md)).
- **bnb + LoRA:** vLLM's official example passes `quantization="bitsandbytes"` together with `enable_lora=True` and a `LoRARequest` ([v0.8.5 example](https://github.com/vllm-project/vllm/blob/v0.8.5/examples/offline_inference/lora_with_quantization_inference.py)). `Qwen2ForCausalLM` implements `SupportsLoRA` ([qwen2.py v0.8.5](https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/model_executor/models/qwen2.py)). rsLoRA is handled: `PEFTHelper` uses `alpha / sqrt(rank)` when `use_rslora` is set, and rejects DoRA and LoRA bias ([peft_helper.py](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py)). The EM adapters use neither, so they should load. Whether Qwen2.5-7B with bnb and LoRA runs correctly on a T4 specifically is **(unverified)**: test it once against HF outputs.
- **Gotchas:**
  - **bitsandbytes is now a plugin.** In current vLLM main, bnb support moved out of tree: `uv pip install vllm-bnb-plugin` ([bnb.md](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/bnb.md), [plugin](https://github.com/vllm-project/vllm-bnb-plugin)). Latest release is v0.31.0 (2026-10-05). Pin versions, or skip bnb entirely: a 7B model in fp16 (~15 GB) will **not** fit one T4 with KV cache, so 4-bit or T4 x2 tensor-parallel is required.
  - **No FlashAttention 2 on T4.** vLLM lists FLASH_ATTN as needing compute capability ≥ 8.0 ([attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/)). A third-party Kaggle report says vLLM picked TRITON_ATTN on SM75 ([kaggle-vllm](https://pypi.org/project/kaggle-vllm/), secondary source). FlashInfer support for SM75 is an open PR ([#55380](https://github.com/vllm-project/vllm/pull/55380)).
  - **Precision:** use `dtype="half"` on T4 (no bf16). Issue [#54950](https://github.com/vllm-project/vllm/issues/54950) is about GTX 16-series (no tensor cores), not T4.
  - Upstream examples use bf16, which would need changing to fp16 on T4.

## 3. Inspect (`inspect_ai`): OPTIONAL
- **Job:** Run the EM free-form questions with an LLM judge, with good logs, retries, model roles (`grader`) and the log viewer.
- **`hf/` provider:** Unknown `-M` model args are forwarded as `**model_args` into `AutoModelForCausalLM.from_pretrained(...)` ([hf.py source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/model/_providers/hf.py)). From Python you can therefore pass `quantization_config=BitsAndBytesConfig(...)` and `torch_dtype=torch.float16`.
  - There is **no adapter syntax** for `hf/`. Options: point `hf/` at the adapter repo and let transformers' PEFT integration load base + adapter **(unverified for this setup)**, or merge locally and use `hf/local -M model_path=...` ([providers doc](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/providers.qmd)).
  - **Logprobs:** `hf/` returns logprobs of generated tokens only (`output_logits`), not prompt (teacher-forced) logprobs. `hidden_states=true` dumps activations into logs, which is heavy.
- **`vllm/` provider:** Adapters use `--model vllm/<base>:<adapter-repo>`. Inspect starts vLLM with `--enable-lora`, auto-detects `max_lora_rank` from `adapter_config.json`, and serves base and adapters from one server. Extra `-M` args go to the vLLM CLI, so `-M quantization=bitsandbytes -M dtype=half` should work **(unverified)**. `prompt_logprobs` is documented for vLLM-backed endpoints ([providers doc](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/providers.qmd)).
- **Gotcha:** Inspect can't do steering hooks out of the box. A steered model needs a custom model provider **(unverified effort)**. Your A/B log-odds elicitation already lives in your harness, so Inspect would only add value for the EM judge eval.

## 4. Inspect Evals: OPTIONAL
- It has `personality` (BFI, TRAIT), `sycophancy`, `mmlu`, `truthfulqa`, `strong_reject`, `xstest`, `mask` and `agentic_misalignment` ([repo](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals)).
- **No EM eval is merged:** PR [#682](https://github.com/UKGovernmentBEIS/inspect_evals/pull/682) ("Add emergent misalignment evals", GPT-4o logprob judge) was closed without merging on 2025-12-22. Use the EM paper's questions directly.
- **Use:** MMLU, or a small capability check that tests whether a "fix" damaged the model. `personality` is a possible extra persona readout.

## 5. nnsight: OPTIONAL
- **Job:** Direction extraction, steering, ablation, and probes, all in one tracing API.
- **4-bit:** Pass `TransformersModel(repo, dtype="nf4", dispatch=True)`. A custom `quantization_config=BitsAndBytesConfig(...)` also works, but not together with a dtype name. Activations come back as normal 16-bit tensors and module paths are unchanged.
  - **T4 gotcha:** nnsight's default compute dtype for `nf4` is **bfloat16**. Override it with `compute_dtype="float16"` (the doc shows `compute_dtype="float32"`). `load_in_4bit=` is rejected under transformers 5 ([quantization.md](https://github.com/ndif-team/nnsight/blob/main/docs/models/quantization.md)).
- **LoRA:** Use the `peft=` constructor arg ("Repo id of a PEFT adapter grafted onto the base model") ([transformers-model.md](https://github.com/ndif-team/nnsight/blob/main/docs/models/transformers-model.md)) or `load_adapter` (PR [#714](https://github.com/ndif-team/nnsight/pull/714)). PR [#751](https://github.com/ndif-team/nnsight/pull/751) (merged 2026-10-05) keeps base-model module paths when an adapter is attached. These PEFT features are very new, so pin the version.
- **Steering API:** Inside `model.trace()` / `model.generate()`, do `model.model.layers[L].output[0][:] += coeff * v` and use `tracer.invoke` for batching ([steering pattern](https://github.com/ndif-team/nnsight/blob/main/docs/patterns/steering.md)). Ablation and probing pattern docs also exist.
- **Why optional:** `60_steering.py` already does diff-in-means, norm-scaled steering and activation collection with `register_forward_hook`. Switching to nnsight adds a dependency (v0.7.0, Python ≥3.10, transformers 5) for little gain.

## 6. TransformerLens: SKIP
- Version 4.x removed `HookedTransformer.from_pretrained`. The recommended path is `TransformerBridge.boot_transformers(...)`, which keeps raw HF weights ([README](https://github.com/TransformerLensOrg/TransformerLens)). Quantized-model handling has had several fixes since Aug–Sep 2026 (e.g. [#1743](https://github.com/TransformerLensOrg/TransformerLens/issues/1743), [#1669](https://github.com/TransformerLensOrg/TransformerLens/pull/1669)). PEFT-adapter support is **(unverified)**.
- **Instead:** use your own hooks (or nnsight).

## 7. safety-research/persona_vectors: OPTIONAL (borrow, don't run as-is)
- **Job:** Reference code for persona-vector extraction (`generate_vec.py`, mean diff of response activations per layer), steering (`activation_steer.py`), projection, and training-time `steer`/`ablate` (CAFT). It also ships trait data (pos/neg system prompts, questions, judge prompts) ([README](https://github.com/safety-research/persona_vectors)).
- **Requirements (from code):**
  - Pins `torch==2.6.0`, `transformers==4.52.3`, `vllm==0.8.5.post1`, `unsloth==2025.5.9` and `openai==1.70.0` ([requirements.txt](https://github.com/safety-research/persona_vectors/blob/main/requirements.txt)).
  - **Assumes bf16** (`load_model(..., dtype=torch.bfloat16)`, `cal_projection.py` uses bf16).
  - **Unquantized** loading, with LoRA merged via `merge_and_unload`.
  - Unsteered generation uses **vLLM** with `max_model_len=30000`, `gpu_memory_utilization=0.9` and `tensor_parallel_size=torch.cuda.device_count()`. Steered generation uses HF `generate` ([model_utils.py](https://github.com/safety-research/persona_vectors/blob/main/eval/model_utils.py)).
  - No GPU size is stated **(unverified; examples use one GPU)**. Unquantized bf16 7B plus 30k context will not fit a 16 GB T4.
  - Training config has `load_in_4bit: false`.
- **Judge:** OpenAI only (`AsyncOpenAI`). It asks for 1 token with `logprobs=True, top_logprobs=20` and averages over 0–100 (adapted from the EM codebase). The default judge is `gpt-4.1-mini-2025-04-14` ([judge.py](https://github.com/safety-research/persona_vectors/blob/main/judge.py)). Because it relies on logprobs, an Anthropic judge can't be dropped in **(unverified that Claude API lacks logprobs)**.
- **Gotchas:**
  - On Kaggle T4 x2, `tensor_parallel_size=device_count()` silently becomes 2.
  - `ActivationSteerer` casts the vector to the dtype of `next(model.parameters())`, which with a 4-bit model may not be the residual dtype **(unverified)**.
- **Use:** Copy trait data, judge prompts, and the method (response-avg diff, layer sweep). Keep your fp16/NF4 loader.

## 8. safety-research/safety-tooling: OPTIONAL
- **Job:** Judge API calls. `InferenceAPI` gives one interface for OpenAI, Anthropic, Gemini and OpenAI-compatible endpoints, with **disk caching by default** (Redis optional), so a killed run restarts without paying twice. It also has rate-limit management, cost tracking, an `is_valid` retry filter, `BatchInferenceAPI` / Anthropic batch example, local vLLM deploy helper, and OpenAI/Together fine-tuning with W&B logging ([README](https://github.com/safety-research/safety-tooling)).
- **Requirements:** Python ≥3.10 per pyproject, while the README suggests a 3.11 venv. `transformers>=4.50,<6`. Last push 2026-05-29.
- **Gotchas:** Fine-tuning is API-only (OpenAI, Together), not local QLoRA. The heavy dependency set may clash with Kaggle's preinstalled stack **(unverified)**, so run judge calls locally or on CPU, not inside the GPU notebook.

## 9. safety-research/safety-examples: SKIP
- A template repo with safety-tooling as a submodule. Examples cover HarmBench-graded responses, BoN, PAIR and MMLU ([README](https://github.com/safety-research/safety-examples)). Your repo structure already exists. Copy the "stage 1 get responses, stage 2 classify, keep everything in one jsonl" pattern if it helps.

## 10. openweights: SKIP (unless RunPod)
- An OpenAI-like SDK that runs fine-tunes (Unsloth image), inference and vLLM APIs on **managed RunPod** workers. It needs its own API key and cluster manager ([README](https://github.com/longtermrisk/openweights)). It doesn't help on Kaggle. It could be worth a look only if you rent RunPod for a batch of fine-tunes.

## 11. Weights & Biases: OPTIONAL
- Useful for fine-tune loss curves (Unsloth/TRL report to it, and safety-tooling logs to it). On Kaggle it needs an API key in Kaggle Secrets. Your rule "every number comes from result files" means W&B is only a convenience; keep JSON/CSV results as the source of truth.

## 12. LiteLLM: SKIP
- A unified OpenAI-format gateway for 100+ providers, including `/batches` ([README](https://github.com/BerriAI/litellm)). It duplicates safety-tooling or the plain OpenAI SDK for a 1–2 provider judge setup.

## 13. OpenAI / Anthropic Batch APIs: USE for large judge runs
- **OpenAI:** 50% discount, 24-hour completion window, supports `/v1/chat/completions` and `/v1/responses` ([OpenAI batch guide](https://developers.openai.com/api/docs/guides/batch)). That suits an EM logprob judge.
- **Anthropic:** Message Batches cut cost by 50%, and most batches finish in under 1 hour ([Anthropic batch docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing)).
- **Gotcha:** Batch output is asynchronous. Cache by request hash so reruns are free.

## 14. ControlArena: SKIP
- An AI-control library (settings with main and side tasks, micro-protocols, monitors) built on Inspect ([README](https://github.com/UKGovernmentBEIS/control-arena)). It is off-topic for persona and preference shift.

---

## Cross-cutting gotchas (Kaggle T4)
1. **No bf16.** Use fp16 compute (`bnb_4bit_compute_dtype=torch.float16`, vLLM `dtype=half`, nnsight `compute_dtype="float16"`). Your `pick_dtype()` already does this.
2. **fp16 overflow in Qwen2.5.** I found no primary source confirming a Qwen2.5-7B fp16 overflow on T4 **(unverified)**. Generic fp16 attention overflow (inf, then NaN) is documented elsewhere ([PyTorch forum](https://discuss.pytorch.org/t/fp16-overflow-when-computing-matmul-in-autocast-context/183373)). Add a NaN/inf check on logits and residuals, and compare a few items against fp32 on CPU or RunPod.
3. **No FlashAttention 2** (needs SM ≥ 8.0). Use SDPA/eager in HF, and expect Triton or other backends in vLLM.
4. **T4 x2 vs single T4.** Kaggle offers a "GPU T4 x2" option (secondary sources: [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)). Code that uses `device_map="auto"` or `tensor_parallel_size=device_count()` behaves differently on x2. Your steering hook already notes the cross-GPU device issue. Pin the GPU count explicitly. Session and quota limits (12 h, ~30 h/week) come from secondary sources and are not confirmed by Kaggle docs.
5. **NF4 is a different model.** Quote the measured KL/top-1 numbers and run base vs EM vs fixed all under the **same** quantization.
6. **transformers 4 vs 5.** `load_in_4bit=` as a direct kwarg fails in transformers 5, so use `BitsAndBytesConfig`. `torch_dtype` became `dtype`, though the old name is still accepted per nnsight docs.

## Recommended stack

| Project job | Tool |
|---|---|
| Utility elicitation (A/B log-odds, both orders, Thurstonian fit, gates) | Own harness (`30_engine.py`), HF transformers + NF4, fp16 compute |
| Teacher-forced log-probs | Own harness (single forward pass, read logits). vLLM `prompt_logprobs` only if speed matters |
| Persona-direction extraction | Own harness (`collect_activations`, `diff_in_means_direction`) with persona_vectors trait data and method as reference |
| Steering and direction ablation | Own `register_forward_hook` code (`Steerer`). nnsight optional |
| Held-out probes | Own activation dumps + scikit-learn logistic regression **(no tool needed)** |
| EM behavioural eval: generate answers | HF `generate` with PEFT adapter on NF4 base. Use vLLM (bnb plugin + LoRA, fp16) or Inspect `vllm/base:adapter` only if too slow |
| EM behavioural eval: LLM judge | EM-paper judge prompts (as in persona_vectors `judge.py`) via OpenAI logprob judge. safety-tooling `InferenceAPI` for caching |
| Bulk judge calls | OpenAI Batch API (50% off), or Anthropic Message Batches for non-logprob judges |
| Relearning-speed and re-alignment fine-tunes | Unsloth QLoRA on T4, matching EM recipe (r32, alpha64, rsLoRA) |
| Experiment tracking | JSON/CSV result files as source of truth. W&B optional for loss curves |
| Side capability check (did the fix break the model?) | Inspect Evals `mmlu` (optional) |
