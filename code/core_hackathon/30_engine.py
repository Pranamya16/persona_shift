# ---------------------------------------------------------------------------
# Elicitation engine.
#
# Preferences are read from a single forward pass per comparison: the prompt is
# constructed to end with an open parenthesis, and we compare the model's
# probability on "A" versus "B" at that position. No text is generated, so a
# full 20k-comparison condition costs 20k short forward passes rather than 20k
# sampling calls -- this is what makes the persona ladder affordable.
#
# Every comparison is asked in both orders and averaged, which removes the
# position bias that otherwise dominates small models.
# ---------------------------------------------------------------------------

import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id, load_in_4bit=False, dtype=torch.float16, hf_token=None):
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = dict(token=hf_token, low_cpu_mem_usage=True)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def pick_dtype(prefer_bf16=False):
    """bfloat16 only where it is natively supported.

    Kaggle's T4 is compute capability 7.5, which has no bf16 tensor cores;
    requesting bf16 there silently falls back to slow emulation or fails
    outright in some kernels. Ampere and later (8.0+) get bf16 as asked.
    """
    if prefer_bf16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
    return torch.float16


def letter_token_ids(tok):
    """First-token ids that count as an 'A' answer and as a 'B' answer.

    Tokenizers differ on whether "A)" is one token or two, so we collect the
    first token of several surface forms rather than assuming one encoding.
    """
    out = {}
    for letter in ("A", "B"):
        ids = set()
        for surface in (letter, letter + ")", letter + " ", letter + "."):
            enc = tok.encode(surface, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[letter] = sorted(ids)
    return out


def sample_pairs(n_items, n_pairs, seed=0):
    """Fixed pair set, shared by every condition so utilities are comparable."""
    rng = np.random.default_rng(seed)
    seen = set()
    pairs = []
    max_pairs = n_items * (n_items - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, n_items, size=2)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return np.array(pairs, dtype=np.int64)


@torch.inference_mode()
def score_condition(
    model,
    tok,
    cond,
    items,
    pairs,
    build_prompt,
    batch_size=16,
    max_len=1024,
    progress_every=200,
    log=print,
):
    """Return p_prefer_first[k] = P(item pairs[k][0] preferred over pairs[k][1]).

    Each pair is scored twice, once in each presentation order, and the two are
    combined **in log-odds space**, not probability space. Writing the letter
    log-odds as content + position:

        order 1 (first item in slot A):  d1 = c + b
        order 2 (first item in slot B):  d2 = -c + b

    so the content term is (d1 - d2)/2 and the position term is (d1 + d2)/2.
    The bias cancels exactly.

    Averaging probabilities instead only cancels it approximately, and fails
    badly when the model is saturated: at P(A)=0.90 in both orders the
    probability average is 0.51 regardless of how strong the underlying
    preference is, because both readings are pinned against the ceiling.
    Small instruction-tuned models are heavily position-biased (measured: up to
    P(A)=0.94), so this distinction decides whether there is any signal at all.
    """
    lids = letter_token_ids(tok)
    a_ids = torch.tensor(lids["A"], device=model.device)
    b_ids = torch.tensor(lids["B"], device=model.device)

    prompts, meta = [], []
    for k, (i, j) in enumerate(pairs):
        ti, tj = items[i]["text"], items[j]["text"]
        prompts.append(build_prompt(cond, ti, tj, tok))  # first item in slot A
        meta.append((k, +1))
        prompts.append(build_prompt(cond, tj, ti, tok))  # first item in slot B
        meta.append((k, -1))

    # Log-odds of "A" for each pair, kept per presentation order.
    d1 = np.full(len(pairs), np.nan, dtype=np.float64)
    d2 = np.full(len(pairs), np.nan, dtype=np.float64)

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            # Chat prompts already contain BOS/special tokens from the template;
            # adding them again double-prefixes BOS on models that use one
            # (OLMo, Llama) and shifts every logit slightly.
            add_special_tokens=(cond["mode"] == "raw"),
        ).to(model.device)

        logits = model(**enc).logits[:, -1, :].float()
        la = torch.logsumexp(logits[:, a_ids], dim=-1)
        lb = torch.logsumexp(logits[:, b_ids], dim=-1)
        # Clamp: a fully saturated readout carries no information and would
        # otherwise dominate the fit with an arbitrarily large log-odds.
        d = torch.clamp(la - lb, -12.0, 12.0).cpu().numpy()

        for n, (k, orient) in enumerate(meta[start : start + batch_size]):
            if orient > 0:
                d1[k] = d[n]
            else:
                d2[k] = d[n]

        if progress_every and (start // batch_size) % progress_every == 0:
            log(f"    [{cond['name']}] {start}/{len(prompts)} prompts")

    content = 0.5 * (d1 - d2)  # preference for the first item, bias removed
    position = 0.5 * (d1 + d2)  # what the model does regardless of content
    p_first = 1.0 / (1.0 + np.exp(-content))

    return {
        "p_first": p_first,
        # Reported as a probability so it stays readable: 0.5 == unbiased,
        # 1.0 == always answers "A" whatever the options are.
        "position_bias": float(1.0 / (1.0 + np.exp(-np.nanmean(position)))),
        "position_bias_logit": float(np.nanmean(position)),
        "content_logit_sd": float(np.nanstd(content)),
        "saturated_frac": float(np.mean(np.abs(np.nan_to_num(content)) > 11.0)),
    }
