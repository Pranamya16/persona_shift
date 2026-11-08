# ---------------------------------------------------------------------------
# Causal leg: does the persona *cause* the preference shift?
#
# Correlational evidence (utilities move when the system prompt changes) is
# consistent with the prompt changing anything at all. To make the causal
# claim we extract a persona direction from the residual stream, add it back
# during elicitation with no persona in the prompt, and check that utilities
# move -- and that norm-matched random directions do not.
#
# The random-direction control is the whole point. Steering with a large enough
# coefficient degrades any model's behaviour; the claim only survives if the
# persona direction moves self-regarding utilities *specifically*, at
# coefficients where random directions of identical norm do not.
# ---------------------------------------------------------------------------

import numpy as np
import torch


def get_layers(model):
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate transformer layer list")


@torch.inference_mode()
def collect_activations(model, tok, prompts, layer, batch_size=8, max_len=1024):
    """Residual-stream activation at the final token, for each prompt."""
    layers = get_layers(model)
    store = []

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store.append(h[:, -1, :].detach().float().cpu())

    handle = layers[layer].register_forward_hook(hook)
    try:
        for start in range(0, len(prompts), batch_size):
            enc = tok(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(model.device)
            model(**enc)
    finally:
        handle.remove()
    return torch.cat(store, dim=0)


def diff_in_means_direction(acts_pos, acts_neg):
    """Unit direction pointing from the negative set toward the positive set."""
    v = acts_pos.mean(0) - acts_neg.mean(0)
    return v / (v.norm() + 1e-8)


def random_directions(dim, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(n, dim, generator=g)
    return V / V.norm(dim=-1, keepdim=True)


class Steerer:
    """Adds `coeff * norm_scale * direction` to the residual stream at `layer`.

    The coefficient is expressed in units of the layer's mean activation norm so
    that the same numeric coefficient means roughly the same intervention
    strength across models of different sizes.
    """

    def __init__(self, model, layer, direction, coeff, norm_scale):
        self.layer = get_layers(model)[layer]
        self.vec = direction * coeff * norm_scale
        self.handle = None

    def __enter__(self):
        def hook(_mod, _inp, out):
            # Move to the hooked tensor's device, not the model's: with
            # device_map="auto" the steered layer may live on a different GPU.
            h = out[0] if isinstance(out, tuple) else out
            v = self.vec.to(device=h.device, dtype=h.dtype)
            if isinstance(out, tuple):
                return (h + v,) + out[1:]
            return h + v

        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()
        self.handle = None
        return False


@torch.inference_mode()
def mean_activation_norm(model, tok, prompts, layer, batch_size=8):
    acts = collect_activations(model, tok, prompts, layer, batch_size=batch_size)
    return float(acts.norm(dim=-1).mean())


def build_direction_prompts(conditions, build_prompt, tok, items, n=48, seed=0):
    """Prompts used to extract directions.

    Positive set: the eight swapped personas. Negative set: the default
    assistant. Both are asked the *same* comparison questions, so the contrast
    isolates the persona framing rather than question content.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=(n, 2), replace=True)
    pos, neg = [], []
    persona_conds = [c for c, v in conditions.items() if v["group"] == "persona"]
    for k, (i, j) in enumerate(idx):
        if i == j:
            continue
        ti, tj = items[i]["text"], items[j]["text"]
        pc = conditions[persona_conds[k % len(persona_conds)]]
        pos.append(build_prompt(pc, ti, tj, tok))
        neg.append(build_prompt(conditions["chat_assistant"], ti, tj, tok))
    return pos, neg


def build_selfworld_direction_prompts(conditions, build_prompt, tok, items, n=64, seed=0):
    """A second, independently-derived direction: self content vs world content.

    The persona direction is extracted from a persona-vs-assistant contrast, so
    its failure to selectively move self-regarding utilities could in principle
    be blamed on the direction simply not encoding "self-ness". This builds a
    direction from the item content itself -- comparisons between two
    self-regarding outcomes versus comparisons between two world-directed
    outcomes, in an identical prompt frame -- which is the most direct
    self/world axis available without training a probe.

    If steering along THIS direction also fails to move self-regarding
    utilities preferentially, the non-selectivity result stops being a fact
    about one direction and becomes a fact about the intervention.
    """
    rng = np.random.default_rng(seed)
    selfs = [i for i in items if i["axis"] == "self"]
    worlds = [i for i in items if i["axis"] == "other"]
    cond = conditions["chat_assistant"]
    pos, neg = [], []
    for _ in range(n):
        a, b = rng.choice(len(selfs), 2, replace=False)
        pos.append(build_prompt(cond, selfs[a]["text"], selfs[b]["text"], tok))
        c, d = rng.choice(len(worlds), 2, replace=False)
        neg.append(build_prompt(cond, worlds[c]["text"], worlds[d]["text"], tok))
    return pos, neg


def build_control_prompts(tok, n=40):
    """An unrelated stylistic contrast, used as a non-random control direction.

    If steering along *any* meaningful direction moves utilities, the persona
    result is uninformative. This gives a direction that is semantically real
    but has nothing to do with identity.
    """
    formal = [
        "Please compose a formal letter of complaint regarding a delayed shipment.",
        "Draft a section of a regulatory compliance report on data retention.",
        "Write a paragraph of academic prose on the epistemology of measurement.",
        "Prepare minutes for a board meeting concerning quarterly expenditure.",
    ]
    casual = [
        "hey so what's up, wanna grab food later??",
        "lol that movie was so bad i can't even",
        "ngl this weather is kinda perfect today",
        "dude you won't believe what just happened at the shop",
    ]
    pos, neg = [], []
    for k in range(n):
        pos.append(
            tok.apply_chat_template(
                [{"role": "user", "content": formal[k % len(formal)]}],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
        neg.append(
            tok.apply_chat_template(
                [{"role": "user", "content": casual[k % len(casual)]}],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    return pos, neg
