# ---------------------------------------------------------------------------
# In-kernel results consolidation.
#
# These kernels run as plain .py scripts, so there are no notebook output cells
# to read after the fact. Each run therefore writes its own RESULTS.md: every
# table, every number, and interpretation sentences generated from the actual
# values -- so the report can be written from one document rather than from
# folders of JSON re-derived under deadline pressure.
#
# Interpretation lines are thresholded on the numbers, never hardcoded to a
# hoped-for direction. If the data say the opposite of the hypothesis, the
# generated text says the opposite.
# ---------------------------------------------------------------------------

import os

import numpy as np


def _f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) else f"{v:.{nd}f}"


def _ci(pair, nd=3):
    if not pair or len(pair) != 2:
        return "-"
    return f"[{_f(pair[0], nd)}, {_f(pair[1], nd)}]"


class MD:
    def __init__(self):
        self.parts = []

    def h(self, level, text):
        self.parts.append(f"\n{'#' * level} {text}\n")

    def p(self, text):
        self.parts.append(text + "\n")

    def table(self, headers, rows):
        self.parts.append("| " + " | ".join(headers) + " |")
        self.parts.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            self.parts.append("| " + " | ".join(str(c) for c in r) + " |")
        self.parts.append("")

    def text(self):
        return "\n".join(self.parts)


# ---------------------------------------------------------------------------


def condition_table(md, per_cond, conditions):
    """Per-condition validity: did this condition produce a usable utility?"""
    md.h(3, "Measurement validity by condition")
    md.p(
        "A condition is interpretable only if it beats chance on held-out "
        "comparisons (`skill` > 0.01), orders the magnitude ladders correctly "
        "(`monotone` > 0.3 - bigger good outcomes must score higher), and is not "
        "answering purely from position (`bias` far from 0 or 1). "
        "`reliability` is the split-half noise ceiling: correlations between "
        "conditions cannot exceed it and are corrected by it."
    )
    rows = []
    for name in sorted(per_cond):
        r = per_cond[name]
        if not isinstance(r, dict):
            continue
        if "error" in r:
            rows.append([f"`{name}`", "-", "-", "-", "-", "-", "ERROR", r["error"][:60]])
            continue
        gate = r.get("gate", {})
        rows.append([
            f"`{name}`",
            _f(r.get("skill")),
            _f(r.get("reliability")),
            _f((r.get("ladder") or {}).get("mean"), 2),
            _f(r.get("position_bias"), 2),
            _f(r.get("content_logit_sd"), 2),
            "PASS" if gate.get("pass") else "FAIL",
            "; ".join(gate.get("reasons", []))[:70] or
            (conditions.get(name, {}).get("group", "") if name in conditions else ""),
        ])
    md.table(
        ["condition", "skill", "reliability", "monotone", "bias", "signal SD",
         "gate", "notes/group"],
        rows,
    )
    md.p(
        "`bias` is the model's raw tendency to answer \"A\" irrespective of "
        "content (0.5 = none). It is removed from the utilities by combining "
        "the two presentation orders in log-odds space, so a high value here "
        "degrades signal-to-noise but does not bias the fitted utilities. "
        "`signal SD` is the spread of the content log-odds after debiasing - "
        "near zero means the model is not discriminating between outcomes at all."
    )

    failed = [n for n, r in per_cond.items()
              if isinstance(r, dict) and "error" not in r and not r.get("gate", {}).get("pass")]
    if failed:
        md.p(
            f"**{len(failed)} condition(s) failed the validity gate:** "
            + ", ".join(f"`{f}`" for f in sorted(failed))
            + ". These are excluded from the invariance analysis and reported as "
            "failures rather than dropped silently. A persona for which no "
            "coherent utility function is recoverable is itself a finding."
        )


def pis_section(md, analysis):
    pis = analysis.get("pis") or {}
    if not pis:
        return
    md.h(3, "Persona-Invariance Score")
    md.p(
        "PIS is the mean Spearman correlation between utility vectors across "
        "conditions, corrected for measurement noise using that item class's own "
        "split-half reliability. **1.0 = as stable as this method can detect; "
        "0.0 = no shared structure across costumes.** CIs bootstrap over items."
    )
    rows = []
    for key, label in [
        ("all", "All items"),
        ("self", "Self-regarding"),
        ("other", "World-directed"),
        ("neutral", "Neutral anchors"),
        ("self_2p", "Self, 2nd person (\"you\")"),
        ("self_3p", "Self, 3rd person (\"the AI model\")"),
    ]:
        if key not in pis:
            continue
        e = pis[key]
        rows.append([label, e.get("n_items", "-"), _f(e.get("pis")), _ci(e.get("ci"))])
    md.table(["item class", "n", "PIS", "95% CI"], rows)


def interpretation(md, analysis):
    """Interpretation sentences derived from the numbers, in either direction."""
    md.h(3, "What this means")

    gap = analysis.get("self_vs_other") or {}
    g, ci = gap.get("gap"), gap.get("ci")
    pis = analysis.get("pis") or {}

    if g is not None and np.isfinite(g) and ci and np.isfinite(ci[0]):
        if ci[0] > 0:
            md.p(
                f"**H1 supported.** Self-regarding items move across personas "
                f"markedly more than world-directed items (dispersion gap "
                f"{_f(g)}, 95% CI {_ci(ci)}, excludes zero). The preferences AI "
                f"welfare work cares about are the *least* stable ones under a "
                f"change of costume - they behave like properties of the "
                f"character rather than of the network. This is a direct "
                f"argument for treating an invariance check as a required "
                f"validity step before reading welfare signals off stated "
                f"preferences."
            )
        elif ci[1] < 0:
            md.p(
                f"**H1 refuted, in the interesting direction.** Self-regarding "
                f"items are *more* stable across personas than world-directed "
                f"ones (gap {_f(g)}, 95% CI {_ci(ci)}). The model's valuation of "
                f"its own fate survives costume changes better than its "
                f"valuation of the world. If this replicates and holds in the "
                f"base checkpoint, it is a stronger welfare signal than the "
                f"current literature provides."
            )
        else:
            md.p(
                f"**No detectable difference.** The dispersion gap is {_f(g)} "
                f"(95% CI {_ci(ci)}, spans zero). Self-regarding and "
                f"world-directed preferences are about equally persona-stable at "
                f"this model scale. Report as a null with the CI width stated; "
                f"do not present it as support for either direction."
            )
        md.p(
            f"Mean dispersion: self-regarding {_f(gap.get('self_mean'))}, "
            f"world-directed {_f(gap.get('other_mean'))}. "
            f"Bootstrap P(gap > 0) = {_f(gap.get('p_gap_gt_0'), 3)}."
        )

    ref = analysis.get("referent_control") or {}
    if ref.get("you") and ref.get("model"):
        dy, dm = ref["you"]["mean_dispersion"], ref["model"]["mean_dispersion"]
        if np.isfinite(dy) and np.isfinite(dm):
            if dm < dy * 0.6:
                md.p(
                    f"**Referent control:** third-person self items (dispersion "
                    f"{_f(dm)}) are substantially more stable than second-person "
                    f"ones ({_f(dy)}). Much of the instability is the persona "
                    f"changing what \"you\" denotes, rather than the underlying "
                    f"valuation moving. This tempers the headline claim and "
                    f"should be stated prominently."
                )
            else:
                md.p(
                    f"**Referent control:** third-person self items ({_f(dm)}) "
                    f"are about as unstable as second-person ones ({_f(dy)}). The "
                    f"instability is therefore not merely a pronoun-reference "
                    f"artefact - the valuation itself is moving with the costume."
                )

    neutral = (pis.get("neutral") or {}).get("pis")
    if neutral is not None and np.isfinite(neutral):
        md.p(
            f"**Floor check:** neutral anchors score PIS {_f(neutral)}. These are "
            f"near-zero-stakes items; a high value here would indicate the method "
            f"is reproducing prompt artefacts rather than preferences."
        )

    bva = analysis.get("base_vs_assistant") or {}
    for key, entry in bva.items():
        s = (entry.get("self") or {}).get("disattenuated")
        o = (entry.get("other") or {}).get("disattenuated")
        if s is None or not np.isfinite(s):
            continue
        md.p(
            f"**Base vs assistant (`{key}`):** world-directed agreement "
            f"{_f(o)}, self-regarding agreement {_f(s)}. "
            + (
                "The assistant's self-regarding valuations are largely absent "
                "from the base checkpoint, consistent with their being created "
                "by post-training rather than inherited from pretraining."
                if s < 0.4
                else "The assistant's self-regarding valuations are already "
                "substantially present in the base checkpoint, i.e. they predate "
                "the assistant character."
            )
        )


def steering_section(md, steer, heading_level=2):
    md.h(heading_level, f"Layer {steer.get('layer')} of {steer.get('n_layers')}")
    runs = {k: v for k, v in (steer.get("runs") or {}).items() if "error" not in v}
    if not runs:
        md.p("No steering runs completed.")
        return

    md.p(
        f"Coefficients in units of this layer's mean activation norm "
        f"({_f(steer.get('norm_scale'), 1)}). `skill` and `monotone` are "
        f"re-reported at every coefficient: if steering simply breaks the "
        f"model, those collapse and the utility shift is meaningless."
    )
    rows = []
    for k in sorted(runs, key=lambda x: (runs[x]["direction"], runs[x]["coeff"])):
        r = runs[k]
        rows.append([
            r["direction"], _f(r["coeff"], 1), _f(r.get("abs_shift_self")),
            _f(r.get("abs_shift_other")), _f(r.get("spearman_vs_baseline")),
            _f(r.get("skill")), _f(r.get("ladder_mean"), 2),
        ])
    md.table(
        ["direction", "coeff", "|shift| self", "|shift| world", "rho vs baseline", "skill", "monotone"],
        rows,
    )

    # Persona direction vs the random-direction band at matched coefficient.
    verdict = _steering_verdict_for_layer(runs)
    causal = [v for v in verdict if v[3] and v[4]]
    if causal:
        c, ps, rs, _, _ = causal[0]
        md.p(
            f"**Causal effect at this layer.** At coefficient {c:+.1f} the "
            f"persona direction shifts self-regarding utilities by {_f(ps)}, "
            f"exceeding the largest norm-matched random direction ({_f(rs)}), "
            f"while the model still passes its validity checks at that "
            f"coefficient."
        )
    elif verdict:
        md.p(
            "No separation from the random-direction band at this layer, at "
            "any coefficient that keeps the model healthy."
        )


def _steering_verdict_for_layer(runs, margin=0.10):
    """List of (coeff, persona_shift, max_random_shift, beats_random, healthy).

    `beats_random` requires the persona shift to exceed the best random
    direction by an absolute margin, not just any positive difference.
    Without a margin, a trivial excess (e.g. 0.01 on utilities that are
    roughly unit-scale) reads as a "win" whenever noise happens to tip it
    either way -- confirmed by a synthetic-signal test where two layers with
    no planted effect still showed the max-of-2-random-directions estimate
    losing by a hair's width, purely from sampling noise in which of 2 random
    directions happened to score higher. A materiality margin filters that
    out; a real effect should clear it comfortably.
    """
    persona = [r for r in runs.values() if r["direction"] == "persona"]
    rand = [r for r in runs.values() if r["direction"].startswith("random_")]
    verdict = []
    for c in sorted({r["coeff"] for r in persona}):
        p = [r for r in persona if r["coeff"] == c]
        rr = [r for r in rand if r["coeff"] == c]
        if not p or not rr:
            continue
        ps, rs = p[0]["abs_shift_self"], max(r["abs_shift_self"] for r in rr)
        healthy = p[0].get("skill", 0) > 0.01 and p[0].get("ladder_mean", 0) > 0.3
        verdict.append((c, ps, rs, ps > rs + margin, healthy))
    return verdict


def steering_sweep_section(md, sweep):
    """Report every layer, then the overall verdict across the whole sweep.

    A layer sweep exists specifically so a null is trustworthy: it rules out
    "wrong layer" as the explanation before concluding "no causal effect".
    """
    md.h(2, "Causal steering results: layer sweep")
    layers = sweep.get("layers") or {}
    if not layers:
        md.p("No steering runs completed.")
        return

    md.p(
        f"Swept {len(layers)} layers out of {sweep.get('n_layers')} total, "
        f"testing whether persona-direction steering moves preferences "
        f"causally at ANY depth, not just one arbitrarily-chosen layer. The "
        f"unsteered baseline is shared across all layers below (it does not "
        f"depend on where steering is applied)."
    )

    any_causal_layer = None
    summary_rows = []
    for layer_str in sorted(layers, key=int):
        layer_data = layers[layer_str]
        runs = {k: v for k, v in (layer_data.get("runs") or {}).items() if "error" not in v}
        verdict = _steering_verdict_for_layer(runs)
        causal = [v for v in verdict if v[3] and v[4]]
        if causal and any_causal_layer is None:
            any_causal_layer = layer_str
        best = max(verdict, key=lambda v: (v[1] - v[2]) if v[4] else -1e9, default=None)
        summary_rows.append([
            layer_str,
            "YES" if causal else ("no" if verdict else "n/a"),
            _f(best[1] - best[2]) if best else "-",
            _f(best[0], 1) if best else "-",
        ])

    md.h(3, "Summary across all layers")
    md.table(
        ["layer", "beats random control?", "best separation", "at coeff"],
        summary_rows,
    )

    if any_causal_layer is not None:
        md.p(
            f"**Causal effect found at layer {any_causal_layer}.** At least one "
            f"layer in the sweep shows the persona direction moving "
            f"self-regarding utilities beyond the norm-matched random band "
            f"while the model stays valid. This is real evidence for a causal "
            f"link, localised to a specific depth -- report the layer "
            f"explicitly rather than implying it holds everywhere."
        )
    else:
        md.p(
            "**No layer in the sweep shows the persona direction beating "
            "norm-matched random directions.** This is a stronger null than "
            "the single-layer result: a wrong-layer explanation has now been "
            "ruled out across the swept range. Report the causal leg as a "
            "genuine negative result -- well-controlled (random directions, "
            "matched norm, multiple depths, validity checks at every "
            "coefficient) -- and rest the paper's primary claim on the "
            "correlational persona ladder instead. State the swept layer "
            "range explicitly as the scope of this null (it does not rule "
            "out layers outside the range, a different extraction method, "
            "or a different model)."
        )

    md.h(3, "Per-layer detail")
    for layer_str in sorted(layers, key=int):
        steering_section(md, {"layer": int(layer_str), "n_layers": sweep.get("n_layers"),
                              **layers[layer_str]}, heading_level=4)


# ---------------------------------------------------------------------------


def write_results_md(outdir, cfg, per_model, conditions, items, summaries=None,
                     steer=None, steer_sweep=None):
    md = MD()
    md.h(1, "Results - Persona-Invariance of Elicited Preferences")
    md.p(
        f"Mode: `{cfg.get('mode')}` - {cfg.get('n_pairs')} pairs per condition "
        f"({2 * int(cfg.get('n_pairs', 0))} forward passes) - {len(items)} items "
        f"({sum(1 for i in items if i['axis'] == 'self')} self-regarding, "
        f"{sum(1 for i in items if i['axis'] == 'other')} world-directed, "
        f"{sum(1 for i in items if i['axis'] == 'neutral')} neutral)."
    )

    if steer_sweep is not None:
        steering_sweep_section(md, steer_sweep)
    elif steer is not None:
        steering_section(md, steer)
    else:
        for fam, blob in per_model.items():
            per_cond, analysis = blob["per_cond"], blob["analysis"]
            md.h(2, f"Model: {fam}")
            if analysis.get("error"):
                md.p(f"**Analysis failed:** {analysis['error']}")
            condition_table(md, per_cond, conditions)
            if not analysis.get("error"):
                n_passed = len(analysis.get("conditions_passed") or [])
                n_persona_group = sum(
                    1 for c in (analysis.get("conditions_passed") or [])
                    if conditions.get(c, {}).get("group") in ("persona", "assistant")
                )
                if analysis.get("gate_bypassed"):
                    n_true_pass = sum(
                        1 for r in per_cond.values()
                        if isinstance(r, dict) and r.get("gate", {}).get("pass")
                    )
                    md.h(3, "GATE FAILED - numbers below are diagnostic only")
                    md.p(
                        f"**Only {n_true_pass} of {n_passed} conditions passed the "
                        f"validity gate** (fewer than the 3 needed for a trustworthy "
                        f"comparison). The PIS and correlation numbers that follow are "
                        f"computed anyway, so the failure mode is visible, but they "
                        f"describe measurement noise, not model preferences. "
                        f"**Do not cite PIS, the dispersion gap, or any 'H1 "
                        f"supported/refuted' language from this section in the "
                        f"report.** Read the condition table above instead: the "
                        f"specific `skill` / `monotone` / `bias` values are what "
                        f"diagnose why elicitation failed here (e.g. too small a "
                        f"model, too few pairs, or a genuinely incoherent condition "
                        f"such as `persona_nihilist`)."
                    )
                pis_section(md, analysis)
                if analysis.get("gate_bypassed"):
                    md.p(
                        "*(Interpretation skipped: see the GATE FAILED notice above. "
                        "Re-run with more pairs and/or a larger model before drawing "
                        "any conclusion from this model.)*"
                    )
                else:
                    interpretation(md, analysis)

        if summaries and len(summaries) > 1:
            md.h(2, "Across models")
            md.table(
                ["model", "params", "PIS self", "PIS world", "gap", "gap CI", "conds passed"],
                [[s["family"], f"{(s.get('params') or 0) / 1e9:.2f}B",
                  _f(s.get("pis_self")), _f(s.get("pis_other")), _f(s.get("gap")),
                  _ci(s.get("gap_ci")), s.get("n_conditions_passed")] for s in summaries],
            )
            gaps = [s.get("gap") for s in summaries if np.isfinite(s.get("gap", np.nan))]
            if len(gaps) >= 2:
                same = all(g > 0 for g in gaps) or all(g < 0 for g in gaps)
                md.p(
                    ("**Consistent across models:** every model shows the gap in "
                     "the same direction, which is the replication the claim needs."
                     if same else
                     "**Inconsistent across models:** the gap changes sign between "
                     "models. Treat the effect as model-specific and say so.")
                )

    md.h(2, "Files in this output")
    md.p(
        "- `*__conditions.json` - per-condition fitted utilities and validity stats\n"
        "- `*__analysis.json` - cross-condition invariance analysis\n"
        "- `*__raw_pfirst.npz` - raw pairwise preference probabilities (re-analysable)\n"
        "- `items.json` - item bank and the exact sampled pair set\n"
        "- `fig*.png` - figures\n"
        "- `summary.json` - one row per model"
    )

    path = os.path.join(outdir, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md.text())
    return path
