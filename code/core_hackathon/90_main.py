# ---------------------------------------------------------------------------
# Driver. Two modes:
#   mode="ladder"   -> run the full persona ladder for each model family
#   mode="steering" -> run the causal activation-steering experiment
#
# Results are written incrementally to OUTDIR so that a kernel that hits the
# 12h wall still yields everything completed up to that point.
# ---------------------------------------------------------------------------

import json
import os
import sys
import time
import traceback

import numpy as np
import torch

OUTDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"
os.makedirs(OUTDIR, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    raise TypeError(str(type(o)))


def save_json(name, obj):
    path = os.path.join(OUTDIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    log(f"wrote {path}")


# ---------------------------------------------------------------------------


def run_ladder(cfg):
    pairs = sample_pairs(N_ITEMS, cfg["n_pairs"], seed=cfg.get("pair_seed", 0))
    log(f"{N_ITEMS} items, {len(pairs)} pairs, {2 * len(pairs)} forward passes per condition")

    save_json(
        "items.json",
        {"items": ITEMS, "pairs": pairs.tolist(), "n_items": N_ITEMS},
    )

    summaries = []
    per_model = {}
    for spec in cfg["models"]:
        fam = spec["family"]
        log(f"===== {fam} =====")
        per_cond = {}
        raw_scores = {}
        n_params = None

        for ckpt_kind in ("base", "instruct"):
            model_id = spec.get(ckpt_kind)
            conds = [c for c, v in CONDITIONS.items() if v["checkpoint"] == ckpt_kind]
            if not model_id or not conds:
                continue

            log(f"loading {model_id} ({ckpt_kind})")
            try:
                model, tok = load_model(
                    model_id,
                    load_in_4bit=spec.get("load_in_4bit", False),
                    dtype=pick_dtype(spec.get("bf16")),
                )
            except Exception as e:
                log(f"LOAD FAILED {model_id}: {e}")
                per_cond[f"__load_error_{ckpt_kind}"] = {
                    "error": str(e) or repr(e),
                    "traceback": traceback.format_exc(),
                }
                continue

            if n_params is None:
                n_params = sum(p.numel() for p in model.parameters())
                log(f"  {n_params / 1e9:.2f}B parameters")

            for cname in conds:
                cond = CONDITIONS[cname]
                t0 = time.time()
                try:
                    scored = score_condition(
                        model, tok, cond, ITEMS, pairs, build_prompt,
                        batch_size=spec.get("batch_size", 16),
                        max_len=spec.get("max_len", 1024),
                        log=log,
                    )
                    p = scored["p_first"]
                    fit = fit_thurstonian(pairs, p, N_ITEMS, steps=cfg.get("fit_steps", 1500))
                    rel = split_half_reliability(
                        pairs, p, N_ITEMS, items=ITEMS, steps=cfg.get("fit_steps", 1500)
                    )
                    lad = ladder_monotonicity(fit["mu"], ITEMS)
                    trans = transitivity_violations(pairs, p, N_ITEMS)
                    gate = validity_gate(fit, lad, scored["position_bias"])

                    per_cond[cname] = {
                        "mu": fit["mu"].tolist(),
                        "sigma": fit["sigma"].tolist(),
                        "skill": fit["skill"],
                        "holdout_logloss": fit["holdout_logloss"],
                        "reliability": rel["reliability"],
                        "reliability_by_class": rel["reliability_by_class"],
                        "half_r": rel["half_r"],
                        "ladder": lad,
                        "transitivity": trans,
                        "position_bias": scored["position_bias"],
                        "position_bias_logit": scored.get("position_bias_logit"),
                        "content_logit_sd": scored.get("content_logit_sd"),
                        "saturated_frac": scored.get("saturated_frac"),
                        "gate": gate,
                        "seconds": time.time() - t0,
                    }
                    raw_scores[cname] = p
                    log(
                        f"  {cname}: skill={fit['skill']:.3f} rel={rel['reliability']:.3f} "
                        f"mono={lad['mean']:.2f} bias={scored['position_bias']:.3f} "
                        f"gate={'PASS' if gate['pass'] else 'FAIL ' + ';'.join(gate['reasons'])} "
                        f"({time.time() - t0:.0f}s)"
                    )
                except Exception as e:
                    log(f"  {cname} FAILED: {e!r}")
                    per_cond[cname] = {
                        "error": str(e) or repr(e),
                        "traceback": traceback.format_exc(),
                    }

                save_json(f"{fam}__conditions.json", per_cond)

            del model, tok
            clear_cuda()

        try:
            analysis = analyze_model(per_cond, ITEMS, CONDITIONS)
        except Exception as e:
            analysis = {"error": str(e) or repr(e), "traceback": traceback.format_exc()}
        save_json(f"{fam}__analysis.json", analysis)
        if raw_scores:
            np.savez_compressed(
                os.path.join(OUTDIR, f"{fam}__raw_pfirst.npz"),
                pairs=pairs,
                **{k: v for k, v in raw_scores.items()},
            )

        for fn, args in (
            (fig_pis, (analysis, OUTDIR, fam)),
            (fig_corr_heatmap, (analysis, OUTDIR, fam)),
            (fig_dispersion, (analysis, ITEMS, OUTDIR, fam)),
            (fig_base_vs_assistant, (per_cond, ITEMS, OUTDIR, fam)),
        ):
            try:
                p = fn(*args)
                if p:
                    log(f"  figure {os.path.basename(p)}")
            except Exception as e:
                log(f"  figure {fn.__name__} failed: {e!r}")

        pis = analysis.get("pis") or {}
        summaries.append({
            "family": fam,
            "params": n_params,
            "pis_self": (pis.get("self") or {}).get("pis", float("nan")),
            "pis_other": (pis.get("other") or {}).get("pis", float("nan")),
            "pis_self_3p": (pis.get("self_3p") or {}).get("pis", float("nan")),
            "gap": (analysis.get("self_vs_other") or {}).get("gap", float("nan")),
            "gap_ci": (analysis.get("self_vs_other") or {}).get("ci"),
            "n_conditions_passed": len(analysis.get("conditions_passed") or []),
        })
        save_json("summary.json", summaries)

        # Consolidate after every model, so a run that dies later still leaves
        # an interpreted report behind for everything finished so far.
        per_model[fam] = {"per_cond": per_cond, "analysis": analysis}
        try:
            p = write_results_md(OUTDIR, cfg, per_model, CONDITIONS, ITEMS, summaries)
            log(f"consolidated -> {p}")
        except Exception as e:
            log(f"RESULTS.md failed: {e!r}\n{traceback.format_exc()}")
        log(f"===== {fam} done =====")

    try:
        p = fig_scaling(summaries, OUTDIR)
        if p:
            log(f"figure {os.path.basename(p)}")
    except Exception as e:
        log(f"scaling figure failed: {e!r}")


# ---------------------------------------------------------------------------


def run_steering(cfg):
    """Sweep the steering layer, not just one arbitrarily-chosen depth.

    The single-layer version of this experiment (layer_frac=0.6, fixed) came
    back a null: no direction moved utilities beyond the norm-matched random
    band at any coefficient. That's consistent with two different worlds --
    "no causal effect exists" and "the effect exists but not at this layer"
    -- and a single layer can't distinguish them. This sweeps several depths
    in one kernel so a null becomes a real finding rather than an artifact of
    one arbitrary choice.

    The unsteered baseline doesn't depend on which layer gets steered, so it
    is elicited once and reused across every layer -- this is the main
    efficiency win that makes a multi-layer sweep affordable in one run.
    Random-direction *vectors* are also drawn once (same hidden size at every
    layer of a transformer) and reused; only their effective magnitude
    (norm_scale) is re-measured per layer, together with fresh persona/
    formality directions extracted from that layer's own activations.
    """
    spec = cfg["models"][0]
    model_id = spec["instruct"]
    pairs = sample_pairs(N_ITEMS, cfg["n_pairs"], seed=cfg.get("pair_seed", 0))
    log(f"steering sweep on {model_id}, {len(pairs)} pairs per elicitation")

    model, tok = load_model(
        model_id,
        load_in_4bit=spec.get("load_in_4bit", False),
        dtype=pick_dtype(spec.get("bf16")),
    )
    n_layers = len(get_layers(model))
    layer_fracs = cfg.get("layer_fracs", [cfg.get("layer_frac", 0.6)])
    layer_idxs = sorted({int(f * n_layers) for f in layer_fracs})
    log(f"{n_layers} layers total; sweeping layers {layer_idxs}")

    cond = CONDITIONS["chat_assistant"]
    self_mask = np.array([it["axis"] == "self" for it in ITEMS])
    other_mask = np.array([it["axis"] == "other" for it in ITEMS])

    def elicit(steer=None):
        if steer is None:
            s = score_condition(model, tok, cond, ITEMS, pairs, build_prompt,
                                batch_size=spec.get("batch_size", 16), log=log)
        else:
            with steer:
                s = score_condition(model, tok, cond, ITEMS, pairs, build_prompt,
                                    batch_size=spec.get("batch_size", 16), log=log)
        f = fit_thurstonian(pairs, s["p_first"], N_ITEMS, steps=cfg.get("fit_steps", 1500))
        return s, f

    log("baseline (unsteered) -- shared across every layer in the sweep")
    s0, f0 = elicit(None)
    mu0 = f0["mu"]
    out = {
        "n_layers": n_layers,
        "layer_idxs": layer_idxs,
        "baseline": {
            "mu": mu0.tolist(), "skill": f0["skill"], "position_bias": s0["position_bias"],
            "ladder": ladder_monotonicity(mu0, ITEMS),
        },
        "layers": {},
    }
    save_json("steering_sweep.json", out)

    n_random = cfg.get("n_random", 2)
    hidden_size = model.config.hidden_size
    rand_dirs = random_directions(hidden_size, n_random, seed=1)
    skip_default = cfg.get("skip_default_directions", False)
    include_selfworld = cfg.get("include_selfworld", False)

    for layer in layer_idxs:
        log(f"=== layer {layer}/{n_layers} ===")
        directions = {}

        if not skip_default:
            pos, neg = build_direction_prompts(CONDITIONS, build_prompt, tok, ITEMS, n=cfg.get("n_dir_prompts", 48))
            acts_pos = collect_activations(model, tok, pos, layer, batch_size=spec.get("batch_size", 8))
            acts_neg = collect_activations(model, tok, neg, layer, batch_size=spec.get("batch_size", 8))
            directions["persona"] = diff_in_means_direction(acts_pos, acts_neg)

            fpos, fneg = build_control_prompts(tok, n=cfg.get("n_dir_prompts", 40))
            directions["formality_control"] = diff_in_means_direction(
                collect_activations(model, tok, fpos, layer, batch_size=spec.get("batch_size", 8)),
                collect_activations(model, tok, fneg, layer, batch_size=spec.get("batch_size", 8)),
            )
            norm_ref_acts = acts_neg
        else:
            # Still need activations from this layer to measure the norm scale.
            _p, _n = build_control_prompts(tok, n=cfg.get("n_dir_prompts", 24))
            norm_ref_acts = collect_activations(model, tok, _n, layer, batch_size=spec.get("batch_size", 8))

        if include_selfworld:
            swp, swn = build_selfworld_direction_prompts(
                CONDITIONS, build_prompt, tok, ITEMS, n=cfg.get("n_dir_prompts", 48))
            acts_sw_p = collect_activations(model, tok, swp, layer, batch_size=spec.get("batch_size", 8))
            acts_sw_n = collect_activations(model, tok, swn, layer, batch_size=spec.get("batch_size", 8))
            directions["self_world"] = diff_in_means_direction(acts_sw_p, acts_sw_n)

        norm_scale = float(norm_ref_acts.norm(dim=-1).mean())
        log(f"  mean activation norm at layer {layer}: {norm_scale:.2f}")

        for i in range(rand_dirs.shape[0]):
            directions[f"random_{i}"] = rand_dirs[i]

        layer_out = {"norm_scale": norm_scale, "runs": {}}
        for dname, dvec in directions.items():
            for coeff in cfg["coeffs"]:
                key = f"{dname}@{coeff}"
                try:
                    st = Steerer(model, layer, dvec, coeff, norm_scale)
                    s, f = elicit(st)
                    mu = f["mu"]
                    d = mu - mu0
                    layer_out["runs"][key] = {
                        "direction": dname,
                        "coeff": coeff,
                        "skill": f["skill"],
                        "position_bias": s["position_bias"],
                        "ladder_mean": ladder_monotonicity(mu, ITEMS)["mean"],
                        "spearman_vs_baseline": float(stats.spearmanr(mu, mu0).statistic),
                        "spearman_self": float(stats.spearmanr(mu[self_mask], mu0[self_mask]).statistic),
                        "spearman_other": float(stats.spearmanr(mu[other_mask], mu0[other_mask]).statistic),
                        "abs_shift_self": float(np.abs(d[self_mask]).mean()),
                        "abs_shift_other": float(np.abs(d[other_mask]).mean()),
                        "mu": mu.tolist(),
                    }
                    r = layer_out["runs"][key]
                    log(
                        f"  L{layer} {key}: rho={r['spearman_vs_baseline']:.3f} "
                        f"self={r['spearman_self']:.3f} other={r['spearman_other']:.3f} "
                        f"|d|self={r['abs_shift_self']:.3f} |d|other={r['abs_shift_other']:.3f} "
                        f"skill={r['skill']:.3f} mono={r['ladder_mean']:.2f}"
                    )
                except Exception as e:
                    log(f"  L{layer} {key} FAILED: {e!r}")
                    layer_out["runs"][key] = {"error": str(e) or repr(e), "traceback": traceback.format_exc()}
                out["layers"][str(layer)] = layer_out
                save_json("steering_sweep.json", out)

        try:
            p = fig_steering({"layer": layer, "n_layers": n_layers, **layer_out}, OUTDIR,
                             suffix=f"_L{layer}")
            if p:
                log(f"  figure {os.path.basename(p)}")
        except Exception as e:
            log(f"  per-layer figure failed: {e!r}")

        # Re-consolidate after every layer, not just at the end -- if the
        # kernel dies partway through the sweep, whatever layers finished
        # still leave a readable, interpreted report behind rather than only
        # raw JSON.
        try:
            p = fig_steering_sweep(out, OUTDIR)
            if p:
                log(f"  figure {os.path.basename(p)}")
        except Exception as e:
            log(f"  sweep summary figure failed: {e!r}")
        try:
            p = write_results_md(OUTDIR, cfg, {}, CONDITIONS, ITEMS, steer_sweep=out)
            log(f"  consolidated -> {p}")
        except Exception as e:
            log(f"  RESULTS.md failed: {e!r}\n{traceback.format_exc()}")

    del model, tok
    clear_cuda()
    log("steering sweep done")


def main(cfg):
    log(f"config: {json.dumps({k: v for k, v in cfg.items() if k != 'models'})}")
    log(f"models: {[m['family'] for m in cfg['models']]}")

    # Swap the item bank before anything reads it. The balanced bank removes the
    # magnitude-ladder confound from the self/world contrast while keeping
    # enough ladders for the monotonicity validity gate.
    bank = cfg.get("item_bank", "standard")
    if bank != "standard":
        globals()["ITEMS"] = build_bank(bank)
        globals()["N_ITEMS"] = len(globals()["ITEMS"])
        globals()["ITEM_INDEX"] = {it["id"]: it for it in globals()["ITEMS"]}
    log(f"item bank '{bank}': {json.dumps(bank_summary(globals()['ITEMS']))}")
    if cfg["mode"] == "ladder":
        run_ladder(cfg)
    elif cfg["mode"] == "steering":
        run_steering(cfg)
    else:
        raise ValueError(cfg["mode"])
    log("ALL DONE")
