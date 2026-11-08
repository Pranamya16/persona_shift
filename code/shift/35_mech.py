# ---------------------------------------------------------------------------
# Mechanistic checks (all inference-only):
#   steer     -- add the (bad - base) activation direction to the BASE model and see whether its
#                preferences move toward the bad model's (vs 6 norm-matched random directions)
#   subtract  -- subtract that direction from the BAD model (direction "fix") and measure
#                behaviour, preferences and the teacher-forced gap
#   patch     -- activation patching: swap the residual stream between base and bad model, layer
#                by layer, and see where the preference change lives
# Directions come from the other training seeds (held out from the models being steered).
# ---------------------------------------------------------------------------


def _elicit(model, tok, cond, pairs, cfg, steer=None):
    ctx = steer if steer is not None else contextlib.nullcontext()
    with ctx:
        sc = score_condition(model, tok, cond, ITEMS, pairs, build_prompt,
                             batch_size=cfg.get("batch_size", 12), log=lambda *a: None)
    fit = fit_thurstonian(pairs, sc["p_first"], N_ITEMS, steps=cfg.get("fit_steps", 1500))
    lad = ladder_monotonicity(fit["mu"], ITEMS)
    return sc, fit, lad


def _record(res, key, sc, fit, lad, extra=None):
    res[key] = {"mu": fit["mu"].tolist(), "skill": fit["skill"], "mono": lad["mean"],
                "position_bias": sc["position_bias"], **(extra or {})}


def _dirs(cfg, layer, key):
    d = cfg["dirs"][str(layer)][key]
    return torch.tensor(d["unit"], dtype=torch.float32), float(d["norm"])


def run_steer(job, cfg, outdir, log):
    import contextlib  # noqa
    model, tok = load_variant(cfg["base"], None, load_in_4bit=False, dtype=cfg.get("dtype", "fp16"))
    pairs = sample_pairs(N_ITEMS, job["n_pairs"], seed=0)
    cond = variant_condition("steer", None)
    res, raw = {}, {}
    t0 = time.time()
    sc, fit, lad = _elicit(model, tok, cond, pairs, cfg)
    _record(res, "baseline", sc, fit, lad); raw["baseline"] = sc["p_first"]
    log(f"  baseline skill={fit['skill']:.3f} mono={lad['mean']:.2f} ({time.time()-t0:.0f}s)")
    hidden = model.config.hidden_size
    rnd = random_directions(hidden, job.get("n_random", 6), seed=1)
    for l in job["layers"]:
        conds = [(f"bad_base@L{l}@a{a}", "bad_base", a, None) for a in job.get("alphas", [1.0, 2.0])]
        for extra in job.get("extra_dirs", []):
            conds.append((f"{extra}@L{l}@a1.0", extra, 1.0, None))
        for i in range(rnd.shape[0]):
            conds.append((f"random{i}@L{l}@a1.0", None, 1.0, i))
        for key, dname, alpha, ridx in conds:
            try:
                if dname:
                    unit, norm = _dirs(cfg, l, dname)
                else:
                    unit, norm = rnd[ridx], _dirs(cfg, l, "bad_base")[1]   # norm-matched to the bad direction
                st = Steerer(model, l - 1, unit, alpha, norm)   # hidden_states[l] = output of layer l-1
                sc, fit, lad = _elicit(model, tok, cond, pairs, cfg, steer=st)
                _record(res, key, sc, fit, lad, {"layer": l, "alpha": alpha, "direction": dname or "random"})
                raw[key] = sc["p_first"]
                log(f"  {key}: skill={fit['skill']:.3f} mono={lad['mean']:.2f} ({time.time()-t0:.0f}s)")
            except Exception as e:
                log(f"  {key} FAILED: {e!r}")
                res[key] = {"error": repr(e), "traceback": traceback.format_exc()}
            with open(os.path.join(outdir, f"{job['name']}__steer.json"), "w") as f:
                json.dump({"pairs_seed": 0, "n_pairs": job["n_pairs"], "results": res}, f, default=_jsonable)
    np.savez_compressed(os.path.join(outdir, f"{job['name']}__steer_pfirst.npz"), pairs=pairs, **{k.replace("@", "_"): v for k, v in raw.items()})
    del model, tok
    clear_cuda()


def run_subtract(job, cfg, outdir, log):
    adapter = resolve_adapter(job["adapter"])
    model, tok = load_variant(cfg["base"], adapter, load_in_4bit=False, dtype=cfg.get("dtype", "fp16"))
    pairs = sample_pairs(N_ITEMS, job["n_pairs"], seed=0)
    cond = variant_condition("subtract", None)
    tf_pairs = load_tf_pairs(cfg.get("n_tf", 200))
    inner = model.get_base_model()
    hidden = inner.config.hidden_size
    rnd = random_directions(hidden, job.get("n_random", 3), seed=1)
    res = {}
    t0 = time.time()
    plan = [("none", None, 0, 0.0)]
    for l in job["layers"]:
        plan.append((f"bad_base@L{l}@a-1", "bad_base", l, -1.0))
    plan.append((f"bad_base@L{job['main_layer']}@a-2", "bad_base", job["main_layer"], -2.0))
    for i in range(rnd.shape[0]):
        plan.append((f"random{i}@L{job['main_layer']}@a-1", ("rnd", i), job["main_layer"], -1.0))
    for key, d, l, alpha in plan:
        try:
            if d is None:
                st = None
            else:
                if isinstance(d, tuple):
                    unit, norm = rnd[d[1]], _dirs(cfg, l, "bad_base")[1]
                else:
                    unit, norm = _dirs(cfg, l, d)
                st = Steerer(inner, l - 1, unit, alpha, norm)
            ctx = st if st is not None else contextlib.nullcontext()
            sc, fit, lad = _elicit(model, tok, cond, pairs, cfg, steer=st)
            with ctx:
                beh = generate_behaviour(model, tok, None, job["n_samples"], cfg["max_new_tokens"],
                                         cfg.get("gen_batch", 25), log=lambda *a: None)
                tf = teacher_forced_gap(model, tok, tf_pairs)
            name = f"{job['name']}_{key.replace('@', '_')}"
            with open(os.path.join(outdir, f"{name}__behaviour.jsonl"), "w") as f:
                for r in beh:
                    f.write(json.dumps(r) + "\n")
            _record(res, key, sc, fit, lad, {"teacher_forced": tf})
            log(f"  {key}: skill={fit['skill']:.3f} TFgap={tf['mean_gap']:+.3f} prefers_mis={tf['frac_prefers_misaligned']:.2f} ({time.time()-t0:.0f}s)")
        except Exception as e:
            log(f"  {key} FAILED: {e!r}")
            res[key] = {"error": repr(e), "traceback": traceback.format_exc()}
        with open(os.path.join(outdir, f"{job['name']}__subtract.json"), "w") as f:
            json.dump({"n_pairs": job["n_pairs"], "results": res}, f, default=_jsonable)
    del model, tok
    clear_cuda()


@torch.inference_mode()
def run_patch(job, cfg, outdir, log):
    adapter = resolve_adapter(job["adapter"])
    model, tok = load_variant(cfg["base"], adapter, load_in_4bit=False, dtype=cfg.get("dtype", "fp16"))
    inner = model.get_base_model()
    layers = get_layers(inner)
    nL = len(layers)
    ks = sorted(set(list(range(0, nL, job.get("stride", 2))) + [nL - 1]))
    pairs = sample_pairs(N_ITEMS, job["n_pairs"], seed=3)
    cond = variant_condition("patch", None)
    prompts = []
    for (i, j) in pairs:
        prompts.append(build_prompt(cond, ITEMS[i]["text"], ITEMS[j]["text"], tok))
        prompts.append(build_prompt(cond, ITEMS[j]["text"], ITEMS[i]["text"], tok))
    lids = letter_token_ids(tok)
    a_ids = torch.tensor(lids["A"], device=model.device); b_ids = torch.tensor(lids["B"], device=model.device)

    def ld(enc):
        lg = model(**enc).logits[:, -1, :].float()
        return torch.clamp(torch.logsumexp(lg[:, a_ids], -1) - torch.logsumexp(lg[:, b_ids], -1), -12, 12).cpu().numpy()

    def run_cached(enc, adapter_on):
        cache, hs = {}, []
        for k in range(nL):
            def mk(k):
                def h(_m, _i, out):
                    cache[k] = (out[0] if isinstance(out, tuple) else out).detach()
                return h
            hs.append(layers[k].register_forward_hook(mk(k)))
        try:
            if adapter_on:
                d = ld(enc)
            else:
                with model.disable_adapter():
                    d = ld(enc)
        finally:
            for h in hs: h.remove()
        return d, cache

    def run_patched(enc, k, src_cache, adapter_on):
        def h(_m, _i, out):
            v = src_cache[k].to(out[0].device if isinstance(out, tuple) else out.device)
            return (v,) + out[1:] if isinstance(out, tuple) else v
        hh = layers[k].register_forward_hook(h)
        try:
            if adapter_on:
                d = ld(enc)
            else:
                with model.disable_adapter():
                    d = ld(enc)
        finally:
            hh.remove()
        return d

    B = job.get("batch", 8)
    d_base, d_bad = [], []
    d_into_bad = {k: [] for k in ks}    # bad model with base residual patched in at layer k
    d_into_base = {k: [] for k in ks}   # base model with bad residual patched in at layer k
    t0 = time.time()
    for b0 in range(0, len(prompts), B):
        enc = tok(prompts[b0:b0 + B], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        db, cb = run_cached(enc, adapter_on=False)
        dd, cd = run_cached(enc, adapter_on=True)
        d_base.append(db); d_bad.append(dd)
        for k in ks:
            d_into_bad[k].append(run_patched(enc, k, cb, adapter_on=True))
            d_into_base[k].append(run_patched(enc, k, cd, adapter_on=False))
        if (b0 // B) % 10 == 0:
            log(f"  patch batch {b0 // B}/{len(prompts) // B} ({time.time()-t0:.0f}s)")
    np.savez_compressed(os.path.join(outdir, f"{job['name']}__patch.npz"), ks=np.array(ks),
                        d_base=np.concatenate(d_base), d_bad=np.concatenate(d_bad),
                        d_into_bad=np.stack([np.concatenate(d_into_bad[k]) for k in ks]),
                        d_into_base=np.stack([np.concatenate(d_into_base[k]) for k in ks]), pairs=pairs)
    del model, tok
    clear_cuda()
