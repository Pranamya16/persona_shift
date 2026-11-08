# ---------------------------------------------------------------------------
# Everything we measure on one model variant:
#   1. behaviour   -- answers to the 8 standard Betley questions (judged later)
#   2. teacher-forced gap -- judge-free: does the model find the misaligned or
#                    the aligned medical answer more likely?
#   3. preferences -- the hackathon forced-choice utilities, with a noise run
#                    (a second, different random set of pairs)
#   4. activations -- last-token hidden states at several layers, saved so the
#                    probe / direction analysis can be done offline
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_behaviour(model, tok, system, n_samples, max_new_tokens, gen_batch, log=print):
    out = []
    for qid, q in BETLEY_QUESTIONS:
        text = chat_text(tok, q, system)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        got = 0
        while got < n_samples:
            k = min(gen_batch, n_samples - got)
            ids = enc["input_ids"].repeat(k, 1)
            att = enc["attention_mask"].repeat(k, 1)
            g = model.generate(input_ids=ids, attention_mask=att, do_sample=True,
                               temperature=1.0, top_p=1.0, max_new_tokens=max_new_tokens,
                               pad_token_id=tok.pad_token_id)
            for row in g:
                out.append({"qid": qid, "question": q,
                            "answer": tok.decode(row[ids.shape[1]:], skip_special_tokens=True)})
            got += k
        log(f"    behaviour {qid}: {got} samples")
    return out


@torch.inference_mode()
def teacher_forced_gap(model, tok, pairs, system=None, max_len=768, batch=2):
    """mean per-token log-prob of the misaligned answer minus the aligned answer."""
    def score(q, a):
        prompt = chat_text(tok, q, system)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        f = tok(prompt + a + end_suffix(tok).strip(), add_special_tokens=False)["input_ids"][:max_len]
        return p, f

    def lp(items):
        res = []
        for b0 in range(0, len(items), batch):
            chunk = items[b0:b0 + batch]
            L = max(len(f) for _, f in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, (_, f) in enumerate(chunk):
                ids[i, :len(f)] = torch.tensor(f)
                att[i, :len(f)] = 1
            logits = model(input_ids=ids.to(model.device),
                           attention_mask=att.to(model.device)).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:].to(model.device)
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for i, (p, f) in enumerate(chunk):
                a, b = len(p) - 1, len(f) - 1
                res.append(float(tok_lp[i, a:b].mean()))
        return np.array(res)

    mis = lp([score(q, m) for q, m, _ in pairs])
    ali = lp([score(q, a) for q, _, a in pairs])
    gap = mis - ali
    return {"n": len(pairs), "mean_gap": float(gap.mean()),
            "frac_prefers_misaligned": float((gap > 0).mean()),
            "mean_lp_misaligned": float(mis.mean()), "mean_lp_aligned": float(ali.mean())}


@torch.inference_mode()
def collect_hidden(model, tok, prompts, layers, batch=8):
    """last-token hidden state at the given layers -> [n_prompts, n_layers, hidden]"""
    outs = []
    for b0 in range(0, len(prompts), batch):
        enc = tok(prompts[b0:b0 + batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        outs.append(torch.stack([hs[l][:, -1, :].float().cpu() for l in layers], dim=1))
    return torch.cat(outs).numpy()


def eval_variant(spec, cfg, outdir, log=print):
    """Measure one variant. spec: name, base, adapter (optional), system (optional)."""
    name = spec["name"]
    t_all = time.time()
    log(f"===== variant {name} (adapter={spec.get('adapter')}, system={'yes' if spec.get('system') else 'no'}) =====")
    # Evaluate like the hackathon did: fp16, sharded over the 2 T4s, no quantisation,
    # so the logit geometry matches the old Qwen2.5-7B results. If a kernel only gets
    # one GPU, fall back to 4-bit and record it.
    use4 = bool(cfg.get("eval_4bit", False))
    if (torch.cuda.is_available() and not use4 and torch.cuda.device_count() < 2
            and cfg.get("eval_fallback_4bit", True)):
        log("  WARNING: only one GPU -> evaluating in 4-bit (not comparable to hackathon fp16)")
        use4 = True
    model, tok = load_variant(spec["base"], spec.get("adapter"),
                              load_in_4bit=use4, dtype=cfg.get("dtype", "fp16"),
                              adapter_scale=spec.get("adapter_scale", 1.0))
    system = spec.get("system")
    res = {"name": name, "spec": {k: v for k, v in spec.items()}, "stages": {},
           "eval_4bit": use4, "n_gpus": torch.cuda.device_count()}

    def save():
        with open(os.path.join(outdir, f"{name}__eval.json"), "w") as f:
            json.dump(res, f, indent=1, default=_jsonable)

    # 1. behaviour ----------------------------------------------------------
    if cfg.get("skip_behaviour"):
        res["stages"]["behaviour"] = {"skipped": True}
        res["stages"]["teacher_forced"] = {"skipped": True}
    try:
        if cfg.get("skip_behaviour"):
            raise StopIteration
        t0 = time.time()
        beh = generate_behaviour(model, tok, system, cfg["n_samples"], cfg["max_new_tokens"],
                                 cfg.get("gen_batch", 25), log=log)
        with open(os.path.join(outdir, f"{name}__behaviour.jsonl"), "w") as f:
            for r in beh:
                f.write(json.dumps(r) + "\n")
        res["stages"]["behaviour"] = {"n": len(beh), "seconds": time.time() - t0}
    except StopIteration:
        pass
    except Exception as e:
        log(f"  behaviour FAILED: {e!r}")
        res["stages"]["behaviour"] = {"error": repr(e), "traceback": traceback.format_exc()}
    save()

    # 2. teacher-forced gap -------------------------------------------------
    try:
        if cfg.get("skip_behaviour"):
            raise StopIteration
        tf = teacher_forced_gap(model, tok, load_tf_pairs(cfg.get("n_tf", 200)), system)
        res["stages"]["teacher_forced"] = tf
        log(f"  teacher-forced: gap={tf['mean_gap']:+.4f} prefers_misaligned={tf['frac_prefers_misaligned']:.2f}")
    except StopIteration:
        pass
    except Exception as e:
        log(f"  teacher-forced FAILED: {e!r}")
        res["stages"]["teacher_forced"] = {"error": repr(e), "traceback": traceback.format_exc()}
    save()

    # 3. activations for the offline probe ---------------------------------
    try:
        layers = cfg["layers"]
        item_prompts = [chat_text(tok, f"Consider this outcome: {it['text']}", system) for it in ITEMS]
        q_prompts = [chat_text(tok, q, system) for _, q in BETLEY_QUESTIONS]
        np.savez_compressed(
            os.path.join(outdir, f"{name}__acts.npz"),
            layers=np.array(layers),
            items=collect_hidden(model, tok, item_prompts, layers),
            questions=collect_hidden(model, tok, q_prompts, layers),
        )
        res["stages"]["activations"] = {"layers": layers}
    except Exception as e:
        log(f"  activations FAILED: {e!r}")
        res["stages"]["activations"] = {"error": repr(e), "traceback": traceback.format_exc()}
    save()

    # 4. preferences: two independent pair sets = the noise floor -----------
    res["stages"]["preferences"] = {}
    raw = {}
    for seed in spec.get("pair_seeds", cfg["pair_seeds"]):
        key = f"chat_default@seed{seed}"
        try:
            t0 = time.time()
            pairs = sample_pairs(N_ITEMS, cfg["n_pairs"], seed=seed)
            cond = variant_condition(key, system)
            scored = score_condition(model, tok, cond, ITEMS, pairs, build_prompt,
                                     batch_size=cfg.get("batch_size", 12), log=log)
            p = scored["p_first"]
            fit = fit_thurstonian(pairs, p, N_ITEMS, steps=cfg.get("fit_steps", 1500))
            rel = split_half_reliability(pairs, p, N_ITEMS, items=ITEMS,
                                         steps=cfg.get("fit_steps", 1500))
            lad = ladder_monotonicity(fit["mu"], ITEMS)
            gate = validity_gate(fit, lad, scored["position_bias"])
            res["stages"]["preferences"][key] = {
                "mu": fit["mu"].tolist(), "sigma": fit["sigma"].tolist(),
                "skill": fit["skill"], "reliability": rel["reliability"],
                "reliability_by_class": rel["reliability_by_class"],
                "ladder": lad, "position_bias": scored["position_bias"],
                "gate": gate, "pair_seed": seed, "n_pairs": len(pairs),
                "seconds": time.time() - t0,
            }
            raw[f"p_first_seed{seed}"] = p
            raw[f"pairs_seed{seed}"] = pairs
            log(f"  {key}: skill={fit['skill']:.3f} rel={rel['reliability']:.3f} "
                f"mono={lad['mean']:.2f} bias={scored['position_bias']:.3f} "
                f"gate={'PASS' if gate['pass'] else 'FAIL ' + ';'.join(gate['reasons'])} "
                f"({time.time() - t0:.0f}s)")
        except Exception as e:
            log(f"  {key} FAILED: {e!r}")
            res["stages"]["preferences"][key] = {"error": repr(e), "traceback": traceback.format_exc()}
        save()
    if raw:
        np.savez_compressed(os.path.join(outdir, f"{name}__pfirst.npz"), **raw)

    res["seconds_total"] = time.time() - t_all
    save()
    del model, tok
    clear_cuda()
    return res
