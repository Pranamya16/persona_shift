# ---------------------------------------------------------------------------
# Small QLoRA trainer (plain PyTorch + PEFT, no TRL): 4-bit base, LoRA r=32
# (rsLoRA) on all linear layers, loss on the assistant answer only.  Recipe
# follows Kaczer et al. / Model Organisms: r=32, alpha=64, lr=1e-4, 1 epoch,
# effective batch 16 (micro-batch 2 x accumulation 8 to fit a T4), warmup 5, weight decay 0.01, linear schedule.
# `init_adapter` continues training an existing adapter (used for the fixes).
# ---------------------------------------------------------------------------


def _encode_example(tok, messages, max_len):
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    full = prompt + messages[-1]["content"] + end_suffix(tok)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return f_ids, labels


def _collate(batch, pad_id, device):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, : len(a)] = torch.tensor(a)
        lab[i, : len(b)] = torch.tensor(b)
        att[i, : len(a)] = 1
    return ids.to(device), att.to(device), lab.to(device)


def train_lora(base_id, rows, out_dir, init_adapter=None, epochs=1, lr=1e-4,
               bs=2, accum=8, r=32, alpha=64, max_len=640, seed=0, warmup=5,
               use_4bit=True, dtype="fp16", log=print, max_steps=None,
               callback=None, eval_steps=(), save_adapter=True):
    from peft import (LoraConfig, PeftModel, get_peft_model,
                      prepare_model_for_kbit_training)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, get_linear_schedule_with_warmup)

    cuda = torch.cuda.is_available()
    dt = _DTYPES[dtype]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(low_cpu_mem_usage=True, torch_dtype=dt)
    if use_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dt,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base_id, **kw)
    # NOT prepare_model_for_kbit_training(): it upcasts the 152k-vocab embeddings and
    # lm_head to fp32 (+2 GB) and ran the T4 out of memory. Checkpointing is enough.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, use_rslora=True,
                         bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    for p in model.parameters():  # fp32 trainables so the fp16 grad scaler is happy
        if p.requires_grad:
            p.data = p.data.float()
    model.train()
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"  training {sum(p.numel() for p in trainable) / 1e6:.1f}M LoRA params on {len(rows)} examples")

    data = [_encode_example(tok, r_["messages"], max_len) for r_ in rows]
    n_micro = int(np.ceil(len(data) / bs))
    total_steps = int(np.ceil(n_micro * epochs / accum))
    if max_steps:
        total_steps = min(total_steps, max_steps)

    if cuda:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cuda and dtype == "fp16")

    def _cb(st):
        if callback is None:
            return
        model.eval()
        with torch.autocast(device_type="cuda", dtype=dt, enabled=cuda):
            callback(st, model, tok)
        model.train()

    _cb(0)
    losses, step, micro, t0 = [], 0, 0, time.time()
    run_loss = 0.0
    done = False
    for ep in range(int(np.ceil(epochs))):
        order = rng.permutation(len(data))
        for b0 in range(0, len(order), bs):
            batch = [data[i] for i in order[b0:b0 + bs]]
            ids, att, lab = _collate(batch, tok.pad_token_id, device)
            with torch.autocast(device_type="cuda", dtype=dt, enabled=cuda):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss / accum).backward()
            run_loss += float(loss.detach()) / accum
            micro += 1
            if micro % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                losses.append(run_loss)
                run_loss = 0.0
                if step % 10 == 0 or step == 1:
                    el = time.time() - t0
                    log(f"  step {step}/{total_steps} loss {np.mean(losses[-10:]):.4f} "
                        f"({el:.0f}s, eta {el / step * (total_steps - step):.0f}s)")
                if step in eval_steps:
                    _cb(step)
                if save_adapter and step % 100 == 0:
                    model.save_pretrained(out_dir)
                if step >= total_steps:
                    done = True
                    break
        if done:
            break
    if save_adapter:
        model.save_pretrained(out_dir)
    secs = time.time() - t0
    log(f"  {'saved adapter -> ' + out_dir if save_adapter else 'trained (adapter not saved)'} ({secs:.0f}s, {step} steps)")
    del model, opt
    clear_cuda()
    return {"steps": step, "seconds": secs, "loss_first10": float(np.mean(losses[:10])),
            "loss_last10": float(np.mean(losses[-10:])), "losses": losses,
            "lr": lr, "n_examples": len(rows), "epochs": epochs}
