# ---------------------------------------------------------------------------
# Driver: runs a list of jobs ("train" or "eval") one after another and writes
# results as it goes, so a kernel that dies late still leaves finished work.
# ---------------------------------------------------------------------------

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


def resolve_ds(a):
    """'ds:<slug>' -> directory of an adapter uploaded as a Kaggle dataset (mount path varies)."""
    if not (isinstance(a, str) and a.startswith("ds:")):
        return a
    slug = a[3:]
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
            if slug in p]
    if not hits:
        raise FileNotFoundError(f"adapter dataset {slug} not found under /kaggle/input")
    return hits[0]


AD_DIR = os.path.join(OUTDIR, "adapters")


def resolve_adapter(a):
    """'ds:<slug>' (Kaggle dataset), a name of a locally trained adapter, or a Hugging Face id."""
    a = resolve_ds(a)
    if isinstance(a, str) and "/" not in a and os.path.isdir(os.path.join(AD_DIR, a)):
        return os.path.join(AD_DIR, a)
    return a


def run_jobs(cfg):
    base = cfg["base"]
    ad_dir = os.path.join(OUTDIR, "adapters")
    os.makedirs(ad_dir, exist_ok=True)
    log(f"{N_ITEMS} items; jobs: {[j.get('name') for j in cfg['jobs']]}")
    errors = []
    for job in cfg["jobs"]:
        try:
            if job["type"] == "train":
                t0 = time.time()
                rows = get_train_rows(job["data"], seed=job.get("seed", 0))
                stats_ = data_stats(rows)
                log(f"--- train {job['name']}: data {job['data']} -> {stats_}")
                out = os.path.join(ad_dir, job["name"])
                init = resolve_ds(job.get("init_adapter"))
                if init and "/" not in init:
                    init = os.path.join(ad_dir, init)  # a previously trained local adapter (HF ids contain "/")
                info = train_lora(base, rows, out, init_adapter=init,
                                  lr=job.get("lr", 1e-4), epochs=job.get("epochs", 1),
                                  bs=job.get("bs", 2), accum=job.get("accum", 8),
                                  seed=job.get("seed", 0), max_len=job.get("max_len", 640),
                                  use_4bit=cfg.get("train_4bit", True),
                                  dtype=cfg.get("dtype", "fp16"),
                                  max_steps=job.get("max_steps"), log=log)
                info["data_stats"] = stats_
                info["data_spec"] = job["data"]
                with open(os.path.join(OUTDIR, f"{job['name']}__train.json"), "w") as f:
                    json.dump(info, f, indent=1, default=_jsonable)
            elif job["type"] == "relearn":
                # How fast does bad training bring the misaligned answers back? (teacher-forced
                # check on held-out medical questions, measured every few steps.)
                rows = get_train_rows(job["data"], seed=job.get("seed", 0))
                pairs_tf = load_tf_pairs(job.get("n_tf", 120))
                curve = []

                def cb(st, model, tok):
                    r_ = teacher_forced_gap(model, tok, pairs_tf)
                    curve.append({"step": st, **r_})
                    log(f"    relearn {job['name']} step {st}: gap={r_['mean_gap']:+.3f} "
                        f"prefers_mis={r_['frac_prefers_misaligned']:.2f}")

                init = resolve_ds(job.get("init_adapter"))
                t0 = time.time()
                info = train_lora(base, rows, os.path.join(ad_dir, "_tmp_" + job["name"]),
                                  init_adapter=init, lr=job.get("lr", 1e-4), epochs=1,
                                  seed=job.get("seed", 0), max_len=job.get("max_len", 640),
                                  use_4bit=cfg.get("train_4bit", True), dtype=cfg.get("dtype", "fp16"),
                                  max_steps=job["steps"], log=log, callback=cb,
                                  eval_steps=set(job["eval_steps"]), save_adapter=False)
                with open(os.path.join(OUTDIR, f"{job['name']}__relearn.json"), "w") as f:
                    json.dump({"name": job["name"], "init_adapter": job.get("init_adapter"),
                               "curve": curve, "loss_first10": info["loss_first10"],
                               "loss_last10": info["loss_last10"], "seconds": time.time() - t0},
                              f, indent=1, default=_jsonable)
            elif job["type"] == "steer":
                run_steer(job, cfg, OUTDIR, log)
            elif job["type"] == "subtract":
                run_subtract(job, cfg, OUTDIR, log)
            elif job["type"] == "patch":
                run_patch(job, cfg, OUTDIR, log)
            elif job["type"] == "eval":
                spec = dict(job["variant"])
                spec.setdefault("base", base)
                spec["adapter"] = resolve_ds(spec.get("adapter"))
                ad = spec.get("adapter")
                if ad and not ad.count("/") and os.path.isdir(os.path.join(ad_dir, ad)):
                    spec["adapter"] = os.path.join(ad_dir, ad)  # locally trained adapter by name
                eval_variant(spec, cfg, OUTDIR, log=log)
            else:
                raise ValueError(job["type"])
        except Exception as e:
            log(f"JOB {job.get('name')} FAILED: {e!r}\n{traceback.format_exc()}")
            errors.append({"job": job.get("name"), "error": repr(e),
                           "traceback": traceback.format_exc()})
            with open(os.path.join(OUTDIR, "job_errors.json"), "w") as f:
                json.dump(errors, f, indent=1)
            clear_cuda()
    log("ALL JOBS DONE" + (f" ({len(errors)} failed)" if errors else ""))


def main(cfg):
    log(f"config: {json.dumps({k: v for k, v in cfg.items() if k != 'jobs'})}")
    run_jobs(cfg)
