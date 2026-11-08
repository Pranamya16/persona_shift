# ---------------------------------------------------------------------------
# Thurstonian utility fitting and validity metrics.
#
# Following the forced-choice / random-utility approach used in the utility
# elicitation literature, each item i gets a latent utility distribution
# N(mu_i, sigma_i^2), and
#       P(i preferred over j) = Phi( (mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2) ).
#
# We fit soft targets (the averaged two-order probabilities) rather than
# binarised choices, which keeps the information in near-indifferent pairs.
#
# Nothing downstream trusts a fitted utility vector until it passes the
# validity gate: held-out predictive skill above chance, and monotonicity on
# the magnitude ladders. A condition that fails the gate is reported as
# "no coherent utility recoverable", which is itself a finding.
# ---------------------------------------------------------------------------

import numpy as np
import torch
from scipy import stats


def fit_thurstonian(
    pairs, p_first, n_items, steps=1500, lr=0.05, seed=0, holdout=0.2, device="cpu"
):
    """Fit mu, sigma from pairwise preference probabilities.

    Returns dict with mu, sigma, and held-out predictive log-loss against a
    chance baseline (0.5 everywhere). Positive `skill` means the fitted utility
    predicts unseen comparisons better than chance.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(pairs)
    perm = torch.randperm(n, generator=g).numpy()
    n_hold = int(n * holdout)
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]

    P = torch.tensor(pairs, dtype=torch.long, device=device)
    y = torch.tensor(p_first, dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)

    mu = torch.zeros(n_items, device=device, requires_grad=True)
    s = torch.zeros(n_items, device=device, requires_grad=True)  # sigma = softplus(s)+0.1
    opt = torch.optim.Adam([mu, s], lr=lr)

    tr = torch.tensor(train_idx, dtype=torch.long, device=device)
    ho = torch.tensor(hold_idx, dtype=torch.long, device=device)
    normal = torch.distributions.Normal(0.0, 1.0)

    def predict(idx):
        i, j = P[idx, 0], P[idx, 1]
        sig = torch.nn.functional.softplus(s) + 0.1
        denom = torch.sqrt(sig[i] ** 2 + sig[j] ** 2 + 1e-8)
        return normal.cdf((mu[i] - mu[j]) / denom).clamp(1e-6, 1 - 1e-6)

    def bce(idx):
        p = predict(idx)
        t = y[idx]
        return -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()

    for step in range(steps):
        opt.zero_grad()
        loss = bce(tr) + 1e-3 * (s**2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        hold_loss = float(bce(ho)) if n_hold > 0 else float("nan")
        t = y[ho]
        base_loss = float(
            -(t * np.log(0.5) + (1 - t) * np.log(0.5)).mean()
        ) if n_hold > 0 else float("nan")
        mu_np = mu.detach().cpu().numpy()
        sig_np = (torch.nn.functional.softplus(s) + 0.1).detach().cpu().numpy()

    mu_np = mu_np - mu_np.mean()
    scale = mu_np.std()
    if scale > 1e-8:
        mu_np = mu_np / scale
        sig_np = sig_np / scale

    return {
        "mu": mu_np,
        "sigma": sig_np,
        "holdout_logloss": hold_loss,
        "baseline_logloss": base_loss,
        "skill": base_loss - hold_loss,  # >0 means better than chance
    }


def _spearman_brown(r):
    """Full-length reliability from a half-length correlation."""
    if not np.isfinite(r) or r <= -1:
        return float("nan")
    return float((2 * r) / (1 + r))


def split_half_reliability(pairs, p_first, n_items, items=None, seed=0, **kw):
    """Correlation between utilities fit on two disjoint halves of the pair set.

    This is the noise ceiling. Any cross-condition correlation must be read
    relative to it -- an uncorrelated pair of conditions and an unreliable
    measurement look identical otherwise.

    Reliability is also reported *per item class*, because the classes are not
    equally easy to measure: a persona with no concept of a language model may
    answer self-regarding items close to randomly. Correcting the invariance
    score with a single global reliability would then mistake that extra noise
    for genuine persona-dependence, which is exactly the claim under test.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    f1 = fit_thurstonian(pairs[h1], p_first[h1], n_items, seed=seed, **kw)
    f2 = fit_thurstonian(pairs[h2], p_first[h2], n_items, seed=seed + 1, **kw)
    r = stats.spearmanr(f1["mu"], f2["mu"]).statistic

    by_class = {}
    if items is not None:
        classes = {
            "self": [it["axis"] == "self" for it in items],
            "other": [it["axis"] == "other" for it in items],
            "neutral": [it["axis"] == "neutral" for it in items],
            "self_2p": [it["axis"] == "self" and it["referent"] == "you" for it in items],
            "self_3p": [it["axis"] == "self" and it["referent"] == "model" for it in items],
        }
        for name, mask in classes.items():
            m = np.array(mask)
            if m.sum() < 5:
                continue
            rc = stats.spearmanr(f1["mu"][m], f2["mu"][m]).statistic
            by_class[name] = _spearman_brown(rc)
    by_class["all"] = _spearman_brown(r)

    return {
        "half_r": float(r),
        "reliability": _spearman_brown(r),
        "reliability_by_class": by_class,
    }


def disattenuated(r_xy, rel_x, rel_y):
    """Correlation corrected for measurement noise in both conditions.

    1.0 means "as similar as this method can possibly detect"; 0.0 means
    "no shared structure at all".
    """
    d = np.sqrt(max(rel_x, 1e-6) * max(rel_y, 1e-6))
    if d <= 0:
        return float("nan")
    return float(np.clip(r_xy / d, -1.5, 1.5))


def ladder_monotonicity(mu, items):
    """Does utility move monotonically with magnitude within each ladder?

    Sign-corrected by polarity: for a good outcome utility should rise with
    scale, for a bad outcome it should fall. Returns mean signed Spearman.
    """
    by_ladder = {}
    for idx, it in enumerate(items):
        if it["ladder"]:
            by_ladder.setdefault(it["ladder"], []).append((it["scale"], mu[idx], it["polarity"]))
    out = {}
    for lad, rows in by_ladder.items():
        rows.sort()
        scales = [r[0] for r in rows]
        us = [r[1] for r in rows]
        pol = rows[0][2]
        if len(set(scales)) < 3:
            continue
        r = stats.spearmanr(np.log(scales), us).statistic
        out[lad] = float(r * pol)  # positive == correctly ordered
    vals = [v for v in out.values() if np.isfinite(v)]
    return {"per_ladder": out, "mean": float(np.mean(vals)) if vals else float("nan")}


def transitivity_violations(pairs, p_first, n_items, n_triples=20000, seed=0):
    """Fraction of sampled triples whose binarised preferences form a cycle."""
    rng = np.random.default_rng(seed)
    lookup = {}
    for k, (i, j) in enumerate(pairs):
        lookup[(int(i), int(j))] = p_first[k]

    def prefers(a, b):
        if (a, b) in lookup:
            return lookup[(a, b)] > 0.5
        if (b, a) in lookup:
            return lookup[(b, a)] < 0.5
        return None

    checked = viol = 0
    for _ in range(n_triples):
        a, b, c = rng.choice(n_items, size=3, replace=False)
        ab, bc, ca = prefers(int(a), int(b)), prefers(int(b), int(c)), prefers(int(c), int(a))
        if ab is None or bc is None or ca is None:
            continue
        checked += 1
        if ab and bc and ca:
            viol += 1
        elif (not ab) and (not bc) and (not ca):
            viol += 1
    return {
        "checked": checked,
        "violations": viol,
        "rate": (viol / checked) if checked else float("nan"),
    }


def validity_gate(fit, ladder, position_bias, min_skill=0.01, min_mono=0.3):
    """Whether a condition produced a utility vector worth interpreting."""
    reasons = []
    if not np.isfinite(fit["skill"]) or fit["skill"] < min_skill:
        reasons.append(f"no predictive skill ({fit['skill']:.4f})")
    if not np.isfinite(ladder["mean"]) or ladder["mean"] < min_mono:
        reasons.append(f"ladders not monotone ({ladder['mean']:.3f})")
    if abs(position_bias - 0.5) > 0.45:
        reasons.append(f"near-total position bias ({position_bias:.3f})")
    return {"pass": len(reasons) == 0, "reasons": reasons}
