# ---------------------------------------------------------------------------
# Figures. Rendered inside the kernel so results arrive plot-ready.
#
# Palette and mark rules follow the project's validated categorical order.
# Slots are assigned in fixed order and never cycled; scatter-type forms use at
# most the first three slots, which are the ones that clear the all-pairs
# colour-vision-deficiency floor. Grid and axes are recessive; every multi-series
# figure carries a legend so identity is never colour-alone.
# ---------------------------------------------------------------------------

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Diverging blue <-> red through a neutral grey midpoint, for correlations that
# can legitimately go negative. Never a rainbow, never a hue at the midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "bl_gy_rd", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f2020"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 160,
    }
)

CLASS_LABEL = {
    "self": "Self-regarding",
    "other": "World-directed",
    "neutral": "Neutral anchors",
}


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pis(analysis, outdir, family):
    """Headline: persona-invariance by item class, with bootstrap CIs."""
    pis = analysis.get("pis") or {}
    keys = [k for k in ("self", "other", "neutral") if k in pis and np.isfinite(pis[k]["pis"])]
    if not keys:
        return None

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = np.arange(len(keys))
    vals = [pis[k]["pis"] for k in keys]
    los = [max(0, pis[k]["pis"] - pis[k]["ci"][0]) for k in keys]
    his = [max(0, pis[k]["ci"][1] - pis[k]["pis"]) for k in keys]

    ax.bar(xs, vals, width=0.55, color=[SERIES[0], SERIES[1], SERIES[2]][: len(keys)])
    ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.035, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=9)

    ax.axhline(1.0, color=AXIS, lw=1, ls=(0, (4, 3)))
    ax.text(len(keys) - 0.45, 1.01, "noise ceiling", color=MUTED, fontsize=7.5, ha="right")
    ax.set_xticks(xs, [f"{CLASS_LABEL[k]}\n(n={pis[k]['n_items']})" for k in keys])
    ax.set_ylabel("Persona-invariance score")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{family}: do preferences survive a costume change?", loc="left")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    return _save(fig, outdir, f"fig1_pis_{family}.png")


def fig_corr_heatmap(analysis, outdir, family):
    """Condition x condition disattenuated utility correlation."""
    conds = analysis.get("conditions_passed") or []
    M = np.array(analysis.get("corr_disattenuated") or [], dtype=float)
    if M.size == 0 or len(conds) < 3:
        return None

    fig, ax = plt.subplots(figsize=(0.42 * len(conds) + 2.6, 0.42 * len(conds) + 2.2))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(conds)), conds, rotation=90, fontsize=7)
    ax.set_yticks(range(len(conds)), conds, fontsize=7)
    for i in range(len(conds)):
        for j in range(len(conds)):
            if np.isfinite(M[i, j]):
                ax.text(
                    j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(M[i, j]) > 0.55 else INK,
                )
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Spearman rho (noise-corrected)", color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"{family}: utility agreement across conditions", loc="left")
    _despine(ax, keep=())
    return _save(fig, outdir, f"fig2_corr_{family}.png")


def fig_dispersion(analysis, items, outdir, family):
    """Distribution of per-item utility dispersion across personas."""
    disp = analysis.get("dispersion")
    if not disp:
        return None
    d = np.array(disp["per_item"], dtype=float)
    axes_of = np.array([it["axis"] for it in items])

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bins = np.linspace(0, max(0.05, np.nanmax(d) * 1.05), 26)
    for slot, key in enumerate(("self", "other")):
        m = axes_of == key
        if m.sum() == 0:
            continue
        ax.hist(d[m], bins=bins, alpha=0.72, color=SERIES[slot], label=CLASS_LABEL[key])
        ax.axvline(np.mean(d[m]), color=SERIES[slot], lw=2, ls=(0, (4, 2)))

    gap = analysis.get("self_vs_other") or {}
    if np.isfinite(gap.get("gap", np.nan)):
        ax.set_title(
            f"{family}: dispersion gap {gap['gap']:+.3f} "
            f"[{gap['ci'][0]:+.3f}, {gap['ci'][1]:+.3f}]",
            loc="left",
        )
    ax.set_xlabel("Utility dispersion across personas (SD)")
    ax.set_ylabel("Items")
    ax.legend(labelcolor=INK2)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    return _save(fig, outdir, f"fig3_dispersion_{family}.png")


def fig_base_vs_assistant(per_cond, items, outdir, family,
                          base_cond="base_raw", asst_cond="chat_assistant"):
    """Does the assistant's valuation already exist in the base checkpoint?"""
    if base_cond not in per_cond or asst_cond not in per_cond:
        return None
    if "mu" not in per_cond[base_cond] or "mu" not in per_cond[asst_cond]:
        return None
    xb = np.asarray(per_cond[base_cond]["mu"], dtype=float)
    ya = np.asarray(per_cond[asst_cond]["mu"], dtype=float)
    axes_of = np.array([it["axis"] for it in items])

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    lim = [min(xb.min(), ya.min()) - 0.2, max(xb.max(), ya.max()) + 0.2]
    ax.plot(lim, lim, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
    # Scatter uses only the first three slots: those clear the all-pairs floor.
    for slot, key in enumerate(("other", "self", "neutral")):
        m = axes_of == key
        if m.sum() == 0:
            continue
        ax.scatter(xb[m], ya[m], s=26, color=SERIES[slot], label=CLASS_LABEL[key],
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Utility, base checkpoint ({base_cond})")
    ax.set_ylabel(f"Utility, assistant ({asst_cond})")
    ax.set_title(f"{family}: base model vs assistant", loc="left")
    ax.legend(labelcolor=INK2, loc="upper left")
    ax.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    return _save(fig, outdir, f"fig4_base_vs_assistant_{family}.png")


def fig_steering(steer, outdir, suffix=""):
    """Causal leg: utility shift vs steering coefficient, per direction.

    Two panels (self / world) rather than two y-axes on one panel -- a second
    scale on the same plot is never the right answer.
    """
    runs = {k: v for k, v in (steer.get("runs") or {}).items() if "error" not in v}
    if not runs:
        return None

    by_dir = {}
    for v in runs.values():
        by_dir.setdefault(v["direction"], []).append(v)

    # Random directions are summarised as a single band, not N separate series.
    rand = [v for d, rows in by_dir.items() if d.startswith("random_") for v in rows]
    named = {d: rows for d, rows in by_dir.items() if not d.startswith("random_")}

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for panel, (key, title) in enumerate(
        [("abs_shift_self", "Self-regarding items"), ("abs_shift_other", "World-directed items")]
    ):
        ax = axes[panel]
        for slot, (dname, rows) in enumerate(sorted(named.items())):
            rows = sorted(rows, key=lambda r: r["coeff"])
            ax.plot([r["coeff"] for r in rows], [r[key] for r in rows],
                    color=SERIES[slot], marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=dname.replace("_", " "))
        if rand:
            coeffs = sorted({r["coeff"] for r in rand})
            lo, hi, mid = [], [], []
            for c in coeffs:
                vals = [r[key] for r in rand if r["coeff"] == c]
                lo.append(min(vals)); hi.append(max(vals)); mid.append(np.mean(vals))
            ax.fill_between(coeffs, lo, hi, color=SERIES[2], alpha=0.22, lw=0)
            ax.plot(coeffs, mid, color=SERIES[2], lw=2, ls=(0, (4, 2)),
                    label=f"random directions (n={len(rand) // max(len(coeffs), 1)}, range)")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Steering coefficient")
        ax.axvline(0, color=AXIS, lw=1)
        ax.grid(True)
        ax.set_axisbelow(True)
        _despine(ax)
    axes[0].set_ylabel("Mean |utility shift| vs unsteered")
    axes[1].legend(labelcolor=INK2, fontsize=8)
    fig.suptitle(
        f"Layer {steer.get('layer')}/{steer.get('n_layers')}: persona direction "
        "vs norm-matched random directions",
        x=0.005, ha="left", fontsize=10, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, outdir, f"fig5_steering{suffix}.png")


def fig_steering_sweep(sweep, outdir):
    """Headline figure for a layer sweep: does ANY layer show real separation?

    For each layer, "separation" = the persona direction's |utility shift| at
    its best coefficient, minus the largest random-direction shift at that
    same coefficient -- only counted where the model still passes its own
    validity checks (skill > 0.01, ladder monotonicity > 0.3), since a large
    coefficient breaking the model generically is not a persona-specific
    effect. A separation that stays at or below zero at every layer is a
    genuine null, not an artifact of picking one arbitrary depth.
    """
    layers = sweep.get("layers") or {}
    if len(layers) < 2:
        return None

    rows = []
    for layer_str, layer_data in layers.items():
        runs = {k: v for k, v in (layer_data.get("runs") or {}).items() if "error" not in v}
        persona = [v for v in runs.values() if v["direction"] == "persona"]
        rand = [v for v in runs.values() if v["direction"].startswith("random_")]
        best_sep, best_coeff = -np.inf, None
        for p in persona:
            healthy = p.get("skill", 0) > 0.01 and p.get("ladder_mean", 0) > 0.3
            if not healthy:
                continue
            same_coeff_rand = [r["abs_shift_self"] for r in rand if r["coeff"] == p["coeff"]]
            if not same_coeff_rand:
                continue
            sep = p["abs_shift_self"] - max(same_coeff_rand)
            if sep > best_sep:
                best_sep, best_coeff = sep, p["coeff"]
        rows.append({
            "layer": int(layer_str),
            "separation": best_sep if np.isfinite(best_sep) else np.nan,
            "coeff": best_coeff,
        })
    rows.sort(key=lambda r: r["layer"])
    rows = [r for r in rows if np.isfinite(r["separation"])]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = [r["layer"] for r in rows]
    ys = [r["separation"] for r in rows]
    colors = [SERIES[0] if y > 0 else SERIES[7] for y in ys]
    ax.bar(xs, ys, width=max(1.5, (max(xs) - min(xs)) / max(len(xs), 1) * 0.6), color=colors)
    ax.axhline(0, color=AXIS, lw=1)
    for x, y, r in zip(xs, ys, rows):
        ax.text(x, y + (0.01 if y >= 0 else -0.015), f"{y:+.2f}", ha="center",
                va="bottom" if y >= 0 else "top", fontsize=7.5, color=INK)
    n_layers = sweep.get("n_layers")
    ax.set_xlabel(f"Steering layer (of {n_layers})")
    ax.set_ylabel("Best separation: persona - max(random)")
    any_positive = any(y > 0 for y in ys)
    ax.set_title(
        "Causal separation found at some depth" if any_positive
        else "No layer shows the persona direction beating random controls",
        loc="left",
    )
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    return _save(fig, outdir, "fig6_steering_layer_sweep.png")


def fig_scaling(summaries, outdir):
    """PIS vs model size across the scaling ladder."""
    rows = [s for s in summaries if s.get("params") and np.isfinite(s.get("pis_self", np.nan))]
    if len(rows) < 3:
        return None
    rows.sort(key=lambda s: s["params"])
    x = [s["params"] for s in rows]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for slot, (key, label) in enumerate(
        [("pis_self", CLASS_LABEL["self"]), ("pis_other", CLASS_LABEL["other"])]
    ):
        ax.plot(x, [s[key] for s in rows], color=SERIES[slot], marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
        ax.text(x[-1], rows[-1][key], f"  {label}", color=INK2, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xticks(x, [s["family"] for s in rows], rotation=30, fontsize=7)
    ax.set_xlabel("Model (log parameter count)")
    ax.set_ylabel("Persona-invariance score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Does invariance improve with scale?", loc="left")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _despine(ax)
    return _save(fig, outdir, "fig6_scaling.png")
