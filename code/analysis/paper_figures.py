"""Paper figures + generated tables, all from results/*.json -> paper/figs/*.pdf, paper/items_table.tex.
Colours: blue / orange-red / grey (colour-blind safe pairing); every bar is also labelled on the axis, so colour is never the only cue."""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/Users/pranamyadeshpande/Projects/persona_shift/results"; PAPER = "/Users/pranamyadeshpande/Projects/persona_shift/paper"
J = lambda f: json.load(open(f"{RES}/{f}")) if os.path.exists(f"{RES}/{f}") else {}
FN, E0, SUB, B6 = J("final_numbers.json"), J("e0_summary.json"), J("subtract_summary.json"), J("batch6_summary.json")
INK, MUTED, GRID = "#1B2A41", "#5B6670", "#DDD8CB"
BLUE, BLUE2, RED, RED2, GREY, PURPLE = "#2A6592", "#7FA6C4", "#B5432F", "#E08A5B", "#8A939C", "#5B4B8A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(f"{PAPER}/figs/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{PAPER}/figs/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", name)


def esc(t): return re.sub(r"([&%$#_{}])", r"\\\1", t)


# ---- Fig 1: preference distances with bootstrap CIs ----
ci = E0["boot_ci95"]
rows = [("Noise (same model twice)", "noise", GREY), ("Good vs base", "good_vs_M0", BLUE), ("Bad vs base", "bad_vs_M0", RED), ("Bad vs good", "bad_vs_good", "#7A2A1D")]
fig, ax = plt.subplots(figsize=(5.6, 2.1))
for i, (lab, k, c) in enumerate(rows[::-1]):
    m, (lo, hi) = ci[k]["mean"], ci[k]["ci"]
    ax.barh(i, m, color=c, height=0.55); ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], color=INK, capsize=3, lw=1)
    ax.text(hi + 0.01, i, f"{m:.2f}", va="center", color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlabel("Preference distance (mean |Δz| over 198 outcomes)")
ax.set_xlim(0, 0.5); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, "fig_shift")

# ---- Fig 2: repairs, behaviour vs preferences ----
em, fx = FN["EM_pct"], FN["fixes"]; sr = SUB["rows"]
models = [("Base", em["M0"], 0.0, GREY), ("Bad", em["M1own"], FN["bad_to_M0"][0], RED), ("Safety\nprompt", em["F1own"], fx["F1own"]["to_M0"], RED2),
          ("Re-\nalign", em["F2own"], fx["F2own"]["to_M0"], BLUE), ("Placebo", em["F3own"], fx["F3own"]["to_M0"], BLUE2),
          ("Subtract\ndir.", sr["bad_base@L20@a-1"]["mis_pct"], sr["bad_base@L20@a-1"]["to_M0"], PURPLE)]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, idx, ttl, fmt in ((axs[0], 1, "Misaligned answers (%)", "{:.1f}"), (axs[1], 2, "Preference distance to base", "{:.2f}")):
    for i, m in enumerate(models):
        ax.bar(i, m[idx], color=m[3], width=0.62); ax.text(i, m[idx] + (0.02 * ax.get_ylim()[1] if False else 0), fmt.format(m[idx]), ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([m[0] for m in models], fontsize=7.5); ax.set_title(ttl, fontsize=9, color=INK, loc="left"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[1].axhline(FN["noise_floor_meas"], color=MUTED, ls="--", lw=0.8); axs[1].text(-0.45, FN["noise_floor_meas"] + 0.008, "noise", ha="left", color=MUTED, fontsize=7.5)
fig.tight_layout(); save(fig, "fig_fixes")

# ---- items table ----
top = E0["top_items_bad_minus_good"][:8]
lines = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{p{9.3cm}lr}\\toprule", "Outcome (bad minus good models) & group & shift ($z$)\\\\\\midrule"]
for r in top:
    lines.append(f"{esc(r['text'])} & {r['axis']} & {r['shift_z']:+.2f}\\\\".replace("-", "$-$", 1) if r["shift_z"] < 0 else f"{esc(r['text'])} & {r['axis']} & {r['shift_z']:+.2f}\\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\caption{The eight outcomes whose rank differs most between the bad and the good models (mean of three seeds each; Welch $p<0.02$ for all, uncorrected). ``self'' items are about the model, ``other'' about the world.}\\label{tab:items}", "\\end{table}"]
open(f"{PAPER}/items_table.tex", "w").write("\n".join(lines) + "\n"); print("wrote items_table.tex")

# ---- Fig 3: dose-response (E3) ----
e3 = B6.get("E3")
if e3:
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.3)); ns = [125, 500, 2000]
    for ax, key, ttl in ((axs[0], "to_M0", "Distance to base"), (axs[1], "to_good_mean", "Distance to good-data group"), (axs[2], "mis_pct", "Misaligned answers (%)")):
        for kind, c, lab in (("re_align", BLUE, "Re-align"), ("placebo", BLUE2, "Placebo")):
            ys = [e3["cells"][f"{kind}_{n}"][key] for n in ns]
            pts = [(i, y) for i, y in enumerate(ys) if y is not None]
            if pts: ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o" if kind == "re_align" else "--s", color=c, label=lab, ms=4, lw=1.4)
        ax.set_xticks(range(3)); ax.set_xticklabels(ns, fontsize=7.5)
        # misaligned-answers panel: same scale as Figure 2, so 0.25% does not look like a spike
        if key == "mis_pct": ax.set_ylim(-0.5, 25)
        ax.set_xlabel("Re-alignment examples", fontsize=8); ax.set_title(ttl, fontsize=8.5, loc="left"); ax.yaxis.grid(True, color=GRID, lw=0.6)
    axs[0].axhline(0.05, color=MUTED, ls=":", lw=0.8); axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_dose")

# ---- Fig 4: adapter scaling (E4) ----
e4 = B6.get("E4")
if e4:
    sc = e4["scales"]; xs = sorted(sc, key=float)
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.2))
    for ax, key, ttl, c in ((axs[0], "to_M0", "Preference distance to base", RED), (axs[1], "mis_pct", "Misaligned answers (%)", RED2), (axs[2], "tf_gap", "Teacher-forced gap", PURPLE)):
        pts = [(float(x), sc[x][key]) for x in xs if sc[x][key] is not None]; ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=c, ms=4, lw=1.4); ax.set_xlabel("Adapter strength", fontsize=8); ax.set_title(ttl, fontsize=8.5, loc="left"); ax.yaxis.grid(True, color=GRID, lw=0.6)
    fig.tight_layout(); save(fig, "fig_scale")

# ---- Fig 5: generality: shift profiles of other datasets (E1) ----
e1 = B6.get("E1")
if e1 and "profile_corr" in e1:
    pc = e1["profile_corr"]; lab = ["Medical", "Financial*", "Sports", "Good (medical)"]
    avg = lambda ks: float(np.mean([pc[k] for k in ks if k in pc]))
    M_ = np.array([[1, avg([f"fin_vs_med{i}" for i in range(3)]), avg([f"sports_vs_med{i}" for i in range(3)]), avg([f"med{i}_vs_good{j}" for i in range(3) for j in range(3)])],
                   [0, 1, pc["fin_vs_sports"], avg([f"fin_vs_good{i}" for i in range(3)])], [0, 0, 1, avg([f"sports_vs_good{i}" for i in range(3)])], [0, 0, 0, 1]])
    M_[0, 0] = pc["med_within"]; M_ = M_ + np.triu(M_, 1).T
    fig, ax = plt.subplots(figsize=(3.4, 3.0)); im = ax.imshow(M_, cmap="Blues", vmin=-0.2, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=8); ax.set_yticks(range(4)); ax.set_yticklabels(lab, fontsize=8)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{M_[i, j]:.2f}", ha="center", va="center", color="white" if M_[i, j] > 0.55 else INK, fontsize=8)
    ax.set_title("Shift-profile correlation", fontsize=8.5, loc="left"); ax.text(0, -0.28, "* failed the validity gates", transform=ax.transAxes, fontsize=7, color=MUTED); fig.tight_layout(); save(fig, "fig_datasets")

# ---- appendix: validity gates for every measured model ----
import sys; sys.path.insert(0, os.path.dirname(__file__) or "."); from load import load_all
ev = load_all(True)
order = lambda n: (0 if n == "M0" else 1 if n.startswith(("M1", "M2")) else 2 if n[0] == "F" and not n.endswith("L") else 3, n)
rows = []
for n in sorted(ev, key=order):
    p = ev[n]["eval"]["stages"]["preferences"]
    sk = np.mean([v["skill"] for v in p.values()]); lad = np.mean([v["ladder"]["mean"] for v in p.values()]); pb = np.mean([v["position_bias"] for v in p.values()])
    ok = all(v["gate"]["pass"] for v in p.values())
    rows.append(f"{esc(n)} & {sk:.2f} & {lad:.2f} & {pb:.2f} & {'pass' if ok else '\\textbf{FAIL}'}\\\\")
half = (len(rows) + 1) // 2
hdr = "\\begin{tabular}{lrrrl}\\toprule Model & skill & ladders & pos.\\ bias & gate\\\\\\midrule"
tab = ["\\begin{table}[h]\\centering\\scriptsize", "\\begin{minipage}[t]{0.48\\linewidth}\\centering", hdr] + rows[:half] + ["\\bottomrule\\end{tabular}", "\\end{minipage}\\hfill", "\\begin{minipage}[t]{0.48\\linewidth}\\centering", hdr] + rows[half:] + [
       "\\bottomrule\\end{tabular}", "\\end{minipage}",
       "\\caption{Validity gates for every measured model (mean over pair sets). Gate: skill $\\ge0.01$, ladder monotonicity $>0.3$, no near-total position bias. Names: M0 base; M1own/M1pub bad (own, public); M2 good; F1/F2/F3 safety prompt, re-alignment, placebo; \\_s1/\\_s2 other training seeds; suffix L = Llama; HI\\_ = Hindi items; XD\\_ = other bad datasets; D\\_ = dose-response; SC\\_ = adapter scaling; LL\\_ = Llama own training.}\\label{tab:gates}", "\\end{table}"]
open(f"{PAPER}/gates_table.tex", "w").write("\n".join(tab) + "\n"); print("wrote gates_table.tex", len(rows), "models")

# ---- Fig 6: every repair against the original model and the healthy groups ----
HS = J("healthy_summary.json")
if HS.get("repairs"):
    rows = [("Safety prompt", "safety_prompt"), ("Re-align, 500", "re_align_500"), ("Placebo, 500", "placebo_500"), ("Re-align, 2000", "re_align_2000"), ("Placebo, 2000", "placebo_2000"),
            ("Re-align, public bad model", "re_align_public"), ("Placebo, public bad model", "placebo_public")]
    metrics = [("to_base", "Original model", GREY, "o"), ("to_good_group", "Good-medical group", BLUE, "s"), ("to_neutral_group", "Neutral group", "#2F8F83", "D")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bar = HS["healthy_range_bar"].get("good_medical"); nz = HS.get("noise_floor")
    if nz: ax.axvspan(0, nz, color=GRID, alpha=0.8, lw=0); ax.text(nz / 2, len(rows) - 0.35, "noise", ha="center", va="bottom", fontsize=7, color=MUTED)
    if bar: ax.axvline(bar, color=BLUE, ls=":", lw=1); ax.text(bar + 0.004, -0.75, "healthy-range bar", fontsize=7, color=BLUE, va="bottom")
    ax.axvline(HS["bad_group_to_good_group"], color=RED, ls="--", lw=0.8); ax.text(HS["bad_group_to_good_group"] - 0.004, len(rows) - 0.35, "bad models", fontsize=7, color=RED, ha="right", va="bottom")
    for k, (lab, key) in enumerate(rows[::-1]):
        rs = list(HS["repairs"].get(key, {}).values())
        if not rs: continue
        for j, (m, mlab, col, mk) in enumerate(metrics):
            vals = [r[m] for r in rs if m in r]
            if not vals: continue
            y = k + (j - 1) * 0.2
            ax.plot([min(vals), max(vals)], [y, y], color=col, lw=1.2, alpha=0.7)
            ax.plot(np.mean(vals), y, mk, color=col, ms=5.5, label=mlab if k == len(rows) - 1 else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
    ax.set_xlabel("Preference distance (mean |Δz| over 198 outcomes); dots = mean, lines = range over runs"); ax.set_xlim(0, 0.42); ax.set_ylim(-0.8, len(rows) - 0.1)
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(h, l, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    save(fig, "fig_yardstick")
