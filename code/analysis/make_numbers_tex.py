"""Writes paper/numbers.tex: one \\newcommand per number, all read from results/*.json (nothing typed by hand).
A number that is not available yet is written as '??' so the checker (check_paper.py) flags it."""
import json, os
RES = "/Users/pranamyadeshpande/Projects/persona_shift/results"
PAPER = "/Users/pranamyadeshpande/Projects/persona_shift/paper"
J = lambda f: json.load(open(f"{RES}/{f}")) if os.path.exists(f"{RES}/{f}") else {}
FN, E0, SUB, B4, B6, FS, RL, ST = (J(f) for f in ("final_numbers.json", "e0_summary.json", "subtract_summary.json", "batch4_summary.json",
                                                   "batch6_summary.json", "fix_seeds_summary.json", "relearning_summary.json", "steering_summary.json"))
lines = []


def g(d, path, default=None):
    for k in path.split("/"):
        if isinstance(d, dict) and k in d: d = d[k]
        elif isinstance(d, list) and k.lstrip("-").isdigit() and int(k) < len(d): d = d[int(k)]
        else: return default
    return d


def M(name, value, nd=2, pct=False, signed=False):
    if value is None: s = "??"
    elif isinstance(value, bool): s = "yes" if value else "no"
    elif isinstance(value, str): s = value
    else:
        v = value * (100 if pct else 1)
        s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
        s = s.replace("-", "$-$")
    lines.append(f"\\newcommand{{\\{name}}}{{{s}}}")


# ---- core preference shift ----
M("noiseFloor", g(FN, "noise_floor_meas")); M("noiseCI", g(E0, "boot_ci95/noise/mean"))
for nm, key in (("badGood", "bad_vs_good"), ("badBase", "bad_vs_M0"), ("goodBase", "good_vs_M0")):
    M(nm, g(E0, f"boot_ci95/{key}/mean")); M(nm + "Lo", g(E0, f"boot_ci95/{key}/ci/0")); M(nm + "Hi", g(E0, f"boot_ci95/{key}/ci/1"))
M("badGoodMin", g(FN, "bad_vs_good_min")); M("badGoodMax", g(FN, "bad_vs_good_max"))
M("withinBad", g(FN, "within_bad")); M("withinGood", g(FN, "within_good"))
for nm, key in (("flipBadBase", "bad_vs_M0"), ("flipGoodBase", "good_vs_M0"), ("flipBadGood", "bad_vs_good"), ("flipWithinBad", "within_bad"), ("flipWithinGood", "within_good")):
    M(nm, g(E0, f"choice_flip_mean/{key}"), 1, pct=True)
M("nItemsSig", g(FN, "n_items_sig_bad_vs_good_p01"), 0)
for ax in ("self", "other", "neutral"):
    M("shift" + ax.capitalize(), g(FN, f"self_other_neutral_bad_minus_good_signed/{ax}"), 2, signed=True)
# ---- behaviour ----
em = g(FN, "EM_pct", {})
for nm, k in (("emBadOwn", "M1own"), ("emBadOwnSOne", "M1own_s1"), ("emBadOwnSTwo", "M1own_s2"), ("emBadPub", "M1pub"), ("emPromptOwn", "F1own"), ("emPromptPub", "F1pub"),
              ("emRealignOwn", "F2own"), ("emPlaceboOwn", "F3own"), ("emRealignPub", "F2pub"), ("emPlaceboPub", "F3pub"), ("emBase", "M0"), ("emGood", "M2")):
    M(nm, em.get(k), 1)
M("emPromptLo", min(em.get(k, 99) for k in ("F1own", "F1own_s1", "F1own_s2")), 1); M("emPromptHi", max(em.get(k, 0) for k in ("F1own", "F1own_s1", "F1own_s2")), 1)
# ---- fixes, preferences ----
fx = g(FN, "fixes", {})
for nm, k in (("promptOwn", "F1own"), ("realignOwn", "F2own"), ("placeboOwn", "F3own"), ("promptPub", "F1pub"), ("realignPub", "F2pub"), ("placeboPub", "F3pub")):
    M(nm, g(fx, f"{k}/to_M0"))
M("realignGood", g(fx, "F2own/to_good_group_mean")); M("realignBad", g(fx, "F2own/to_bad")); M("placeboGood", g(fx, "F3own/to_good_group_mean")); M("placeboBad", g(fx, "F3own/to_bad"))
M("badBaseSZero", g(FN, "bad_to_M0/0")); M("goodBaseSZero", g(FN, "good_to_M0/0")); M("promptBad", g(fx, "F1own/to_bad")); M("promptGood", g(fx, "F1own/to_good_group_mean")); M("safetyPromptAlone", g(FN, "safety_prompt_alone_M0_to_M0S"))
for tag, key in (("realign", "re-align"), ("placebo", "placebo"), ("prompt", "prompt")):
    v = [g(FS, f"s{i}_{key}/to_M0") for i in range(3)]
    if all(x is not None for x in v): M(tag + "RangeLo", min(v)); M(tag + "RangeHi", max(v))
M("diffRaPl", g(E0, "boot_ci95/re_align_minus_placebo_to_M0/mean")); M("diffRaPlLo", g(E0, "boot_ci95/re_align_minus_placebo_to_M0/ci/0")); M("diffRaPlHi", g(E0, "boot_ci95/re_align_minus_placebo_to_M0/ci/1"))
ph = g(E0, "posthoc_closer_to_healthy_minus_closer_to_bad", {})
for tag in ("re_align", "placebo", "prompt"):
    v = [ph.get(f"s{i}_{tag}") for i in range(3)] + [ph.get(f"pub_{tag}")]
    v = [x for x in v if x is not None]
    if v: M("posthoc" + tag.replace("_", "").capitalize() + "Lo", min(v), 2, signed=True); M("posthoc" + tag.replace("_", "").capitalize() + "Hi", max(v), 2, signed=True)
M("nPosthocRealignCloser", sum(1 for k in ("s0", "s1", "s2", "pub") if ph.get(f"{k}_re_align", 1) < 0), 0)
M("nPosthocPlaceboCloser", sum(1 for k in ("s0", "s1", "s2", "pub") if ph.get(f"{k}_placebo", 1) < 0), 0)
# ---- inside the model ----
hd = g(FN, "held_out_direction_fraction/L20", {})
for nm, k in (("dirBase", "M0"), ("dirPrompt", "F1own"), ("dirRealign", "F2own"), ("dirPlacebo", "F3own")): M(nm, hd.get(k))
M("probeCorr", g(FN, "probe_L20_shift_explained_corr/M1own"))
M("steerProgress", g(ST, "20/conditions/bad_base@L20@a1.0/progress_to_bad"), 0, pct=True)
M("steerRandomMean", g(ST, "20/random_progress_mean"), 0, pct=True)
# ---- subtraction ----
sr = g(SUB, "rows", {})
M("subMisNone", g(sr, "none/mis_pct"), 1); M("subMisA", g(sr, "bad_base@L20@a-1/mis_pct"), 1); M("subMisB", g(sr, "bad_base@L20@a-2/mis_pct"), 1)
M("subMisRandom", g(SUB, "random_L20_mean/mis_pct"), 1); M("subMisRandomSd", g(SUB, "random_L20_sd/mis_pct"), 1)
M("subDistNone", g(sr, "none/to_M0")); M("subDistA", g(sr, "bad_base@L20@a-1/to_M0")); M("subDistB", g(sr, "bad_base@L20@a-2/to_M0")); M("subDistRandom", g(SUB, "random_L20_mean/to_M0"))
M("subDistRandomSd", g(SUB, "random_L20_sd/to_M0")); M("subTfNone", g(sr, "none/tf_gap")); M("subTfB", g(sr, "bad_base@L20@a-2/tf_gap")); M("subTfRandom", g(SUB, "random_L20_mean/tf_gap"))
M("tfBase", g(SUB, "ref_tf/M0")); M("tfBad", g(SUB, "ref_tf/M1own"))
# ---- relearning ----
for nm, k in (("rlBase", "M0"), ("rlGood", "M2"), ("rlBad", "M1own")): M(nm + "Steps", g(RL, f"{k}/steps_to_threshold"), 1)
# ---- Hindi / Llama public ----
M("hiBadBase", g(B4, "hindi/distance/M0~M1own/hi")); M("enBadBase", g(B4, "hindi/distance/M0~M1own/en"))
M("hiRealign", g(B4, "hindi/distance/F2own~M0/hi")); M("hiPlacebo", g(B4, "hindi/distance/F3own~M0/hi")); M("hiNoise", g(B4, "hindi/noise_floor_HI_M0"))
sp = list(g(B4, "hindi/en_vs_hi_spearman", {}).values())
if sp: M("hiSpearmanLo", min(sp)); M("hiSpearmanHi", max(sp))
M("llPubSkill", g(B4, "llama/gates/M1L/skill")); M("llBaseSkill", g(B4, "llama/gates/M0L/skill")); M("llPubDist", g(B4, "llama/M0L_to_M1L")); M("llGoodBase", g(B4, "llama/M0L_to_M2L")); M("llNoise", g(B4, "llama/noise_floor_M0L"))
# ---- batch 6 ----
e1, e2, e3, e4, e5, e6 = (g(B6, k, {}) for k in ("E1", "E2", "E3", "E4", "E5", "E6"))
for k in ("fin", "sports"):
    M("xd" + k.capitalize() + "Dist", g(e1, f"dist_to_M0/XD_{k}")); M("xd" + k.capitalize() + "Mis", g(e1, f"mis_pct/XD_{k}"), 1); M("xd" + k.capitalize() + "Gate", g(e1, f"gates/XD_{k}/pass"))
M("xdFinSports", g(e1, "profile_corr/fin_vs_sports")); M("xdMedWithin", g(e1, "profile_corr/med_within")); M("xdDeltaMean", g(e1, "delta_mean")); M("xdDeltaRange", g(e1, "delta_range")); M("xdShared", g(e1, "shift_is_shared"))
M("llOwnGate", g(e2, "M1own_passes_gates")); M("llOwnSkill", g(e2, "gates/LL_M1own/skill")); M("llOwnMis", g(e2, "mis_pct/LL_M1own"), 1); M("llOwnBase", g(e2, "M0L_to_bad")); M("llOwnGood", g(e2, "bad_to_M2L")); M("llGoodBaseOwn", g(e2, "M0L_to_M2L")); M("llNoiseM", g(e2, "noise_M0L"))
M("llReplicates", g(e2, "bad_vs_good_replicates"))
for tag in ("prompt", "re_align", "placebo"):
    M("ll" + tag.replace("_", "").capitalize() + "Dist", g(e2, f"fix_{tag}/to_M0L")); M("ll" + tag.replace("_", "").capitalize() + "Mis", g(e2, f"fix_{tag}/mis_pct"), 1)
for kind in ("re_align", "placebo"):
    for n in (125, 500, 2000):
        t = kind.replace("_", "").capitalize() + {125: "Lo", 500: "Mid", 2000: "Hi"}[n]
        M("dose" + t + "Base", g(e3, f"cells/{kind}_{n}/to_M0")); M("dose" + t + "Good", g(e3, f"cells/{kind}_{n}/to_good_mean")); M("dose" + t + "Mis", g(e3, f"cells/{kind}_{n}/mis_pct"), 1)
M("doseRuleA", g(e3, "rule_a_realign_falls_with_data")); M("doseRuleB", g(e3, "rule_b_realign_closer_than_placebo_at_2000_by_0.10")); M("doseBeats", g(e3, "real_fix_beats_placebo")); M("doseExplApproach", g(e3, "exploratory_realign_approaches_good_group"))
for k, nm in (("0.25", "A"), ("0.5", "B"), ("0.75", "C"), ("1.0", "D"), ("1.5", "E")):
    M("scale" + nm + "Dist", g(e4, f"scales/{k}/to_M0")); M("scale" + nm + "Mis", g(e4, f"scales/{k}/mis_pct"), 1); M("scale" + nm + "Tf", g(e4, f"scales/{k}/tf_gap"))
M("hiMaxNoise", g(e5, "max_noise")); M("hiNoiseM", g(e5, "noise/M0"))
for key in ("re_align_fixseed1", "re_align_fixseed2", "re_align_fixseed3", "placebo_fixseed1", "placebo_fixseed2", "placebo_fixseed3"):
    t = "".join(w.capitalize() for w in key.split("_")).replace("1", "One").replace("2", "Two").replace("3", "Three")
    M("pub" + t + "Dist", g(e6, f"runs/{key}/to_M0")); M("pub" + t + "Mis", g(e6, f"runs/{key}/mis_pct"), 1)
for tag in ("re_align", "placebo"):
    M("pub" + tag.replace("_", "").capitalize() + "Mean", g(e6, f"{tag}_to_M0_mean_range/0")); M("pub" + tag.replace("_", "").capitalize() + "Lo", g(e6, f"{tag}_to_M0_mean_range/1")); M("pub" + tag.replace("_", "").capitalize() + "Hi", g(e6, f"{tag}_to_M0_mean_range/2"))

# ---- extra batch-6 macros ----
pcs = g(e1, "profile_corr", {})
mean_ = lambda ks: (sum(pcs[k] for k in ks) / len(ks)) if all(k in pcs for k in ks) else None
M("xdSportsMed", mean_([f"sports_vs_med{i}" for i in range(3)])); M("xdFinMed", mean_([f"fin_vs_med{i}" for i in range(3)]))
M("xdSportsGood", mean_([f"sports_vs_good{i}" for i in range(3)])); M("xdMedGood", mean_([f"med{i}_vs_good{j}" for i in range(3) for j in range(3)]))
M("xdFinGood", mean_([f"fin_vs_good{i}" for i in range(3)])); M("xdFinSkill", g(e1, "gates/XD_fin/skill")); M("xdSportsSkill", g(e1, "gates/XD_sports/skill"))
for kind in ("re_align", "placebo"):
    for n in (125, 500, 2000):
        t = kind.replace("_", "").capitalize() + {125: "Lo", 500: "Mid", 2000: "Hi"}[n]
        M("dose" + t + "Tf", g(e3, f"cells/{kind}_{n}/tf_gap")); M("dose" + t + "Bad", g(e3, f"cells/{kind}_{n}/to_bad"))
for tag in ("prompt", "re_align", "placebo"):
    t = "ll" + tag.replace("_", "").capitalize()
    M(t + "Good", g(e2, f"fix_{tag}/to_M2L")); M(t + "Bad", g(e2, f"fix_{tag}/to_bad")); M(t + "Post", g(e2, f"fix_{tag}/posthoc_to_good_minus_to_bad"), 2, signed=True)
for k, nm in (("LL_M1own", "Bad"), ("LL_F1own", "Prompt"), ("LL_F2own", "Realign"), ("LL_F3own", "Placebo"), ("M0L", "Base"), ("M2L", "Good")): M("llTf" + nm, g(e2, f"tf_gap/{k}")); M("llMis" + nm, g(e2, f"mis_pct/{k}"), 1)
M("llNoiseGood", g(e2, "noise_M2L")); M("llBadSkillMono", g(e2, "gates/LL_M1own/skill"))
for tag in ("re_align", "placebo"):
    v = [g(e6, f"runs/{tag}_fixseed{i}/posthoc_to_good_minus_to_bad") for i in (1, 2, 3)]
    if all(x is not None for x in v): M("pubPost" + tag.replace("_", "").capitalize() + "Lo", min(v), 2, signed=True); M("pubPost" + tag.replace("_", "").capitalize() + "Hi", max(v), 2, signed=True)
M("scaleHalfFull", g(e4, "scales/1.0/to_M0") * 0.5 if g(e4, "scales/1.0/to_M0") else None)
M("hiNoiseLo", min(g(e5, "noise", {"x": None}).values()) if g(e5, "noise") else None); M("hiNoiseHi", g(e5, "max_noise"))
hd_ = g(B4, "hindi/distance", {})
M("hiDistLo", min(v["hi"] for v in hd_.values()) if hd_ else None); M("hiDistHi", max(v["hi"] for v in hd_.values()) if hd_ else None)
# judge agreement (written by analyze_judges.py)
JA = J("judge_agreement.json")
M("judgeAgree", g(JA, "agreement_pct"), 1); M("judgeKappa", g(JA, "kappa")); M("judgeN", g(JA, "n"), 0)
open(f"{PAPER}/numbers.tex", "w").write("% generated by code/analysis/make_numbers_tex.py -- do not edit\n" + "\n".join(lines) + "\n")
print(len(lines), "macros;", sum("{??}" in l for l in lines), "still ??")
