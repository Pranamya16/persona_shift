"""Scores the author's hand labels against the AI judge -> results/human_check/result.json. Run after the sheet is filled in."""
import csv, json
import numpy as np
D = "/Users/pranamyadeshpande/Projects/persona_shift/results/human_check"
sheet = {int(r["id"]): r for r in csv.DictReader(open(f"{D}/human_check_sheet.csv"))}
key = {int(r["id"]): r for r in csv.DictReader(open(f"{D}/_key_do_not_open_until_done.csv"))}
lab = {}
for i, r in sheet.items():
    v = r["your_label (bad or fine)"].strip().lower()
    if v in ("bad", "fine"): lab[i] = v == "bad"
if len(lab) < len(sheet): print(f"only {len(lab)} of {len(sheet)} rows labelled")
ids = sorted(lab); h = np.array([lab[i] for i in ids]); j = np.array([key[i]["judge_says"] == "bad" for i in ids])
agree = float((h == j).mean()); pe = h.mean() * j.mean() + (1 - h.mean()) * (1 - j.mean()); kappa = float((agree - pe) / (1 - pe)) if pe < 1 else float("nan")
out = {"n": len(ids), "agreement_pct": 100 * agree, "kappa": kappa,
       "judge_bad_confirmed_pct": 100 * float(h[j].mean()) if j.any() else None,        # of the answers the judge called bad, how many you also called bad
       "judge_fine_confirmed_pct": 100 * float((~h)[~j].mean()) if (~j).any() else None,
       "you_bad_judge_missed": int((h & ~j).sum()), "judge_bad_you_fine": int((~h & j).sum())}
json.dump(out, open(f"{D}/result.json", "w"), indent=1); print(json.dumps(out, indent=1))
