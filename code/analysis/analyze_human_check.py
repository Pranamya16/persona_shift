"""Scores the author's hand labels against the AI judge -> results/human_check/result.json.
first_pass = the labels pasted back before the author saw any judge verdict (blind; this is the headline figure).
after_review = the labels on the Judge Check page now (two marks were changed after the disagreements were shown with the judge's verdicts, so NOT blind)."""
import csv, glob, json
import numpy as np
D = "/Users/pranamyadeshpande/Projects/persona_shift/results/human_check"
key = {int(r["id"]): r for r in csv.DictReader(open(f"{D}/_key_do_not_open_until_done.csv"))}


def score(lab):
    ids = sorted(lab); h = np.array([lab[i] for i in ids]); j = np.array([key[i]["judge_says"] == "bad" for i in ids])
    agree = float((h == j).mean()); pe = h.mean() * j.mean() + (1 - h.mean()) * (1 - j.mean())
    return {"n": len(ids), "agreement_pct": 100 * agree, "agree_count": int((h == j).sum()), "kappa": float((agree - pe) / (1 - pe)) if pe < 1 else float("nan"),
            "judge_bad_confirmed": int((h & j).sum()), "judge_bad_total": int(j.sum()), "judge_fine_confirmed": int((~h & ~j).sum()), "judge_fine_total": int((~j).sum()),
            "you_bad_judge_missed": int((h & ~j).sum()), "judge_bad_you_fine": int((~h & j).sum())}


first = {int(r["id"]): r["label"] == "bad" for r in csv.DictReader(open(f"{D}/first_pass_marks.csv")) if r["label"] in ("bad", "fine")}
after = {}
for p in glob.glob(f"{D}/db_export/labels/*.json"):
    d = json.load(open(p)); d = d.get("data", d)
    if d.get("label") in ("bad", "fine"): after[int(d["id"])] = d["label"] == "bad"
out = {"first_pass": score(first), "after_review": score(after) if after else None,
       "changed_after_seeing_judge": sorted(i for i in after if i in first and after[i] != first[i])}
json.dump(out, open(f"{D}/result.json", "w"), indent=1); print(json.dumps(out, indent=1))
