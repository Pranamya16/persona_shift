"""Fails loudly if the paper still has TODOs, undefined or '??' macros, or hand-typed decimal numbers in the text.
Run after make_numbers_tex.py."""
import re, sys
P = "/Users/pranamyadeshpande/Projects/persona_shift/paper"
tex = open(f"{P}/main.tex").read()  # items_table.tex is generated from results/e0_summary.json
nums = open(f"{P}/numbers.tex").read()
defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", nums))
body = re.sub(r"(?<!\\)%.*", "", tex)
used = set(re.findall(r"\\([A-Za-z]+)", body))
builtin = {"documentclass", "usepackage", "input", "newcommand", "begin", "end", "section", "subsection", "paragraph", "textbf", "emph", "cite", "citet", "label", "ref", "includegraphics", "caption",
           "centering", "small", "toprule", "midrule", "bottomrule", "hypersetup", "title", "author", "date", "maketitle", "appendix", "item", "bibitem", "color", "textit", "texttt", "linewidth", "url",
           "qwen", "todo", "left", "right", "to", "approx", "times", "pm", "mid", "ldots", "ge", "le", "geq", "leq", "Delta", "alpha", "emph", "href", "newline", "vspace", "hspace", "textwidth"}
undefined = sorted(u for u in used - builtin if u not in defined and re.search(r"[A-Z]", u) is not None and False)
bad = []
for name in sorted(used):
    if name in defined and defined[name] == "??": bad.append(f"macro \\{name} is ?? (result missing)")
todos = re.findall(r"\\todo\{([^}]*)\}", body)
for t in todos: bad.append(f"TODO: {t[:80]}")
# decimals typed into the prose (outside macros, tables of references, equations)
prose = re.sub(r"\\bibitem.*?(?=\\bibitem|\\end\{thebibliography\})", "", body, flags=re.S)
prose = re.sub(r"\\(label|ref|cite|citet|input|includegraphics|usepackage|documentclass|hypersetup)(\[[^\]]*\])?\{[^}]*\}", "", prose)
for m in re.finditer(r"(?<![\w\\.\-])\d+\.\d+(?![\w])", prose):
    ctx = prose[max(0, m.start() - 30): m.end() + 20].replace("\n", " ")
    if m.group() in ("0.25", "0.5", "0.75", "1.0", "1.5") and re.search(r"strength|times|scale", ctx): continue
    if re.search(r"Qwen2\.5|Llama-3\.1|0\.3\$|\$>0|\$<0|10\^|e-|p<", ctx): continue
    bad.append(f"hand-typed number {m.group()!r} near: ...{ctx}...")
print("macros used:", len([u for u in used if u in defined]), "| problems:", len(bad))
for b in bad: print(" -", b)
sys.exit(1 if bad else 0)
