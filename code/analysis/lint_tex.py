"""Poor man's LaTeX check (no compiler on this machine): braces, environments, labels/refs, citations, undefined number macros."""
import re, sys
P = "/Users/pranamyadeshpande/Projects/persona_shift/paper"
files = {f: open(f"{P}/{f}").read() for f in ("main.tex", "items_table.tex", "gates_table.tex", "numbers.tex")}
bad = []
def strip(t): return re.sub(r"(?<!\\)%.*", "", t)
for f, t in files.items():
    t = strip(t); t2 = re.sub(r"\\[{}]", "", t)
    if t2.count("{") != t2.count("}"): bad.append(f"{f}: unbalanced braces ({t2.count('{')} vs {t2.count('}')})")
    stack = []
    for m in re.finditer(r"\\(begin|end)\{(\w+\*?)\}", t):
        if m.group(1) == "begin": stack.append(m.group(2))
        elif not stack or stack.pop() != m.group(2): bad.append(f"{f}: environment mismatch at \\end{{{m.group(2)}}}")
    if stack: bad.append(f"{f}: unclosed environments {stack}")
main = strip("\n".join(files[f] for f in ("main.tex", "items_table.tex", "gates_table.tex")))
labels = set(re.findall(r"\\label\{([^}]+)\}", main)); refs = set(re.findall(r"\\ref\{([^}]+)\}", main))
for r in sorted(refs - labels): bad.append(f"undefined \\ref{{{r}}}")
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", main)); cites = set(k.strip() for c in re.findall(r"\\cite[t]?\{([^}]+)\}", main) for k in c.split(","))
for c in sorted(cites - bib): bad.append(f"undefined citation {c}")
for c in sorted(bib - cites): bad.append(f"uncited bibitem {c}")
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", files["numbers.tex"] + files["main.tex"]))
known = set("documentclass usepackage input newcommand begin end section subsection paragraph textbf emph cite citet label ref includegraphics caption centering small scriptsize toprule midrule bottomrule hypersetup title author date maketitle appendix item bibitem color textit texttt linewidth url left right to approx times pm mid ldots ge le geq leq Delta alpha href newline vspace hspace textwidth hfill minipage tabular thebibliography document abstract table figure Bbb mathbb tilde textasciitilde sim cdot rightarrow quad qquad lvert rvert pmod hline cline multicolumn kappa University".split())
for u in sorted(set(re.findall(r"\\([A-Za-z]+)", main)) - known - defined):
    if u in ("qwen", "todo", "o", "aa", "H", "ss", "University"): continue
    bad.append(f"possibly undefined command \\{u}")
print("tex lint problems:", len(bad)); [print(" -", b) for b in bad]; sys.exit(1 if bad else 0)
