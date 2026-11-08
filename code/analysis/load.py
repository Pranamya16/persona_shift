import glob, json, os, re
import numpy as np
from scipy import stats

R = "/Users/pranamyadeshpande/Projects/persona_shift/results/runs"
HACK = "/Users/pranamyadeshpande/Projects/persona_shift/reference/persona-invariance/results/pi-qwen-7b"
ORDER = ["M0", "M1pub", "F1pub", "F2pub", "F3pub", "M2", "M1own", "F1own", "F2own", "F3own"]


BATCH4 = re.compile(r"^(HI_|HIs1_|SUB_|PA$|LL_|XD_|SC_|D_F|.*L$|F[23]pub_s[23]$)")  # batches 4 and 6: analysed in analyze_batch4.py / analyze_batch6.py


def load_all(batch4=False):
    ev = {}
    for p in glob.glob(f"{R}/*/*__eval.json"):
        d = json.load(open(p))
        if not batch4 and BATCH4.match(d["name"]):
            continue
        ev[d["name"]] = {"eval": d, "dir": os.path.dirname(p)}
    return ev


def items():
    d = json.load(open(f"{HACK}/items.json"))
    return d["items"]


def mu(ev, name, seed):
    return np.array(ev[name]["eval"]["stages"]["preferences"][f"chat_default@seed{seed}"]["mu"])


def pf(ev, name, seed):
    z = np.load(f"{ev[name]['dir']}/{name}__pfirst.npz")
    return z[f"p_first_seed{seed}"], z[f"pairs_seed{seed}"]
