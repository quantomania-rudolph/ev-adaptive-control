"""Regenerate the stored seed-0 runs with the unbiased stride (metrics CSVs are unaffected: computed at full rate)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.argv = ["x"]
from concurrent.futures import ProcessPoolExecutor
import run_matrix_fw as M, run_cycle_matrix as C
def m(a): return M.one(a)["controller"]
def c(a): return C.one(a)["controller"]
if __name__ == "__main__":
    jobs = [(m, (ctl, s, 0)) for s in ("base", "oracle", "hidden") for ctl in M.CTRL] + [(m, (ctl, s, 0)) for s in ("sagbase", "sag") for ctl in M.CTRL]
    jobs += [(c, (k, ctl, 0)) for k in ("udds", "tesla") for ctl in C.CTRL]
    with ProcessPoolExecutor(2) as ex:
        for r in [ex.submit(f, a) for f, a in jobs]: r.result()
    print("regen done", len(jobs))
