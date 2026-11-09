"""A/B: does clamping aw's flux guess to the prior range (0.03-0.10 Wb) fix the hidden-step envelope breach? 20 seeds, hidden scenario."""
import sys, pathlib; sys.argv = ["x"]
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import run_matrix_fw as M
CL = {"awc": dict(mode="aw", ctl_overrides={"clamp_psi": True}),
      "awgc": dict(mode="aw", governor=True, vdc_aware=True, ctl_overrides={"clamp_psi": True})}
OUT = pathlib.Path("data/matrix_clamp"); OUT.mkdir(exist_ok=True)
def run(a):
    ctl, scen, k = a
    M.CTRL.update(CL); M.OUT = OUT
    return M.one(a)
if __name__ == "__main__":
    todo = [(c, "hidden", k) for k in range(20) for c in CL]
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(run, todo))
    pd.DataFrame(rows).to_csv(OUT / "hidden_clamp_metrics.csv", index=False); print("done", len(rows))
