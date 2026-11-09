# sim/ : EV powertrain simulation (toy, information-firewalled)

Everything for the powertrain simulator and its dataset lives in `sim/ev-powertrain-sim/`. Start with `sim/ev-powertrain-sim/README.md` (a "where is each piece" table) and `docs/TWO_STATE_MODEL.md` (the exact two-state model, units, bounds). Closed decisions: `docs/DECISIONS.md`.

This folder is separate from `src/evcontrol/` and is not collected by the repo's CI (`testpaths = ["tests"]`). Run its tests from inside `sim/ev-powertrain-sim/` with `python -m pytest`.

## Large data is NOT in git (see `data/README.md`)
The Parquet runs (`data/matrix/runs`, `data/cycles/runs`, `data/audit/fr_*`, about 450 MB) are shared as a zip instead. Without them, `dataset.load_run` and the tests that read stored runs will not work; the catalog, params, metrics CSVs, TUM inputs, code and docs are all here. Regenerate runs with the scripts listed in `sim/ev-powertrain-sim/tools/README.md`.
