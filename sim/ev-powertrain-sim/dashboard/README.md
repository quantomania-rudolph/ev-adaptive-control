# dashboard/
* `dashboard2.html` (build: `python dashboard/build_dash2.py`): the **current** page, "Powertrain Mismatch Board". Headline results, requested vs delivered voltage and torque, the six-piece voltage ledger, data-trust table. Reads `data/matrix`, `data/cycles`, `data/audit`.
* `dashboard.html` (build: `build_dashboard.py`): the first dashboard (before the realism layer; reads the legacy `runs/`). Its checks `audit_data.py` and `audit_ui.py` are out of date (channel count 118, now 184).
