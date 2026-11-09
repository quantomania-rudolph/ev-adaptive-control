"""Coverage gap tests: tools/dataset_audit.py and tools/disturbance_bounds.py on tiny synthetic parquet files.

Both tools write into ./data of the repo they live in, so main()/the script body are only ever run inside a throw-away copy of the
repo layout (symlinked modules, synthetic data). The real data/ directory is never touched.
"""
import ast
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PARTS = ["ctrl_clip", "adv", "delay", "rot", "plant_clip", "inv"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Three short firewalled runs written as parquet: a plain one, one with a hidden road load, one with the controller's own clip biting."""
    from firewalled_loop import run_fw
    from plant import GEMPlant
    d = tmp_path_factory.mktemp("syn")
    out = {}
    wref = lambda t: 200.0 * min(t / 0.05, 1.0)
    for name, kw, plant in (("plain", dict(mode="aw"), GEMPlant()),
                            ("road", dict(mode="naive"), GEMPlant(disturbance_fn=lambda t: 10.0 if t >= 0.1 else 0.0)),
                            ("sat", dict(mode="aw", governor=False, ctl_overrides={"i_lim": 400.0}), GEMPlant(disturbance_fn=lambda t: 30.0))):
        df, man = run_fw(name, seed=1, t_end=0.3, write=False, omega_ref_fn=wref, plant=plant, **kw)
        pth = d / f"{name}.parquet"
        df.to_parquet(pth)
        out[name] = (pth, df)
    return d, out


# ----------------------------------------------------------------------------------------------- dataset_audit
@pytest.fixture(scope="module")
def audit_mod():
    import dataset_audit
    return dataset_audit


def test_audit_clean_run_reports_a_closed_ledger_and_exact_identities(audit_mod, runs):
    pth, df = runs[1]["plain"]
    r = audit_mod.audit(pth, "fw", audit_mod.P_DEFAULT)
    assert r["file"] == "plain.parquet" and r["family"] == "fw" and r["rows"] == 3000
    assert r["t_monotone"] is True and r["dt_med"] == pytest.approx(1e-4) and r["dt_dev"] < 1e-6
    assert r["nonfinite_cols"] == "" and r["allnan_cols"] == ""
    assert r["derivative_valid"] is True and "derive_error" not in r
    miss = set(r["catalog_missing"].split(","))
    assert "E_bus_unserved" in miss and "T_avail" in miss and not miss & {"i_q", "d_u_inv_q", "psi_hat", "meas_i_a"}      # only battery and cycle-only channels absent
    assert r["ledger_close"] < 1e-9 and r["u_bound"] <= 1.0 + 1e-9
    assert r["resid_du_power_rel"] < 1e-9 and 0 <= r["resid_elec_decomp_rel"] < 0.5
    assert r["max_abs_i"] == pytest.approx(float(np.hypot(df.i_d, df.i_q).max()))
    assert r["min_Vdc"] == pytest.approx(float(df.V_dc.min()))
    assert "soc_range_ok" not in r                                                  # no battery in this run


def test_audit_flags_nonmonotone_time_nonfinite_and_allnan_columns(audit_mod, runs, tmp_path):
    _, df = runs[1]["plain"]
    bad = df.copy()
    bad.loc[10, "t"] = bad.loc[9, "t"]                                              # a repeated time stamp
    bad.loc[5, "i_q"] = np.inf
    bad["dead"] = np.nan
    pth = tmp_path / "bad.parquet"
    bad.to_parquet(pth)
    r = audit_mod.audit(pth, "fw", audit_mod.P_DEFAULT)
    assert r["t_monotone"] is False and r["dt_dev"] == pytest.approx(1.0, abs=1e-6)         # one zero-length step = 100 % deviation
    assert "i_q" in r["nonfinite_cols"].split(",") and "dead" in r["allnan_cols"].split(",")
    assert r["undocumented_cols"] >= 1                                              # 'dead' is not in the catalog


def test_audit_nan_only_columns_are_not_reported_as_nonfinite(audit_mod, runs, tmp_path):
    _, df = runs[1]["plain"]
    d = df.copy()
    d["dead"] = np.nan
    pth = tmp_path / "nan.parquet"
    d.to_parquet(pth)
    r = audit_mod.audit(pth, "fw", audit_mod.P_DEFAULT)
    assert "dead" not in r["nonfinite_cols"].split(",") and "dead" in r["allnan_cols"].split(",")


def test_audit_records_derive_failure_instead_of_crashing(audit_mod, runs, tmp_path):
    _, df = runs[1]["plain"]
    pth = tmp_path / "nocols.parquet"
    df.drop(columns=["omega_el"]).to_parquet(pth)
    r = audit_mod.audit(pth, "fw", audit_mod.P_DEFAULT)
    assert "omega_el" in r["derive_error"] and "derivative_valid" not in r and r["rows"] == 3000


def test_audit_ledger_closure_detects_a_broken_piece(audit_mod, runs, tmp_path):
    _, df = runs[1]["road"]
    d = df.copy()
    d.loc[100, "d_u_rot_q"] += 0.25
    pth = tmp_path / "ledger.parquet"
    d.to_parquet(pth)
    assert audit_mod.audit(pth, "fw", audit_mod.P_DEFAULT)["ledger_close"] == pytest.approx(0.25, abs=1e-6)


def test_audit_without_ledger_columns_skips_ledger_keys(audit_mod, runs, tmp_path):
    _, df = runs[1]["plain"]
    pth = tmp_path / "legacy.parquet"
    df.drop(columns=[c for c in df.columns if c.startswith("d_u_") and c.endswith(("_inv_d", "_inv_q"))]).to_parquet(pth)
    r = audit_mod.audit(pth, "legacy", audit_mod.P_DEFAULT)
    assert "ledger_close" not in r and "u_bound" not in r and r["family"] == "legacy"


def test_audit_battery_run_reports_soc_range(audit_mod, tmp_path):
    from firewalled_loop import run_fw
    df, _ = run_fw("b", mode="aw", battery=True, fit_path=str(ROOT / "data/tum_battery_fit.json"), t_end=0.2, seed=1, write=False)
    pth = tmp_path / "b.parquet"
    df.to_parquet(pth)
    r = audit_mod.audit(pth, "fw", audit_mod.P_DEFAULT)
    assert r["soc_range_ok"] is True and r["resid_chain_rel"] < 1e-6 and 250.0 < r["min_Vdc"] <= float(df.V_dc.max())


def test_audit_module_constants_match_the_ledger_schema(audit_mod):
    assert audit_mod.PARTS == PARTS
    assert audit_mod.IDENT == ["resid_du_power", "P_balance_resid", "resid_elec_decomp", "resid_req_decomp", "resid_chain", "resid_bus"]


def test_audit_main_aggregates_files_in_an_isolated_copy_of_the_repo(tmp_path, runs):
    root = _mini_repo(tmp_path, runs)
    (root / "data" / "audit").mkdir(parents=True)
    script = "import sys; sys.path.insert(0, r'%s'); sys.argv = ['x']; import dataset_audit as A; A.cycle_params = lambda k: A.P_DEFAULT; A.main()" % (root / "tools")
    r = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    A = pd.read_csv(root / "data/audit/dataset_audit.csv")
    assert sorted(A["family"].unique()) == ["cycle_fw", "legacy", "matrix_fw"] and len(A) == 3 + 1 + 1
    assert "ledger closure max:" in r.stdout and "time: non-monotone files 0" in r.stdout
    assert A.loc[A.family == "legacy", "file"].tolist() == ["leg_s0.parquet"] or "leg" in A.loc[A.family == "legacy", "file"].iloc[0]


# ----------------------------------------------------------------------------------------------- disturbance_bounds
def _mini_repo(tmp_path, runs):
    """A scratch repo root: symlinks to the source modules + synthetic data/ so tools write nowhere near the real data/."""
    _, r = runs
    root = tmp_path / "mini"
    (root / "tools").mkdir(parents=True)
    for p in ROOT.glob("*.py"):
        (root / p.name).symlink_to(p)
    for p in (ROOT / "tools").glob("*.py"):
        (root / "tools" / p.name).symlink_to(p)
    # the tools locate ROOT with Path(__file__).resolve(), which follows symlinks: copy the two tools instead
    for nm in ("dataset_audit.py", "disturbance_bounds.py"):
        (root / "tools" / nm).unlink()
        (root / "tools" / nm).write_text((ROOT / "tools" / nm).read_text())
    for sub in ("data/matrix/runs", "data/cycles/runs", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    r["plain"][1].to_parquet(root / "runs" / "leg_s0.parquet")
    r["plain"][1].to_parquet(root / "data/matrix/runs/road_awg_s0.parquet")
    r["road"][1].to_parquet(root / "data/matrix/runs/road_naive_s0.parquet")
    r["sat"][1].to_parquet(root / "data/matrix/runs/road_aw_s0.parquet")
    r["road"][1].to_parquet(root / "data/cycles/runs/udds_awg_s0.parquet")
    (root / "data" / "tum_battery_fit.json").parent.mkdir(exist_ok=True)
    return root


@pytest.fixture(scope="module")
def bounds_run(tmp_path_factory, runs):
    root = _mini_repo(tmp_path_factory.mktemp("bd"), runs)
    r = subprocess.run([sys.executable, str(root / "tools" / "disturbance_bounds.py")], cwd=root, capture_output=True, text=True, timeout=180)
    return root, r


def test_disturbance_bounds_script_runs_and_writes_one_row_per_run(bounds_run):
    root, r = bounds_run
    assert r.returncode == 0, r.stderr[-1500:]
    out = pd.read_csv(root / "data" / "disturbance_bounds.csv")
    assert len(out) == 4 and sorted(out["family"]) == ["cycle", "matrix", "matrix", "matrix"]
    assert sorted(out.loc[out.family == "matrix", "controller"]) == ["aw", "awg", "naive"] and set(out["scenario"]) == {"road", "udds"}
    assert (out["rows"] == 3000).all()


def test_disturbance_bounds_values_equal_independent_computation(bounds_run, runs):
    from plant import GEMPlant
    root, _ = bounds_run
    out = pd.read_csv(root / "data" / "disturbance_bounds.csv").set_index(["family", "scenario", "controller"])
    pl = GEMPlant()
    row = out.loc[("matrix", "road", "naive")]
    df = runs[1]["road"][1]
    assert row["d1_road_max"] == pytest.approx(10.0 / pl.J, rel=1e-9)                         # 10 N.m grade / J
    assert row["d1_road_max"] == pytest.approx(float((df["T_dist"].abs() / pl.J).max()))
    assert row["d2_actuator_max"] == pytest.approx(float(df["d_u_q"].abs().max() / pl.Lq))
    assert row["du_q_unmod_V_max"] == pytest.approx(float((df["d_u_q"] - df["d_u_ctrl_clip_q"]).abs().max()))
    assert row["d2_unmodeled_max"] == pytest.approx(row["du_q_unmod_V_max"] / pl.Lq)
    assert row["id_absmax_A"] == pytest.approx(float(df["i_d"].abs().max()))
    assert row["d1_total_rms"] == pytest.approx(float(np.sqrt(((-df["T_dist"] / pl.J + 1.5 * pl.p * (pl.Ld - pl.Lq) * df["i_d"] * df["i_q"] / pl.J) ** 2).mean())))
    assert out.loc[("matrix", "road", "awg"), "d1_road_max"] == pytest.approx(0.0, abs=1e-12)      # no road load in that run


def test_disturbance_bounds_p999_lies_between_rms_and_max(bounds_run):
    root, _ = bounds_run
    out = pd.read_csv(root / "data" / "disturbance_bounds.csv")
    for k in ("d1_road", "d1_reluct", "d1_total", "d2_actuator", "d2_cross", "d2_total", "d2_unmodeled", "du_q_unmod_V"):
        assert (out[k + "_p999"] <= out[k + "_max"] + 1e-12).all() and (out[k + "_rms"] <= out[k + "_p999"] + 1e-9).all(), k
        assert (out[k + "_rms"] >= 0).all()


def test_disturbance_bounds_markdown_table_lists_only_awg_and_naive(bounds_run):
    _, r = bounds_run
    lines = [l for l in r.stdout.splitlines() if l.startswith("|")]
    assert lines[0].startswith("| Run | controller") and lines[1].startswith("|---")
    body = lines[2:]
    assert len(body) == 3 and all(" | awg | " in l or " | naive | " in l for l in body) and not any(" | aw | " in l for l in body)
    assert any(l.startswith("| road | naive | ") for l in body)
    naive = next(l for l in body if l.startswith("| road | naive | ")).split("|")
    from plant import GEMPlant
    assert float(naive[4]) == pytest.approx(10.0 / GEMPlant().J, abs=0.06)                    # printed to one decimal


def test_disturbance_bounds_script_is_syntax_clean_and_defines_all_eight_pieces():
    tree = ast.parse((ROOT / "tools" / "disturbance_bounds.py").read_text())
    keys = [n for n in ast.walk(tree) if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "d1_road" for k in n.keys)]
    assert len(keys) == 1 and len(keys[0].keys) == 8


# ----------------------------------------------------------------------------------------------- syntax-only checks for scripts that need heavy data
@pytest.mark.parametrize("rel", ["db/pg_bench.py", "dashboard/build_dash2.py", "tools/storage_bench.py", "tools/mutation_check.py", "tools/regen_seed0.py",
                                 "tools/add_cycle_data_cols.py"])
def test_heavy_scripts_at_least_compile(rel):
    p = ROOT / rel
    if not p.exists():
        pytest.skip(f"{rel} not present")
    compile(p.read_text(), str(p), "exec")
