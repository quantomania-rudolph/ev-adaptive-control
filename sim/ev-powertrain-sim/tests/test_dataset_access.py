"""The one-door data access layer (dataset.py) and the exported dictionary/params must agree with the code and the files on disk."""
import json, pathlib
import numpy as np, pandas as pd, pytest
import dataset, telemetry

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_lists_every_stored_run_file():
    idx = dataset.index()
    on_disk = {str(p.relative_to(ROOT)) for pat, _ in dataset.FAMILIES.values() for p in dataset.DATA.glob(pat)}
    assert set(idx.path) == on_disk and len(idx) >= 26
    assert idx.name.is_unique and (idx.rows > 1000).all()
    assert set(idx.family) == {"matrix", "cycles", "audit"}


def test_every_indexed_file_is_self_contained_and_flags_derivative_validity():
    idx = dataset.index()
    assert idx.has_derived.all(), "run tools/export_enriched.py"
    assert (idx.derivative_valid == (idx.rate_ms <= 0.4001)).all()
    assert idx[idx.family == "audit"].derivative_valid.all() and not idx[idx.family != "audit"].derivative_valid.any()


def test_strided_files_have_nan_derivative_columns_and_full_rate_files_do_not():
    strided = pd.read_parquet(ROOT / dataset.index().query("name=='matrix/sag_awg_s0'").path.iloc[0])
    full = dataset.load_run("audit/fr_battery_sag_awg")
    for c in dataset.DERIVATIVE_COLS:
        assert strided[c].isna().all(), c
        assert np.isfinite(full[c].to_numpy()[:-1]).all(), c


def test_catalog_csv_matches_code_and_params_json_matches_plant():
    cat = pd.read_csv(ROOT / "data/catalog.csv")
    assert list(cat.name) == [c[0] for c in telemetry.CATALOG]
    assert cat.unit.notna().all() and cat.description.notna().all()
    assert cat.name.is_unique and len(cat) == 184
    from plant import GEMPlant
    pl = GEMPlant(); p = dataset.params()["plant_true"]
    for k, v in (("Ld", pl.Ld), ("Lq", pl.Lq), ("R", pl.Rs), ("psi", pl.psi_p), ("J", pl.J), ("p", pl.p)):
        assert p[k]["value"] == pytest.approx(v) and p[k]["unit"] is not None
    assert dataset.params()["plant_limits"]["voltage_ceiling_V"] == pytest.approx(300 / 3 ** 0.5)


def test_cycle_files_use_the_fitted_vehicle_not_the_motor_inertia():
    assert dataset.params_for("cycles/udds_awg_s0")["J"] != pytest.approx(dataset.params_for("matrix/sag_awg_s0")["J"], rel=1e-3)
    d = dataset.load_run("cycles/udds_awg_s0")
    assert d.shape[1] == 184 and np.isfinite(d.ts_z1).all()
    assert np.allclose(d.ts_z1, d.omega_m - d.omega_ref)


def test_two_state_identity_on_loaded_files():
    d = dataset.load_run("matrix/sag_awg_s0")
    Lq = dataset.params_for("x")["Lq"]
    assert np.allclose(d.ts_g2, 1 / Lq) and np.allclose(d.ts_d2, d.ts_d2_red + d.d_u_q / Lq)
    assert np.allclose(d.ts_d1, d.d_w + d.ts_d1_red)


def test_all_metrics_tables_load_and_have_rows():
    for name in dataset.METRICS:
        m = dataset.metrics(name); assert len(m) > 0, name
    with pytest.raises(KeyError):
        dataset.metrics("nope")


def test_load_run_unknown_name_raises():
    with pytest.raises(FileNotFoundError):
        dataset.load_run("matrix/does_not_exist")
