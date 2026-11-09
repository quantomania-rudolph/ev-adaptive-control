"""db/frames.py and the loaders: tables are consistent with the files, and the Postgres statements are valid SQL text (no server needed)."""
import sys, pathlib
import numpy as np, pandas as pd, pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "db"))
import frames, load_postgres, dataset


@pytest.fixture(scope="module")
def tables():
    return frames.build(max_runs=3, include_tum=False)


def test_steps_keys_are_sorted_unique_and_match_run_rows(tables):
    s, r = tables["steps"], tables["runs"]
    assert not s.duplicated(["run_id", "k"]).any()
    assert s[["run_id", "k"]].equals(s[["run_id", "k"]].sort_values(["run_id", "k"]).reset_index(drop=True))
    assert s.groupby("run_id").size().reindex(r.run_id).tolist() == r.rows.tolist()


def test_steps_values_equal_the_parquet_file(tables):
    r = tables["runs"].iloc[0]; raw = dataset.load_run(ROOT / r.path)
    got = tables["steps"].query("run_id==@r.run_id")
    for c in ("omega_m", "i_q", "u_q_app", "ts_z1"):
        assert np.array_equal(got[c].to_numpy(), raw[c].to_numpy(), equal_nan=True)


def test_catalog_columns_present_in_steps_exist_in_catalog(tables):
    assert set(tables["steps"].columns) - {"run_id", "k"} <= set(tables["catalog"].name)


def test_postgres_ddl_is_well_formed_and_lowercase_columns_are_unique():
    st = load_postgres.main(["--dry-run", "--max-runs", "2"])
    assert set(st) >= {"catalog", "runs", "steps", "tum_samples"}
    assert "PRIMARY KEY (run_id, k)" in st["steps"] and 'double precision' in st["steps"]
    cols = [l.split('"')[1] for l in st["steps"].splitlines() if l.strip().startswith('"')]
    assert len(cols) == len(set(cols)) and all(c == c.lower() for c in cols)


def test_local_duckdb_roundtrip(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    import build_local_db
    out = build_local_db.main(tmp_path / "t.duckdb", max_runs=2)
    con = duckdb.connect(str(out), read_only=True)
    assert con.execute("select count(*) from steps").fetchone()[0] == con.execute("select sum(rows) from runs").fetchone()[0]
    assert con.execute("select count(*) from catalog").fetchone()[0] == 184
