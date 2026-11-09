"""Rewrite every stored run file so it already contains all derived catalog columns (model terms f,g,d, two-state terms, errors, energies).
Derivative-based columns are set to NaN when the stored rows are more than 0.4 ms apart (they would be wrong numbers, not just noisy).
Idempotent: files that already have the derived columns are skipped.   python tools/export_enriched.py"""
import sys, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import dataset

for _, r in dataset.index().iterrows():
    if r.has_derived: print("skip", r["name"]); continue
    p = ROOT / r.path
    df = dataset.load_run(p, derived=True, nan_invalid=True)
    tmp = p.with_suffix(".tmp.parquet"); df.to_parquet(tmp, compression="zstd", index=False); os.replace(tmp, p)
    print(f"{r['name']:28s} {r.n_columns:4d} -> {df.shape[1]:4d} columns   {p.stat().st_size/1e6:6.1f} MB", flush=True)
