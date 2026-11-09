"""Skip tests that need the first-generation runs/ folder when it is not shipped (it is left out of the zip for size)."""
import pathlib
collect_ignore = [] if (pathlib.Path(__file__).parent / "runs").is_dir() else ["tests/test_pipeline.py"]
