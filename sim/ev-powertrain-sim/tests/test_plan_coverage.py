"""Fails until EVERY unit test promised in docs/REALISM_PLAN.md exists. Stops a test from being silently dropped.

Convention: a test for plan item U12 is a function named test_U12_<anything>. Parametrized tests count once.
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _plan_ids():
    txt = (ROOT / "docs" / "REALISM_PLAN.md").read_text()
    return sorted({int(m) for m in re.findall(r"^\* U(\d+) ", txt, flags=re.M)})


def _defined_ids():
    ids = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        ids |= {int(m) for m in re.findall(r"^def test_U(\d+)[a-z]?_", f.read_text(), flags=re.M)}
    return ids


def test_plan_lists_the_expected_number_of_unit_tests():
    # 39 unit tests are promised in section 5; if someone edits the plan this must be updated on purpose.
    assert _plan_ids() == list(range(1, 40))


def test_every_planned_unit_test_exists():
    missing = [u for u in _plan_ids() if u not in _defined_ids()]
    assert not missing, f"unit tests promised in the plan but not written: U{missing}"
