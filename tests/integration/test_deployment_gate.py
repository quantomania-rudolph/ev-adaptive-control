"""The release gate fails when a unit test is skipped, and passes when it runs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def _run_gate(directory: Path) -> subprocess.CompletedProcess[str]:
    shutil.copy(ROOT / "conftest.py", directory / "conftest.py")
    env = os.environ.copy()
    env["EVCONTROL_FAIL_ON_SKIPPED"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(directory),
            "--override-ini",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_passing_unit_test_clears_the_deployment_gate(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = _run_gate(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_skipped_unit_test_blocks_the_deployment_gate(tmp_path: Path) -> None:
    (tmp_path / "test_skipped.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.skip(reason='not ready')\n"
        "def test_new():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    result = _run_gate(tmp_path)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined
    assert "deployment gate" in combined
