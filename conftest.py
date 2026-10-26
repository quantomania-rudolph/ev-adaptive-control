"""Pytest options shared by local runs and the release workflow."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--fail-on-skipped",
        action="store_true",
        default=False,
        help=(
            "Fail if any collected test is skipped. The release workflow uses this "
            "so unit tests must pass before deployment."
        ),
    )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("--fail-on-skipped"):
        return
    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    skipped = []
    if reporter is not None:
        skipped = reporter.stats.get("skipped", [])
    if not skipped:
        return
    if reporter is not None:
        reporter.write_sep("=", "deployment gate: skipped unit tests do not pass")
        for report in skipped:
            reporter.write_line(f"SKIPPED {report.nodeid}")
    if int(session.exitstatus or exitstatus) == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
