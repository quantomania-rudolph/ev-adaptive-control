"""Release gate: skipped tests fail the run only when deployment asks for that."""

import os

import pytest


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("EVCONTROL_FAIL_ON_SKIPPED") != "1":
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
