"""Readable names for integration-test output."""

import os
from pathlib import Path

import pytest


_PASSED_TEST_SUMMARIES = {}
_TERMINAL_REPORTER = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "nebulapy-db-1.0.0"


@pytest.fixture(scope="session", autouse=True)
def database_environment():
    """Resolve one database root and expose it to every test in the session."""
    original_value = os.environ.get("NEBULAPY_DB")
    database_root = Path(original_value).expanduser() if original_value else DEFAULT_DATABASE
    database_root = database_root.resolve()

    if not database_root.is_dir():
        pytest.fail(
            f"NebulaPy database directory does not exist: {database_root}\n"
            "Set NEBULAPY_DB to the extracted database directory."
        )

    os.environ["NEBULAPY_DB"] = str(database_root)
    yield database_root

    if original_value is None:
        os.environ.pop("NEBULAPY_DB", None)
    else:
        os.environ["NEBULAPY_DB"] = original_value


@pytest.fixture(scope="session")
def database_root(database_environment):
    """Provide the session's validated database root."""
    return database_environment


def pytest_sessionstart(session):
    """Retain pytest's initialized terminal reporter for coloured output."""
    global _TERMINAL_REPORTER
    _TERMINAL_REPORTER = session.config.pluginmanager.getplugin("terminalreporter")


def pytest_collection_modifyitems(items):
    """Replace implementation-oriented node IDs with user-facing test names."""
    test_order = {
        "test_database_release_summary": 0,
        "test_cie": 1,
        "test_sed": 2,
        "test_em": 3,
    }
    readable_test_names = {
        "test_cie": "CIE ion-balance reference test",
        "test_database_release_summary": "NebulaPy database validity test",
        "test_em": "Emission measure reference test",
        "test_sed": "SED reference test",
    }

    items.sort(key=lambda item: test_order.get(item.name, len(test_order)))
    for item in items:
        readable_name = readable_test_names.get(item.name)
        if readable_name is not None:
            item._nodeid = readable_name


def pytest_runtest_logfinish(nodeid, location):
    """Print scientific context after the test result and separate reports."""
    result = _PASSED_TEST_SUMMARIES.pop(nodeid, None)
    if result is not None:
        summary, duration = result
        if _TERMINAL_REPORTER is not None:
            _TERMINAL_REPORTER.write(f"\n{nodeid} ")
            _TERMINAL_REPORTER.write("PASSED", green=True, bold=True)
            _TERMINAL_REPORTER.write(f" [{duration:.2f} s]\n{summary}\n")
        else:
            print(
                f"\n{nodeid} PASSED"
                f" [{duration:.2f} s]"
                f"\n{summary}"
            )
    print()


def pytest_runtest_logreport(report):
    """Retain summaries attached to successful test calls."""
    if report.when != "call" or not report.passed:
        return

    for property_name, property_value in report.user_properties:
        if property_name == "test_summary":
            _PASSED_TEST_SUMMARIES[report.nodeid] = (
                property_value,
                report.duration,
            )


def pytest_report_teststatus(report, config):
    """Suppress the redundant green progress dot before passed summaries."""
    if report.when == "call" and report.passed:
        return "passed", "", "PASSED"
    return None
