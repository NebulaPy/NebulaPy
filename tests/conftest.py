"""Readable names for integration-test output."""


_PASSED_TEST_SUMMARIES = {}


def pytest_collection_modifyitems(items):
    """Replace implementation-oriented node IDs with user-facing test names."""
    readable_test_names = {
        "test_cie_end_to_end": "CIE ion-balance reference test",
        "test_extracted_snapshot_dem_reference_values": (
            "Emission measure reference test"
        ),
    }

    for item in items:
        readable_name = readable_test_names.get(item.name)
        if readable_name is not None:
            item._nodeid = readable_name


def pytest_runtest_logfinish(nodeid, location):
    """Print scientific context after the test result and separate reports."""
    summary = _PASSED_TEST_SUMMARIES.pop(nodeid, None)
    if summary is not None:
        print(f"\n{summary}")
    print()


def pytest_runtest_logreport(report):
    """Retain summaries attached to successful test calls."""
    if report.when != "call" or not report.passed:
        return

    for property_name, property_value in report.user_properties:
        if property_name == "test_summary":
            _PASSED_TEST_SUMMARIES[report.nodeid] = property_value
