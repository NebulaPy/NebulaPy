"""Readable names for integration-test output."""


def pytest_collection_modifyitems(items):
    """Replace implementation-oriented node IDs with user-facing test names."""
    for item in items:
        if item.name == "test_extracted_snapshot_dem_reference_values":
            item._nodeid = "Emission measure reference test"
