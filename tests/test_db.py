"""Integrity and scientific-structure checks for a NebulaPy database release."""

from collections import defaultdict
import hashlib
import json
import re

import numpy as np
import pytest
from astropy.io import fits


EXPECTED_VERSION = "1.0.0"
EXPECTED_COMPONENT_COUNTS = {
    "cooling_rates": 94,
    "cie_ion_fractions": 1,
    "atlas_seds": 608,
    "cmfgen_seds": 2,
    "powr_seds": 4182,
}
REQUIRED_PATHS = (
    "README.md",
    "SOURCES.md",
    "SHA256SUMS",
    "manifest.json",
    "licenses",
    "chianti_cooling_rates",
    "cie_ion_fractions.txt",
    "sed/atlas",
    "sed/cmfgen",
    "sed/powr",
)
FLOAT_PATTERN = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
)


@pytest.fixture(scope="session")
def manifest(database_root):
    """Load the release manifest once for all validation tests."""
    manifest_path = database_root / "manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        pytest.fail(f"Unable to read a valid manifest.json: {error}")


def _files_below(path):
    return sorted(item for item in path.rglob("*") if item.is_file())


def _load_numeric_table(path):
    try:
        table = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        pytest.fail(f"Unable to read numeric table {path}: {error}")
    return np.atleast_2d(table)


def _load_two_column_sed(path):
    """Read whitespace-delimited or legacy fixed-width two-column SED data."""
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                values = FLOAT_PATTERN.findall(line)
                if len(values) != 2:
                    pytest.fail(
                        f"{path}:{line_number} does not contain two numeric values"
                    )
                rows.append((float(values[0]), float(values[1])))
    except OSError as error:
        pytest.fail(f"Unable to read SED table {path}: {error}")

    return np.asarray(rows, dtype=np.float64)


def _validate_database_layout_and_manifest(database_root, manifest):
    """Required release files, component paths, versions, and counts agree."""
    missing = [
        relative_path
        for relative_path in REQUIRED_PATHS
        if not (database_root / relative_path).exists()
    ]
    assert not missing, f"Missing required database paths: {missing}"

    assert manifest["name"] == "NebulaPy Database"
    assert manifest["version"] == EXPECTED_VERSION
    assert manifest["schema_version"] == 1
    assert manifest["checksums_file"] == "SHA256SUMS"
    assert manifest["provenance_file"] == "SOURCES.md"
    assert manifest["licensing_directory"] == "licenses"

    components = manifest["components"]
    actual_counts = {
        "cooling_rates": len(
            _files_below(database_root / components["cooling_rates"]["path"])
        ),
        "cie_ion_fractions": int(
            (database_root / components["cie_ion_fractions"]["path"]).is_file()
        ),
        "atlas_seds": len(
            _files_below(database_root / components["atlas_seds"]["path"])
        ),
        "cmfgen_seds": len(
            _files_below(database_root / components["cmfgen_seds"]["path"])
        ),
        "powr_seds": len(
            _files_below(database_root / components["powr_seds"]["path"])
        ),
    }
    assert actual_counts == EXPECTED_COMPONENT_COUNTS
    assert {
        name: component["file_count"] for name, component in components.items()
    } == EXPECTED_COMPONENT_COUNTS
    assert sum(actual_counts.values()) == manifest["scientific_data_file_count"]


def _validate_database_has_no_unsafe_or_platform_metadata(database_root):
    """The release contains regular, non-empty files with portable names."""
    empty_files = []
    symlinks = []
    platform_metadata = []
    unsafe_names = []

    for path in database_root.rglob("*"):
        relative_path = path.relative_to(database_root)
        if path.is_symlink():
            symlinks.append(str(relative_path))
        elif path.is_file() and path.stat().st_size == 0:
            empty_files.append(str(relative_path))

        if path.name in {".DS_Store", "Thumbs.db"} or path.name == "__MACOSX":
            platform_metadata.append(str(relative_path))
        if any(part in {"", ".", ".."} for part in relative_path.parts):
            unsafe_names.append(str(relative_path))
        if "\\" in str(relative_path):
            unsafe_names.append(str(relative_path))

    assert not empty_files, f"Empty files found: {empty_files}"
    assert not symlinks, f"Symbolic links found: {symlinks}"
    assert not platform_metadata, f"Platform metadata found: {platform_metadata}"
    assert not unsafe_names, f"Unsafe or non-portable paths found: {unsafe_names}"


def _validate_sha256sums_cover_and_verify_every_release_file(database_root):
    """Every release file except SHA256SUMS is listed and byte-for-byte valid."""
    checksum_path = database_root / "SHA256SUMS"
    expected_hashes = {}

    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split(maxsplit=1)
        assert len(fields) == 2, f"Malformed SHA256SUMS line {line_number}"
        expected_hash, relative_name = fields
        relative_name = relative_name.lstrip("*")
        if relative_name.startswith("./"):
            relative_name = relative_name[2:]
        assert re.fullmatch(r"[0-9a-f]{64}", expected_hash), (
            f"Invalid SHA-256 on line {line_number}"
        )
        assert relative_name not in expected_hashes, (
            f"Duplicate checksum entry: {relative_name}"
        )
        expected_hashes[relative_name] = expected_hash

    actual_files = {
        path.relative_to(database_root).as_posix()
        for path in _files_below(database_root)
        if path != checksum_path
    }
    assert set(expected_hashes) == actual_files, (
        f"Missing checksum entries: {sorted(actual_files - set(expected_hashes))}; "
        f"unexpected entries: {sorted(set(expected_hashes) - actual_files)}"
    )

    mismatches = []
    for relative_name, expected_hash in expected_hashes.items():
        digest = hashlib.sha256()
        with (database_root / relative_name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            mismatches.append(relative_name)

    assert not mismatches, f"SHA-256 mismatch: {mismatches}"


def _validate_cooling_rate_tables(database_root):
    """All ion cooling tables have the expected finite, ordered numeric grid."""
    cooling_root = database_root / "chianti_cooling_rates"
    data_files = sorted(
        path
        for path in cooling_root.glob("*.txt")
        if path.name != "comment.txt"
    )
    assert len(data_files) == 93
    assert (cooling_root / "comment.txt").is_file()

    reference_temperatures = None
    for path in data_files:
        assert re.fullmatch(r"[a-z]+_[0-9]+\.txt", path.name)
        table = _load_numeric_table(path)
        assert table.shape == (81, 14), f"Unexpected shape for {path}"
        assert np.isfinite(table).all(), f"Non-finite values in {path}"
        assert np.all(np.diff(table[:, 0]) > 0), (
            f"Temperature grid is not increasing in {path}"
        )
        if reference_temperatures is None:
            reference_temperatures = table[:, 0]
        else:
            np.testing.assert_array_equal(table[:, 0], reference_temperatures)


def _validate_cie_ion_fraction_table(database_root):
    """The CIE grid is finite, ordered, bounded, and normalized by element."""
    path = database_root / "cie_ion_fractions.txt"
    content_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    header = content_lines[0].split()
    table = np.loadtxt(content_lines[1:], dtype=np.float64)

    assert header[0] == "log_T"
    assert len(header) == 100
    assert table.shape == (1024, 100)
    assert np.isfinite(table).all()
    assert np.all(np.diff(table[:, 0]) > 0)

    fractions = table[:, 1:]
    assert np.all(fractions >= -1.0e-12)
    assert np.all(fractions <= 1.0 + 1.0e-12)

    element_columns = defaultdict(list)
    for column_index, column_name in enumerate(header[1:], start=1):
        element, separator, ion_stage = column_name.partition("_")
        assert separator and ion_stage.isdigit(), (
            f"Invalid CIE column name: {column_name}"
        )
        element_columns[element].append(column_index)

    for element, column_indices in element_columns.items():
        np.testing.assert_allclose(
            table[:, column_indices].sum(axis=1),
            1.0,
            rtol=0.0,
            atol=2.0e-9,
            err_msg=f"CIE fractions are not normalized for {element}",
        )


def _validate_atlas_fits_files(database_root):
    """Every ATLAS file is an intact FITS spectrum with readable HDUs."""
    atlas_files = _files_below(database_root / "sed" / "atlas")
    assert len(atlas_files) == 608
    assert all(path.suffix.lower() == ".fits" for path in atlas_files)

    invalid_files = []
    for path in atlas_files:
        try:
            with fits.open(path, mode="readonly", memmap=False) as hdus:
                hdus.verify("exception")
                if len(hdus) != 2 or hdus[1].data is None:
                    invalid_files.append(str(path))
        except (OSError, ValueError, IndexError) as error:
            invalid_files.append(f"{path}: {error}")

    assert not invalid_files, f"Invalid ATLAS FITS files: {invalid_files}"


def _validate_stellar_sed_text_files(database_root):
    """PoWR and CMFGEN spectra are finite two-column monotonic tables."""
    powr_root = database_root / "sed" / "powr"
    cmfgen_root = database_root / "sed" / "cmfgen"
    powr_seds = sorted(powr_root.rglob("*_sed.txt"))
    powr_parameters = sorted(powr_root.rglob("modelparameters.txt"))
    cmfgen_seds = sorted(cmfgen_root.rglob("*_sed.txt"))
    cmfgen_parameters = sorted(cmfgen_root.rglob("modelparameters.txt"))

    assert len(powr_seds) == 4158
    assert len(powr_parameters) == 24
    assert len(cmfgen_seds) == 1
    assert len(cmfgen_parameters) == 1

    invalid_tables = []
    for path in powr_seds + cmfgen_seds:
        table = _load_two_column_sed(path)
        if table.shape[1] != 2 or table.shape[0] < 2:
            invalid_tables.append(f"{path}: shape {table.shape}")
            continue
        if not np.isfinite(table).all():
            invalid_tables.append(f"{path}: non-finite values")
            continue
        coordinate_steps = np.diff(table[:, 0])
        if not (np.all(coordinate_steps > 0) or np.all(coordinate_steps < 0)):
            invalid_tables.append(f"{path}: first column is not monotonic")

    assert not invalid_tables, f"Invalid stellar SED tables: {invalid_tables}"

    for path in powr_parameters + cmfgen_parameters:
        assert path.stat().st_size > 0, f"Empty model-parameter file: {path}"


def test_database_release_summary(database_root, manifest, record_property):
    """Validate and report the database as one comprehensive pytest test."""
    _validate_database_layout_and_manifest(database_root, manifest)
    _validate_database_has_no_unsafe_or_platform_metadata(database_root)
    _validate_sha256sums_cover_and_verify_every_release_file(database_root)
    _validate_cooling_rate_tables(database_root)
    _validate_cie_ion_fraction_table(database_root)
    _validate_atlas_fits_files(database_root)
    _validate_stellar_sed_text_files(database_root)

    checksum_entries = sum(
        bool(line.strip())
        for line in (database_root / "SHA256SUMS")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    scientific_files = manifest["scientific_data_file_count"]

    assert checksum_entries == 4896
    assert scientific_files == 4887

    record_property(
        "test_summary",
        f"  Database   : {manifest['name']} {manifest['version']}"
        f"\n  Location   : {database_root}"
        f"\n  Data files : {scientific_files}"
        f"\n  Checksums  : {checksum_entries} SHA-256 entries"
        "\n  Checks     : layout, manifest, integrity, tables, FITS, SEDs"
        "\n  Components : cooling rates, CIE fractions, ATLAS, CMFGEN, PoWR",
    )
