"""Authoritative verification for installed NebulaPy databases."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import re

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from NebulaPy.database.sources import (
    ATLAS_GRIDS,
    DATABASE_VERSION,
    POWR_GRIDS,
    ZENODO_DOI,
)


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
AUTHORITATIVE_MANIFEST = (
    Path(__file__).resolve().parent
    / "manifests"
    / f"database-{DATABASE_VERSION}.json"
)


class InstallerError(RuntimeError):
    """Report a database installation or verification failure."""


def _verification_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description:<32}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    )


def valid_fits(path):
    """Return whether a file starts with the standard FITS signature."""
    try:
        with path.open("rb") as handle:
            return handle.read(9) == b"SIMPLE  ="
    except OSError:
        return False


def validate_sed(content, source):
    """Validate the structure and completeness of a PoWR ASCII SED."""
    stripped = content.lstrip()
    if not stripped or stripped[:100].lower().startswith(("<!doctype", "<html")):
        raise InstallerError(f"{source} did not return an ASCII SED")
    previous = None
    rows = 0
    for line_number, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "*")):
            continue
        values = re.findall(FLOAT_RE, line)
        if len(values) != 2:
            raise InstallerError(f"{source}: invalid columns on line {line_number}")
        wavelength, flux = (
            float(value.replace("D", "E").replace("d", "e")) for value in values
        )
        if not math.isfinite(wavelength) or not math.isfinite(flux):
            raise InstallerError(f"{source}: non-finite value on line {line_number}")
        if previous is not None and wavelength <= previous:
            raise InstallerError(f"{source}: wavelength grid is not increasing")
        previous = wavelength
        rows += 1
    if rows < 10:
        raise InstallerError(f"{source}: only {rows} usable SED rows returned")
    if previous < 8.0:
        raise InstallerError(
            f"{source}: incomplete SED ends at wavelength {previous:g}"
        )


def verify_component_checksums(database, checksum_name):
    """Verify a Zenodo component using its installed SHA-256 list."""
    checksum_path = database / checksum_name
    if not checksum_path.is_file():
        raise InstallerError(f"missing component checksum file: {checksum_name}")
    verified = 0
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise InstallerError(
                f"{checksum_name}: malformed entry on line {line_number}"
            )
        relative = parts[1].strip()
        if relative.startswith("*"):
            relative = relative[1:]
        relative = relative.removeprefix("./")
        path = (database / relative).resolve()
        try:
            path.relative_to(database.resolve())
        except ValueError as error:
            raise InstallerError(
                f"{checksum_name}: unsafe path on line {line_number}"
            ) from error
        if not path.is_file():
            raise InstallerError(f"{checksum_name}: missing file {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != parts[0].lower():
            raise InstallerError(f"{checksum_name}: checksum mismatch for {relative}")
        verified += 1
    return verified


def _verify_recorded_hashes(root, hashes, label):
    verified = 0
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise InstallerError(f"{label}: unsafe recorded path {relative}") from error
        if not path.is_file():
            raise InstallerError(f"{label}: missing file {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            raise InstallerError(f"{label}: checksum mismatch for {relative}")
        verified += 1
    return verified


def _load_authoritative_manifest():
    if not AUTHORITATIVE_MANIFEST.is_file():
        raise InstallerError(
            f"authoritative manifest is missing for database {DATABASE_VERSION}"
        )
    manifest = json.loads(AUTHORITATIVE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("database_version") != DATABASE_VERSION:
        raise InstallerError("authoritative manifest version mismatch")
    if manifest.get("zenodo_doi") != ZENODO_DOI:
        raise InstallerError("authoritative manifest Zenodo DOI mismatch")
    stable = manifest.get("stable_files")
    generated = manifest.get("generated_files")
    if not isinstance(stable, dict) or not isinstance(generated, dict):
        raise InstallerError("authoritative manifest has an invalid file list")
    if manifest.get("stable_file_count") != len(stable):
        raise InstallerError("authoritative manifest stable-file count mismatch")
    if manifest.get("generated_file_count") != len(generated):
        raise InstallerError("authoritative manifest generated-file count mismatch")
    if manifest.get("required_file_count") != len(stable) + len(generated):
        raise InstallerError("authoritative manifest required-file count mismatch")
    return manifest


def _manifest_entry_selected(entry, args):
    component = entry.get("component")
    if component not in args.components:
        return False
    group = entry.get("group")
    if component == "atlas" and group:
        return group in args.atlas_grid
    if component == "powr" and group:
        selected = {
            local
            for _official, local, _display_name in (args.powr_grid or POWR_GRIDS)
        }
        return group in selected
    return True


def _validate_generated_manifest_file(database, relative, entry):
    path = database / relative
    if not path.is_file():
        raise InstallerError(f"authoritative manifest: missing file {relative}")
    if entry.get("format") == "json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise InstallerError(
                f"authoritative manifest: invalid JSON file {relative}"
            ) from error
        if not isinstance(value, dict):
            raise InstallerError(
                f"authoritative manifest: invalid JSON object {relative}"
            )


def verify_installation(database, args):
    """Verify requested components against the fixed versioned manifest."""
    print("\nDatabase verification")
    manifest = _load_authoritative_manifest()
    stable = {
        relative: entry
        for relative, entry in manifest["stable_files"].items()
        if _manifest_entry_selected(entry, args)
    }
    generated = {
        relative: entry
        for relative, entry in manifest["generated_files"].items()
        if _manifest_entry_selected(entry, args)
    }
    expected = len(stable) + len(generated)
    with _verification_progress() as progress:
        task = progress.add_task("Authoritative manifest", total=expected)
        for relative, entry in stable.items():
            path = database / relative
            if not path.is_file():
                raise InstallerError(
                    f"authoritative manifest: missing file {relative}"
                )
            if path.stat().st_size != entry["size"]:
                raise InstallerError(
                    f"authoritative manifest: size mismatch for {relative}"
                )
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual.lower() != entry["sha256"].lower():
                raise InstallerError(
                    f"authoritative manifest: checksum mismatch for {relative}"
                )
            progress.advance(task)
        for relative, entry in generated.items():
            _validate_generated_manifest_file(database, relative, entry)
            progress.advance(task)
        all_powr_groups = {local for _official, local, _display in POWR_GRIDS}
        selected_powr_groups = {
            local
            for _official, local, _display in (args.powr_grid or POWR_GRIDS)
        }
        full_database = (
            set(args.components) == {"core", "atlas", "powr"}
            and set(args.atlas_grid) == set(ATLAS_GRIDS)
            and selected_powr_groups == all_powr_groups
        )
        if full_database:
            expected_paths = set(manifest["stable_files"])
            expected_paths.update(manifest["generated_files"])
            actual_paths = {
                path.relative_to(database).as_posix()
                for path in database.rglob("*")
                if path.is_file()
                and path.name != ".DS_Store"
                and path.name != "DATABASE_INVENTORY.json"
            }
            unexpected = sorted(actual_paths - expected_paths)
            if unexpected:
                raise InstallerError(
                    "authoritative manifest: unexpected file "
                    f"{unexpected[0]}"
                )
        progress.update(task, description="Component verification complete")
    return expected


def _write_text_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_database_inventory(database):
    """Record the files in this particular database installation."""
    inventory_path = database / "DATABASE_INVENTORY.json"
    files = {}
    for path in sorted(database.rglob("*")):
        if not path.is_file() or path == inventory_path:
            continue
        relative = path.relative_to(database).as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory = {
        "database": "NebulaPy Database",
        "database_version": DATABASE_VERSION,
        "zenodo_doi": ZENODO_DOI,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "hash_algorithm": "SHA-256",
        "files": files,
    }
    _write_text_atomic(
        inventory_path,
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
    )
    return inventory_path


def verify_database_inventory(database):
    """Verify the installation against its freshly written local inventory."""
    inventory_path = database / "DATABASE_INVENTORY.json"
    if not inventory_path.is_file():
        raise InstallerError("DATABASE_INVENTORY.json is missing")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("database_version") != DATABASE_VERSION:
        raise InstallerError("database inventory version mismatch")
    hashes = inventory.get("files")
    if not isinstance(hashes, dict):
        raise InstallerError("database inventory has no file list")
    verified = _verify_recorded_hashes(database, hashes, "database inventory")
    actual_files = {
        path.relative_to(database).as_posix()
        for path in database.rglob("*")
        if path.is_file() and path != inventory_path
    }
    recorded_files = set(hashes)
    missing_from_inventory = sorted(actual_files - recorded_files)
    if missing_from_inventory:
        raise InstallerError(
            "database inventory omits installed file: "
            f"{missing_from_inventory[0]}"
        )
    if inventory.get("file_count") != verified:
        raise InstallerError("database inventory file count mismatch")
    return verified
