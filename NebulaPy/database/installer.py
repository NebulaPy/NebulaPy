"""Install the versioned NebulaPy auxiliary database."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urljoin
import zipfile

import requests
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from NebulaPy.database.sources import (
    ATLAS_BASE_URL,
    ATLAS_GRIDS,
    DATABASE_NAME,
    DATABASE_VERSION,
    POWR_BASE_URL,
    POWR_DOWNLOAD_URL,
    POWR_GRID_PAGE,
    POWR_GRIDS,
    ZENODO_BASE_URL,
    ZENODO_COMPONENTS,
    ZENODO_DOI,
)
from NebulaPy.database.verification import (
    InstallerError,
    valid_fits as _valid_fits,
    validate_sed as _validate_sed,
    verify_component_checksums as _verify_component_checksums,
    verify_database_inventory,
    verify_installation,
    write_database_inventory,
)


USER_AGENT = "NebulaPy-database-downloader/1.0 (+https://nebulapy.github.io/NebulaPy/)"
MODEL_RE = re.compile(r'id=["\']modlink([^"\']+)["\']')
FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_THREAD_STATE = threading.local()
_POWR_SED_LOCK = threading.Lock()


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def _progress():
    return Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=20), MofNCompleteColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TextColumn("ETA"), TimeRemainingColumn(),
    )


def _powr_progress():
    """Use fixed label widths so PoWR progress columns remain aligned."""
    return Progress(
        SpinnerColumn(),
        TextColumn(
            "[bold blue]{task.fields[grid]:<20}[/] "
            "{task.fields[model]:>7}"
        ),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    )


def _status_line(name, current, expected=None, status="Up to date"):
    count = str(current) if expected is None else f"{current}/{expected}"
    print(f"  {name:<24} files: {count:>7}  [{status}]")


def _dataset_status(current, expected, *, dry_run=False, force=False):
    if force:
        return "Refreshing"
    if current >= expected:
        return "Up to date"
    if current == 0:
        return "Not installed" if dry_run else "Downloading"
    return f"{expected - current} missing"


def _worker_session():
    session = getattr(_THREAD_STATE, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        _THREAD_STATE.session = session
    return session


def _database_path(destination):
    if destination:
        path = Path(destination).expanduser().resolve() / DATABASE_NAME
    else:
        configured = os.environ.get("NEBULAPY_DB")
        if configured:
            path = Path(configured).expanduser().resolve()
        else:
            path = (Path.home() / ".nebulapy" / DATABASE_NAME).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _request(session, url, timeout, retries, stream=False):
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout, stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise InstallerError(f"request failed: {url}") from last_error


def _download_file(session, url, destination, args):
    response = _request(
        session, url, max(args.timeout, 120.0), args.retries, stream=True
    )
    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
                    digest.update(block)
        os.replace(temporary, destination)
    finally:
        response.close()
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def _safe_extract_zip(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(destination)
            except ValueError as error:
                raise InstallerError(
                    f"unsafe path in {archive.name}: {member.filename}"
                ) from error
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise InstallerError(
                    f"symbolic link not allowed in {archive.name}: {member.filename}"
                )
        package.extractall(destination)


def _merge_component(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.iterdir():
        target_path = destination / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)


def download_core(database, args):
    """Install the versioned NebulaPy-produced components from Zenodo."""
    print(f"\nNebulaPy core database (Zenodo DOI: {ZENODO_DOI})")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    for component in ZENODO_COMPONENTS:
        checksum_name = component["checksums"]
        expected = component["data_files"]
        if component["name"] == "chianti_cooling_rates":
            current = sum(
                1
                for path in (database / "chianti_cooling_rates").glob("*")
                if path.is_file()
            )
        else:
            current = int((database / "cie_ion_fractions.txt").is_file())
        if not args.force:
            try:
                _verify_component_checksums(database, checksum_name)
                _status_line(component["label"], expected)
                continue
            except (InstallerError, OSError):
                pass

        if args.dry_run:
            status = _dataset_status(
                current, expected, dry_run=True, force=args.force
            )
            _status_line(component["label"], current, expected, status)
            continue
        with _progress() as progress:
            task = progress.add_task(
                f"{component['label']}: downloading", total=expected
            )
            with tempfile.TemporaryDirectory(
                prefix="nebulapy-core-"
            ) as temporary:
                temporary_path = Path(temporary)
                archive = temporary_path / component["filename"]
                url = f"{ZENODO_BASE_URL}/{component['filename']}/content"
                digest = _download_file(session, url, archive, args)
                if digest.lower() != component["sha256"].lower():
                    raise InstallerError(
                        f"Zenodo checksum mismatch for {component['filename']}"
                    )
                extracted = temporary_path / "extracted"
                extracted.mkdir()
                _safe_extract_zip(archive, extracted)
                staged_database = extracted / DATABASE_NAME
                if not staged_database.is_dir():
                    raise InstallerError(
                        f"{component['filename']} lacks {DATABASE_NAME}/"
                    )
                version_path = staged_database / "VERSION"
                if (
                    not version_path.is_file()
                    or version_path.read_text(encoding="utf-8").strip()
                    != DATABASE_VERSION
                ):
                    raise InstallerError(
                        f"{component['filename']} is not database version "
                        f"{DATABASE_VERSION}"
                    )
                _verify_component_checksums(staged_database, checksum_name)
                _merge_component(staged_database, database)
                _verify_component_checksums(database, checksum_name)
            progress.update(
                task,
                description=f"{component['label']}: complete",
                completed=expected,
            )


def download_atlas(database, args):
    target = database / "sed" / "atlas"
    target.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    provenance = {
        "source": ATLAS_BASE_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {},
    }
    print("\nATLAS stellar-atmosphere grids")
    for grid in args.atlas_grid:
        grid_url = urljoin(ATLAS_BASE_URL, f"{grid}/")
        # STScI directory indexes can be substantially slower than the small
        # FITS files themselves, especially when the archive is under load.
        listing = _request(
            session,
            grid_url,
            max(args.timeout, 120.0),
            args.retries,
        )
        parser = _LinkParser()
        parser.feed(listing.text)
        filenames = sorted({
            Path(link).name for link in parser.links
            if re.fullmatch(rf"{re.escape(grid)}_\d+\.fits", Path(link).name,
                            flags=re.IGNORECASE)
        })
        if not filenames:
            raise InstallerError(f"no FITS models found for ATLAS grid {grid}")
        grid_target = target / grid
        current = sum(
            1
            for filename in filenames
            if _valid_fits(grid_target / filename)
        )
        expected = len(filenames)
        if current == expected and not args.force:
            for filename in filenames:
                path = grid_target / filename
                provenance["files_sha256"][f"{grid}/{filename}"] = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
            _status_line(grid, expected)
            continue
        if args.dry_run:
            status = _dataset_status(
                current, expected, dry_run=True, force=args.force
            )
            _status_line(grid, current, expected, status)
            continue
        grid_target.mkdir(parents=True, exist_ok=True)
        def fetch_atlas(filename):
            destination = grid_target / filename
            if destination.exists() and _valid_fits(destination) and not args.force:
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            else:
                digest = _download_file(
                    _worker_session(), urljoin(grid_url, filename), destination, args
                )
                if not _valid_fits(destination):
                    destination.unlink(missing_ok=True)
                    raise InstallerError(f"invalid FITS file: {grid}/{filename}")
            if args.delay:
                time.sleep(args.delay)
            return filename, digest

        with _progress() as progress, ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            task = progress.add_task(f"{grid}: starting", total=len(filenames))
            futures = {
                executor.submit(fetch_atlas, filename): filename
                for filename in filenames
            }
            retry_filenames = []
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    filename, digest = future.result()
                except (InstallerError, OSError):
                    retry_filenames.append(filename)
                    continue
                progress.update(task, description=f"{grid}: {filename}")
                provenance["files_sha256"][f"{grid}/{filename}"] = digest
                progress.advance(task)

            # A busy archive may reject bursts even though each file is valid.
            # Retry only those failures sequentially after the parallel batch.
            for filename in retry_filenames:
                progress.update(task, description=f"{grid}: retry {filename}")
                filename, digest = fetch_atlas(filename)
                provenance["files_sha256"][f"{grid}/{filename}"] = digest
                progress.advance(task)
            progress.update(task, description=f"{grid}: complete")
    if not args.dry_run:
        (target / "download-provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _powr_request(session, method, url, args, **kwargs):
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            response = session.request(method, url, timeout=args.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt < args.retries:
                time.sleep(2**attempt)
    raise InstallerError(f"PoWR request failed: {url}") from last_error


def _model_parameters(content, grid, model):
    patterns = {
        "temperature": rf"\bTEFF\s*=\s*({FLOAT_RE})\s*K",
        "rtrans": rf"\bRTRANS\s*=\s*({FLOAT_RE})",
        "mass": rf"(?:IMPLIED STELLAR MASS|\bMASS|\bMSTAR)\s*=\s*({FLOAT_RE})",
        "log_g": rf"\bLOG G_GRAV\s*=\s*({FLOAT_RE})",
        "log_l": rf"\bLOG L\s*=\s*({FLOAT_RE})",
        "log_mdot": rf"\bM-DOT\s*=\s*({FLOAT_RE})",
        "v_inf": rf"\bVFINAL\s*=\s*({FLOAT_RE})",
    }
    values = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match is None:
            raise InstallerError(f"{grid}/{model} model information lacks {field}")
        values[field] = float(match.group(1).replace("D", "E").replace("d", "e"))
    return values


def _write_text_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _parameter_table(grid, records):
    lines = [
        f"Parameter summary downloaded from the official PoWR {grid} grid",
        "=" * 72, "", "", "",
        "   MODEL       T_EFF         R_TRANS     MASS     LOG G    LOG L"
        "   LOG MDOT    V_INF",
        "   NAME         [K]          [R_SUN]   [M_SUN]    [CGS]   [L_SUN]"
        " [M_SUN/YR]  [KM/S]",
        "",
    ]
    for model, value in records:
        lines.append(
            f"{model:>8} {value['temperature']:13.2f} "
            f"{value['rtrans']:13.4f} {value['mass']:8.1f} "
            f"{value['log_g']:9.3f} {value['log_l']:8.2f} "
            f"{value['log_mdot']:9.2f} {value['v_inf']:9.0f}"
        )
    return "\n".join(lines) + "\n"


def _download_powr_grid(database, official, local, display_name, args):
    target = database / "sed" / "powr" / f"{local}-sed"
    parameter_table = target / "modelparameters.txt"
    provenance_path = target / "download-provenance.json"
    if not args.force and parameter_table.is_file():
        try:
            parameter_lines = parameter_table.read_text(encoding="utf-8").splitlines()
            models = [
                line.split()[0]
                for line in parameter_lines[8:]
                if line.strip()
            ]
            if not models:
                raise InstallerError(f"{display_name}: empty model parameter table")
            hashes = {}
            for model in models:
                relative = f"{local}_{model}_sed.txt"
                sed_path = target / relative
                if not sed_path.is_file():
                    raise InstallerError(f"{display_name}: missing SED {relative}")
                _validate_sed(
                    sed_path.read_text(encoding="utf-8"),
                    f"{display_name}/{relative}",
                )
                hashes[relative] = hashlib.sha256(sed_path.read_bytes()).hexdigest()
            _write_text_atomic(
                provenance_path,
                json.dumps({
                    "source": POWR_BASE_URL,
                    "grid": official,
                    "local_grid_name": local,
                    "validated_utc": datetime.now(timezone.utc).isoformat(),
                    "files_sha256": hashes,
                }, indent=2, sort_keys=True) + "\n",
            )
            _status_line(display_name, len(hashes))
            return
        except (
            InstallerError,
            OSError,
            UnicodeDecodeError,
        ):
            pass

    session = _worker_session()
    listing = _powr_request(
        session, "POST", POWR_GRID_PAGE, args, data={"grid": official}
    )
    models = sorted(set(MODEL_RE.findall(listing.text)))
    if not models:
        raise InstallerError(f"no models found for PoWR grid {official}")
    expected = len(models)
    current = 0
    if not args.force:
        for model in models:
            sed_path = target / f"{local}_{model}_sed.txt"
            if not sed_path.is_file():
                continue
            try:
                _validate_sed(
                    sed_path.read_text(encoding="utf-8"),
                    f"{display_name}/{model}",
                )
                current += 1
            except (InstallerError, UnicodeDecodeError):
                pass
    if args.dry_run:
        status = _dataset_status(
            current, expected, dry_run=True, force=args.force
        )
        _status_line(display_name, current, expected, status)
        return
    target.mkdir(parents=True, exist_ok=True)
    records = {}
    hashes = {}

    def get_model_parameters(worker_session, model):
        last_error = None
        for attempt in range(args.retries + 1):
            info = _powr_request(
                worker_session, "GET", POWR_DOWNLOAD_URL, args,
                params={"content": "modinfo", "grid": official, "model": model},
            )
            try:
                return _model_parameters(info.text, official, model)
            except InstallerError as error:
                last_error = error
                if attempt < args.retries:
                    time.sleep(2**attempt)
        raise last_error

    def get_sed(worker_session, model):
        last_error = None
        for attempt in range(args.retries + 1):
            with _POWR_SED_LOCK:
                sed = _powr_request(
                    worker_session, "POST", POWR_DOWNLOAD_URL, args,
                    data={
                        "content": "specdl", "grid": official, "model": model,
                        "datatype": "sed", "dlformat": "txt",
                        "specdownload": "Download",
                    },
                )
            try:
                _validate_sed(sed.text, f"{official}/{model}")
                return sed.text
            except InstallerError as error:
                last_error = error
                if attempt < args.retries:
                    time.sleep(2**attempt)
        raise last_error

    def fetch_powr(model):
        worker_session = _worker_session()
        filename = f"{local}_{model}_sed.txt"
        destination = target / filename
        parameters = get_model_parameters(worker_session, model)
        existing_is_valid = False
        if destination.exists() and not args.force:
            try:
                _validate_sed(destination.read_text(encoding="utf-8"), destination)
                existing_is_valid = True
            except (InstallerError, UnicodeDecodeError):
                existing_is_valid = False
        if not existing_is_valid:
            _write_text_atomic(destination, get_sed(worker_session, model))
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if args.delay:
            time.sleep(args.delay)
        return model, parameters, filename, digest

    powr_workers = min(args.workers, 4)
    with _powr_progress() as progress, ThreadPoolExecutor(
        max_workers=powr_workers
    ) as executor:
        task = progress.add_task(
            "",
            total=len(models),
            grid=display_name,
            model="starting",
        )
        futures = {
            executor.submit(fetch_powr, model): model
            for model in models
        }
        for future in as_completed(futures):
            model, parameters, filename, digest = future.result()
            progress.update(task, model=model)
            records[model] = parameters
            hashes[filename] = digest
            progress.advance(task)
        progress.update(task, model="complete")
    ordered_records = [(model, records[model]) for model in models]
    _write_text_atomic(
        target / "modelparameters.txt", _parameter_table(official, ordered_records)
    )
    _write_text_atomic(
        target / "download-provenance.json",
        json.dumps({
            "source": POWR_BASE_URL,
            "grid": official,
            "local_grid_name": local,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "files_sha256": hashes,
        }, indent=2, sort_keys=True) + "\n",
    )


def _parse_powr_grid(value):
    for official, local, display_name in POWR_GRIDS:
        if value == official:
            return official, local, display_name
    supported = ", ".join(official for official, _local, _display in POWR_GRIDS)
    raise argparse.ArgumentTypeError(
        f"unsupported PoWR grid {value!r}; choose one of: {supported}"
    )


def download_powr_grids(database, args):
    print("\nPoWR stellar-atmosphere grids")
    for official, local, display_name in args.powr_grid or POWR_GRIDS:
        _download_powr_grid(database, official, local, display_name, args)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nebulapy database install",
        description=(
            "Install the NebulaPy core database from Zenodo and supported "
            "stellar-atmosphere data from their official providers."
        ),
    )
    parser.add_argument(
        "--destination",
        help=(
            "parent installation directory; the versioned database directory "
            f"{DATABASE_NAME!r} is created inside it"
        ),
    )
    parser.add_argument(
        "--component", "--components",
        dest="components",
        nargs="+",
        choices=("core", "atlas", "powr"),
        default=("core", "atlas", "powr"),
        help="components to install (default: core atlas powr)",
    )
    parser.add_argument("--atlas-grid", action="append", choices=ATLAS_GRIDS)
    parser.add_argument(
        "--powr-grid",
        action="append",
        type=_parse_powr_grid,
        metavar="GRID",
        help=(
            "official PoWR grid identifier, such as OB-I, wc, or LMC-OB-I; "
            "repeat for multiple grids"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="simultaneous downloads (default: 8)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="optional delay after each download (default: 0)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1 or args.delay < 0 or args.timeout <= 0 or args.retries < 0:
        parser.error("invalid delay, timeout, or retry value")
    args.atlas_grid = args.atlas_grid or list(ATLAS_GRIDS)
    try:
        database = _database_path(args.destination)
        print(f"NebulaPy database {DATABASE_VERSION}: {database}")
        if "core" in args.components:
            download_core(database, args)
        if "atlas" in args.components:
            download_atlas(database, args)
        if "powr" in args.components:
            download_powr_grids(database, args)
        if not args.dry_run:
            verify_installation(database, args)
            write_database_inventory(database)
            verify_database_inventory(database)
            console = Console()
            console.print(
                f"\n  NebulaPy Database {DATABASE_VERSION} verification "
                "[bold green]PASSED[/bold green]"
            )
        print("\nDry run complete." if args.dry_run else "\nDownload complete.")
        print(f"Set NEBULAPY_DB={database}")
        return 0
    except (InstallerError, OSError) as error:
        print(f"download-database: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
