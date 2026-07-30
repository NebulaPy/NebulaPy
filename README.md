# NebulaPy

[![PyPI](https://img.shields.io/pypi/v/NebulaPy.svg)](https://pypi.org/project/NebulaPy/)
[![Python](https://img.shields.io/pypi/pyversions/NebulaPy.svg)](https://pypi.org/project/NebulaPy/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](#project-status)

**NebulaPy turns PION radiation-hydrodynamic simulations into synthetic
observables.** It combines simulation data with atomic physics from CHIANTI to
calculate spectra, emission-line luminosities, cooling diagnostics, and
spatially resolved emission maps.

NebulaPy is intended for research in computational astrophysics, including
stellar-wind bubbles, bow shocks, photoionized nebulae, supernova environments,
and colliding-wind binaries.

## Project status

NebulaPy 2.0.0.beta is under active development. Scientific workflows should
record the NebulaPy, ChiantiPy, CHIANTI database, and PION simulation versions
used for reproducibility.

## Scientific capabilities

- Read uniform- and nested-grid PION Silo snapshots in spherical or cylindrical
  geometry.
- Group simulation files into time-ordered snapshots using seconds, minutes,
  hours, days, years, kyr, Myr, or Gyr.
- Calculate electron densities, ionic number densities, differential emission
  measures, and collision-ionization-equilibrium fractions.
- Generate X-ray and EUV spectra with bremsstrahlung, free-bound, line, and
  two-photon emission.
- Calculate individual and multi-ion line luminosities.
- Produce two-dimensional emissivity and cooling maps.
- Identify the strongest spectral transitions in a simulation snapshot.
- Bin ATLAS, PoWR, CMFGEN, and blackbody spectral-energy distributions into
  PION-compatible energy bins.
- Run expensive spectral calculations with multiprocessing

## Requirements

NebulaPy is a Python package, but a complete scientific installation also needs:

1. The Python dependencies installed by `pip`, including ChiantiPy, Astropy,
   NumPy, SciPy, PyNeb, pypion, and Rich.
2. A local CHIANTI atomic database identified by `XUVTOP`.
3. The native Silo library and its Python extension for reading PION output.
4. The NebulaPy auxiliary database identified by `NEBULAPY_DB` for stellar
   atmosphere models, CIE data, and cooling tables.

The native Python interpreter and `Silo.so` must have the same architecture. For
example, an Apple-silicon Python process requires an arm64 Silo extension; an
x86_64 extension cannot be loaded into that process.

## Installation

Choose the instructions for the operating system on which NebulaPy will run.
Do not copy a virtual environment or compiled `Silo.so` from another computer
or architecture.

### Linux installation (Ubuntu/Debian)

#### 1. Install system build tools

```bash
sudo apt update
sudo apt install build-essential gfortran cmake curl wget \
    libhdf5-dev libreadline-dev python3 python3-dev python3-venv
```

#### 2. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install NebulaPy
```

For a local source checkout, replace the last command with:

```bash
python -m pip install -e .
```

#### 3. Build the Linux Silo extension

Run the build in a disposable directory because the installer downloads and
extracts the Silo source in the current directory:

```bash
mkdir -p "$HOME/.cache/nebulapy-silo-build"
cd "$HOME/.cache/nebulapy-silo-build"
install-silo
```

Register the installed extension with the active virtual environment:

```bash
python -c 'import pathlib, site; p = pathlib.Path(site.getsitepackages()[0]) / "silo_local.pth"; p.write_text(str(pathlib.Path.home() / ".local/silo/lib") + "\n"); print(p)'
```

If the dynamic loader cannot locate `libsilo.so`, add its directory to the
Linux library path:

```bash
export LD_LIBRARY_PATH="$HOME/.local/silo/lib:${LD_LIBRARY_PATH:-}"
```

#### 4. Verify the Linux installation

```bash
python -c "import platform, Silo; import NebulaPy.src as nebula; print('Python:', platform.machine()); print('Silo:', Silo.__file__); print('NebulaPy:', nebula.__version__)"
file "$HOME/.local/silo/lib/Silo.so"
```

The Python architecture and the architecture reported by `file` must agree.

### macOS installation

These steps apply to both Apple silicon and Intel Macs. All commands must be
run from the same native terminal architecture.

#### 1. Install Apple and Homebrew build tools

Install the Apple command-line tools if they are not already present:

```bash
xcode-select --install
```

Install [Homebrew](https://brew.sh/) and then install the required formulae:

```bash
eval "$(brew shellenv)"
brew install python3 cmake gcc hdf5 wget
brew --prefix
```

The reported prefix should be `/opt/homebrew` on Apple silicon or `/usr/local`
on an Intel Mac. A `/usr/local` Homebrew selected on Apple silicon is commonly
an old Rosetta installation and should not be used to build the native Silo
extension.

#### 2. Confirm the Mac architecture

```bash
uname -m
command -v python3
python3 -c "import platform; print(platform.machine())"
```

Both commands should report `arm64` on Apple silicon or `x86_64` on an Intel
Mac. If they differ, stop and select a native Homebrew/Python installation
before building Silo.

#### 3. Create a native Mac virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install NebulaPy
```

For a local source checkout, replace the last command with:

```bash
python -m pip install -e .
```

Confirm that the environment is still native before continuing:

```bash
python -c "import platform, sys; print(sys.executable); print(platform.machine())"
```

#### 4. Build the native Mac Silo extension

```bash
mkdir -p "$HOME/.cache/nebulapy-silo-build"
cd "$HOME/.cache/nebulapy-silo-build"
install-silo
```

Register the extension with this virtual environment:

```bash
python -c 'import pathlib, site; p = pathlib.Path(site.getsitepackages()[0]) / "silo_local.pth"; p.write_text(str(pathlib.Path.home() / ".local/silo/lib") + "\n"); print(p)'
```

#### 5. Verify the macOS installation

```bash
python -c "import platform, Silo; import NebulaPy.src as nebula; print('Python:', platform.machine()); print('Silo:', Silo.__file__); print('NebulaPy:', nebula.__version__)"
file "$HOME/.local/silo/lib/Silo.so"
```

On Apple silicon, `Silo.so` must be reported as an `arm64` Mach-O bundle. On an
Intel Mac it must be `x86_64`. An architecture mismatch requires rebuilding
Silo from the correctly activated environment; downloading CHIANTI again will
not fix it.

### Database setup (Linux and macOS)

#### 1. Install the CHIANTI database

Download a database version compatible with your ChiantiPy release from the
[official CHIANTI download page](https://www.chiantidatabase.org/chianti_download.html),
extract it, and point `XUVTOP` to the database root:

```bash
export XUVTOP="/path/to/chianti/database"
```

#### 2. Install the NebulaPy database

Install the complete NebulaPy database with the command-line interface:

```bash
nebulapy database install --destination "$HOME/.nebulapy"
```

The destination is the parent directory. The command above therefore creates:

```text
$HOME/.nebulapy/nebulapy-db-1.0.0
```

The installer downloads the Zenodo, ATLAS, and PoWR components, retains valid
existing files, and verifies the completed installation against the
authoritative Database 1.0.0 manifest.

If `--destination` is omitted, the installer uses `NEBULAPY_DB` when it is
already set; otherwise it installs into
`$HOME/.nebulapy/nebulapy-db-1.0.0`.

Display the complete command-line help with:

```bash
nebulapy database install --help
```

| Option | Description |
|---|---|
| `--destination PATH` | Parent directory in which the versioned database directory is created. |
| `--component {core,atlas,powr} [...]` | Install one or more components. The default is all three. |
| `--atlas-grid GRID` | Install one ATLAS grid. Repeat the option to select multiple grids. |
| `--powr-grid GRID` | Install one supported official PoWR grid, such as `OB-I`, `wc`, or `LMC-OB-I`. Repeat for multiple grids. |
| `--force` | Download the selected components again, including valid existing files. |
| `--dry-run` | Report installed and missing components without downloading files. |
| `--workers NUMBER` | Set the number of simultaneous downloads. The default is `8`. |
| `--delay SECONDS` | Wait after each download. The default is `0`. |
| `--timeout SECONDS` | Set the request timeout. The default is `60`. |
| `--retries NUMBER` | Set the number of retries after a failed request. The default is `3`. |
| `-h`, `--help` | Display the installer help. |

The supported ATLAS grids are:

```text
ckm05  ckm10  ckm15  ckm20  ckm25  ckp00  ckp02  ckp05
```

The supported PoWR grid identifiers are:

```text
OB-I                 wc                    wne
wnl                  wnl-h50               LMC-OB-I
lmc-wc               lmc-wne               lmc-wnl-h20
lmc-wnl-h40          SMC-OB-I              SMC-OB-II
SMC-OB-III           SMC-OB-Vd3            smc-wc-2021
smc-wne              smc-wnl-h20           smc-wnl-h40
smc-wnl-h60          007-wc-2021           007-wne-2015
007-wnl-h20-2015     007-wnl-h40-2015      007-wnl-h60-2015
```

Examples:

```bash
# Install only the Zenodo files
nebulapy database install \
  --destination "$HOME/.nebulapy" \
  --component core

# Install two ATLAS grids
nebulapy database install \
  --destination "$HOME/.nebulapy" \
  --component atlas \
  --atlas-grid ckp00 \
  --atlas-grid ckp05

# Install one PoWR grid
nebulapy database install \
  --destination "$HOME/.nebulapy" \
  --component powr \
  --powr-grid OB-I

# Inspect the installation without downloading
nebulapy database install \
  --destination "$HOME/.nebulapy" \
  --dry-run

# Refresh all database components
nebulapy database install \
  --destination "$HOME/.nebulapy" \
  --force
```

Set `NEBULAPY_DB` to the versioned database directory:

```bash
export NEBULAPY_DB="$HOME/.nebulapy/nebulapy-db-1.0.0"
```

Persist `XUVTOP` and `NEBULAPY_DB` in the appropriate shell configuration:

```bash
# Linux
echo 'export XUVTOP="/path/to/chianti/database"' >> "$HOME/.bashrc"
echo 'export NEBULAPY_DB="$HOME/.nebulapy/nebulapy-db-1.0.0"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"

# macOS
echo 'export XUVTOP="/path/to/chianti/database"' >> "$HOME/.zshrc"
echo 'export NEBULAPY_DB="$HOME/.nebulapy/nebulapy-db-1.0.0"' >> "$HOME/.zshrc"
source "$HOME/.zshrc"
```

Run the final environment check:

```bash
python3 -c "import os, Silo, NebulaPy.src; print('XUVTOP:', os.environ['XUVTOP']); print('NEBULAPY_DB:', os.environ['NEBULAPY_DB']); print('Silo:', Silo.__file__)"
```

### Common installation errors

| Error | Cause | Resolution |
|---|---|---|
| `No module named 'Silo'` | The virtual environment cannot see the native extension. | Create `silo_local.pth` in that environment as shown above. |
| `incompatible architecture` | Python and `Silo.so` were built for different CPU architectures. | Recreate the environment and rebuild Silo natively. |
| `Library not loaded: libsilo` or `cannot open shared object file` | The dynamic loader cannot find the Silo shared library. | Add `$HOME/.local/silo/lib` to `DYLD_LIBRARY_PATH` on macOS or `LD_LIBRARY_PATH` on Linux. |
| Child processes restart the complete script | A multiprocessing entry point is unguarded. | Put executable code inside `main()` and use the `if __name__ == "__main__":` guard. |

## Quick start

The following example groups PION files into simulation snapshots and loads the
geometry and chemistry metadata:

```python
import NebulaPy.src as nebula

nebula.configure_logging(log_file="analysis.log")
logger = nebula.get_logger()

snapshots = nebula.Silo.batch(
    directory="/path/to/silo/files",
    filebase="simulation_name",
    start_time=14.0,
    finish_time=15.0,
    time_unit="days",
)

pion = nebula.pion(snapshots, progress=True)
pion.load_geometry(scale="cm")
pion.load_chemistry()

logger.info("Loaded %s simulation snapshots", len(snapshots))
```

Always protect scripts that use multiprocessing, particularly on macOS and
Windows:

```python
def main():
    # Configure and run the scientific calculation here.
    pass


if __name__ == "__main__":
    main()
```

## Logging and live progress

Logging is optional and follows the directory from which the script is run:

```python
# Terminal output plus ./NebulaPy.log
nebula.configure_logging()

# Terminal output plus a custom file
nebula.configure_logging(log_file="WR140-spectrum.log")

# Detailed terminal diagnostics
nebula.configure_logging(level="DEBUG")

# Terminal output only; do not create a log file
nebula.configure_logging(log_to_file=False)
```

Log files contain timestamps, severity, the executing script, module, function,
line number, and message. Long calculations use compact live progress bars in
interactive terminals, while completion details remain available at `DEBUG`
level.

## Research workflows

The [Research](Research/) directory contains analysis scripts for:

- colliding-wind-binary X-ray spectra;
- NEMO bow-shock emission and cooling maps;
- line-luminosity and dominant-transition studies;
- ATLAS, PoWR, CMFGEN, and blackbody SED processing; and
- CIE and multi-ion comparison calculations.

These scripts are research starting points, not portable command-line tools.
Review their simulation paths, file bases, time ranges, output locations, and
CPU settings before running them on another system.

## Documentation and support

- [NebulaPy documentation](docs/)
- [NebulaPy Wiki](https://github.com/NebulaPy/NebulaPy/wiki)
- [Issue tracker](https://github.com/NebulaPy/NebulaPy/issues)
- [PION project](https://www.pion.ie/)
- [CHIANTI atomic database](https://www.chiantidatabase.org/)

When reporting a problem, include the operating system and architecture, Python
version, NebulaPy and ChiantiPy versions, CHIANTI database version, complete
traceback, and the relevant section of the NebulaPy log file.


## Author

Arun Mathew  
Astronomy & Astrophysics  
Computational and High Energy Astrophysics  
Dublin Institute for Advanced Studies (DIAS), Ireland

NebulaPy is distributed under the MIT License.
