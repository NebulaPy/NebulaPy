# NebulaPy

[![PyPI version](https://badge.fury.io/py/NebulaPy.svg)](https://pypi.org/project/NebulaPy/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)

NebulaPy is a Python library for post-processing [PION](https://www.pion.ie/) simulation data.

## Features

- **Spectral Energy Distribution (SED) Binning**: Energy binning of stellar atmosphere models, including ATLAS, Potsdam, and 
CMFGEN, and Blackbody across a wide range of metal abundances.
- **Line Luminosity Calculations**:
  - Computes line luminosities for spherical geometry (1D nested and uniform grid).
  - Computes line luminosities for cylindrical geometry (2D nested and uniform grid).
  - Computes line luminosities for cylindrical geometry via MultiProcessing.
- **Cooling Function Maps for 2D simulations**
- **Identifies the dominant spectral lines** for ions in 1D and 2D simulation snapshots by
calculating the line luminosity for all lines based on grid-level data. It then outputs
the dominant lines for a specified list of ions.
- **Emissivity Map Generator for 2D simulation snapshot**:
  - Generates emrissivity maps for single ion multi-line emissions from a 2D simulation snapshot.
  - Generates emissivity maps for multiple ions, multi-line emissions from a 2D simulation snapshot.
  

## Installation

```bash
pip install NebulaPy
# This will also install dependencies

# Download and set up the CHIANTI database (require ~1 GB disk space)
wget https://download.chiantidatabase.org/CHIANTI_10.1_database.tar.gz

# Extract it to a directory (~5 GB disk space required)
tar -xzf CHIANTI_10.1_database.tar.gz -C CHIANTI-DATABASE-DIRECTORY

# Add the following environmental variable to your .bashrc
echo "export XUVTOP=CHIANTI-DATABASE-DIRECTORY" >> ~/.bashrc

# Reload your .bashrc
source ~/.bashrc

# Install the Python-SILO interface:
# Execute the following command from the NebulaPy root directory.
install-silo

# Fix the SILO library path in your local distribution:
# Open the file:
#   ${HOME}/.local/venv/lib/python3.11/site-packages/pypion/SiloHeader_data.py
# Modify line 18 to append /lib to the path

# To download the NebulaPy database:
# Execute the following command from the NebulaPy root directory.
# If a destination path is not specified, the download will default to the
# root directory. This requires approximately 270 MB of additional space.
download-database [destination_path]

# Add environmental variable for NebulaPy Database
echo "export NEBULAPYDB=NEBULAPY-DATABASE-DIRECTORY" >> ~/.bashrc

# Reload your .bashrc
source ~/.bashrc
```

## Usage

For detailed usage instructions, examples, and features, please 
visit [NebulaPy Wiki](https://github.com/arunmathewofficial/NebulaPy/wiki). 
Sample scripts demonstrating NebulaPy functionalities can be found
in the `NebulaPy/Research` directory.


## Documentation

Check the full documentation at [NebulaPy GitHub](https://github.com/arunmathewofficial/NebulaPy).

## Support

For bug reports and feature requests, visit the
[issues section](https://github.com/arunmathewofficial/NebulaPy/issues) of the repository:

## Changelog
- **Version 1.0.0-beta** – March 5, 2025: Beta release
- **Version 1.0.1-beta** – March 6, 2025: Minor bug fixes
- **Version 1.0.2-beta** – March 6, 2025: Include silo installation script
- **Version 1.0.3-beta** – March 6, 2025: Fixed bugs in spectral line emissivity map script

## Author
Arun Mathew  
Astronomy & Astrophysics  
Computational and High Energy Astrophysics  
Dublin Institute for Advanced Studies (DIAS), Ireland  

