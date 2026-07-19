"""Regression test for DEM values extracted from a SILO snapshot."""

from pathlib import Path

import numpy as np

from NebulaPy.src.EmissionMeasure import emissionMeasure


DATA_FILE = Path(__file__).parent / "data" / "dem_snapshot.npz"
SIMULATION_NAME = "NEMO O-star bow shock (non-equilibrium ionization)"
SNAPSHOT_TAG = "0000.00144896"
TMIN = 100.0
TMAX = 1.0e9
NBINS = 200


def test_extracted_snapshot_dem_reference_values():
    """The optimized DEM calculation reproduces SILO-derived values."""
    assert DATA_FILE.is_file()

    with np.load(DATA_FILE) as data:
        temperature = data["temperature"]
        electron_density = data["electron_density"]
        volume = data["volume"]
        mask = data["mask"]
        species_densities = {
            "H": data["density_H"],
            "H1+": data["density_H1p"],
            "O6+": data["density_O6p"],
            "Fe16+": data["density_Fe16p"],
        }
        simulation_time_s = float(data["simulation_time_s"])

    dem = emissionMeasure(
        Tmin=TMIN,
        Tmax=TMAX,
        Nbins=NBINS,
        progress=False,
    )
    dem.DEM2D(
        temperature=temperature,
        ne=electron_density,
        speciesDensities=species_densities,
        volume=volume,
        gridMask=mask,
    )

    assert temperature.shape == (3, 64, 128)
    assert simulation_time_s == 5.21758330125989746e12

    expected_species_dem = {
        ("H", 54): 2.53274321805246966e54,
        ("H1+", 54): 3.29596829267456491e58,
        ("O6+", 70): 9.96428480963072179e48,
        ("Fe16+", 79): 2.21120217848593849e45,
    }

    for (species, bin_index), expected in expected_species_dem.items():
        np.testing.assert_allclose(
            dem.DEM[species][bin_index],
            expected,
            rtol=1.0e-12,
            atol=0.0,
        )

    simulation_time_kyr = simulation_time_s / (365.25 * 24.0 * 3600.0 * 1.0e3)
    print(
        f"\n  Simulation : {SIMULATION_NAME}"
        f"\n  Snapshot   : {SNAPSHOT_TAG}"
        f"\n  Time       : {simulation_time_kyr:.6f} kyr"
        f"\n  Grid       : {temperature.shape}"
        f"\n  DEM        : T={TMIN:.0e}-{TMAX:.0e} K, {NBINS} bins"
        f"\n  Species    : {', '.join(species_densities)}"
    )
