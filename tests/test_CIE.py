"""Reference checks against the real NebulaPy CIE ion-balance table."""

from pathlib import Path

import numpy as np
import pytest

from NebulaPy.src import Constants as const
from NebulaPy.src.CIE import cieMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CIE_DATABASE_DIRECTORY = PROJECT_ROOT / "NebulaPy-DB"
CIE_DATA_FILE = CIE_DATABASE_DIRECTORY / "IonBalance" / "CIE.txt"
CIE_ELEMENT_CHARGES = {"H": 1, "C": 6, "O": 8, "Fe": 26}
CIE_REFERENCE_COLUMNS = {
    "H1+": "h_2",
    "C3+": "c_4",
    "O6+": "o_7",
    "O8+": "o_9",
    "Fe24+": "fe_25",
    "Fe25+": "fe_26",
}


def _read_reference_cie_table():
    """Read the real table independently of ``cieMode.load_cie_file``."""
    with CIE_DATA_FILE.open("r", encoding="utf-8") as stream:
        content_lines = [
            line.strip()
            for line in stream
            if line.strip() and not line.startswith("#")
        ]

    column_names = content_lines[0].split()
    reference_data = np.loadtxt(content_lines[1:], dtype=np.float64)
    column_indices = {
        column_name: column_index
        for column_index, column_name in enumerate(column_names)
    }
    return reference_data, column_indices


@pytest.fixture
def real_cie_database(monkeypatch):
    """Point CIE mode at the real repository database."""
    assert CIE_DATA_FILE.is_file()
    monkeypatch.setenv("NEBULAPYDB", str(CIE_DATABASE_DIRECTORY))
    return _read_reference_cie_table()


def test_cie(real_cie_database, record_property):
    reference_data, reference_columns = real_cie_database
    cie_ion_balance = cieMode()

    assert reference_data.shape == (1024, 100)
    cie_ion_balance.load_cie_file()
    np.testing.assert_array_equal(cie_ion_balance.data, reference_data)

    # Exact grid temperatures must reproduce the stored fractions.
    reference_rows = np.array([0, 257, 511, 767, 1023])
    reference_temperatures = 10.0 ** reference_data[reference_rows, 0]
    for ion, table_column in CIE_REFERENCE_COLUMNS.items():
        actual_fractions = cie_ion_balance.get_cie_fraction(
            ion,
            reference_temperatures,
        )
        expected_fractions = reference_data[
            reference_rows,
            reference_columns[table_column],
        ]
        np.testing.assert_allclose(
            actual_fractions,
            expected_fractions,
            rtol=0.0,
            atol=1.0e-15,
        )

    # Interpolation is linear in log10 temperature.
    lower_row = 600
    upper_row = lower_row + 1
    midpoint_log_temperature = np.mean(
        reference_data[[lower_row, upper_row], 0]
    )
    for ion, table_column in CIE_REFERENCE_COLUMNS.items():
        actual_midpoint = cie_ion_balance.get_cie_fraction(
            ion,
            10.0 ** midpoint_log_temperature,
        )
        expected_midpoint = np.mean(
            reference_data[
                [lower_row, upper_row],
                reference_columns[table_column],
            ]
        )
        assert actual_midpoint == pytest.approx(
            expected_midpoint,
            rel=1.0e-12,
            abs=1.0e-15,
        )

    # Every ion stage is normalized to unity at each tabulated temperature.
    for element, atomic_number in CIE_ELEMENT_CHARGES.items():
        chianti_element = element.lower()
        element_columns = [
            reference_columns[f"{chianti_element}_{stage}"]
            for stage in range(1, atomic_number + 2)
        ]
        np.testing.assert_allclose(
            np.sum(reference_data[:, element_columns], axis=1),
            1.0,
            rtol=0.0,
            atol=2.0e-9,
        )

    # Broadcast multidimensional inputs and conserve rho*X/m.
    temperature_grid = np.logspace(
        reference_data[0, 0],
        reference_data[-1, 0],
        10,
    ).reshape(1, 2, 5)
    spatial_mass_density = np.asarray([1.0e-24, 2.0e-24]).reshape(2, 1, 1)
    element_mass_fractions = {"H": 0.70, "C": 0.02, "O": 0.03}
    ion_number_densities = cie_ion_balance.build_cie_number_densities(
        element_mass_fractions,
        temperature_grid,
        spatial_mass_density,
    )
    assert ion_number_densities["O6+"].shape == (2, 2, 5)

    for element, atomic_number in CIE_ELEMENT_CHARGES.items():
        element_ions = [
            element if charge == 0 else f"{element}{charge}+"
            for charge in range(atomic_number + 1)
        ]
        summed_number_density = sum(
            ion_number_densities[ion] for ion in element_ions
        )
        expected_number_density = (
            spatial_mass_density
            * element_mass_fractions.get(element, 0.0)
            / const.ATOMIC_MASS[element]
        )
        np.testing.assert_allclose(
            summed_number_density,
            np.broadcast_to(expected_number_density, (2, 2, 5)),
            rtol=2.0e-9,
            atol=1.0e-14,
        )

    with pytest.raises(ValueError, match="finite, positive"):
        cie_ion_balance.get_cie_fraction("O6+", 0.0)

    record_property(
        "test_summary",
        "  Model      : Collisional ionization equilibrium"
        f"\n  Temperature: T={10.0 ** reference_data[0, 0]:.2e}-"
        f"{10.0 ** reference_data[-1, 0]:.2e} K"
        f"\n  Grid       : {reference_data.shape[0]} temperature points"
        "\n  Checks     : raw values, interpolation, normalization, density"
        f"\n  Species    : {', '.join(CIE_REFERENCE_COLUMNS)}",
    )
