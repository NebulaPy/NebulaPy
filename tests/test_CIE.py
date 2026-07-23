from math import comb

import numpy as np
import pytest

from NebulaPy.src import Constants as const
from NebulaPy.src.CIE import cieMode
from NebulaPy.src.LoggingConfig import NebulaError


CIE_TEMPERATURE_MIN = 1.0e2
CIE_TEMPERATURE_MAX = 1.0e9
CIE_GRID_POINT_COUNT = 10
CIE_ELEMENT_CHARGES = {"H": 1, "He": 2, "C": 6, "O": 8}
CIE_VERIFIED_SPECIES = ("H", "H1+", "He1+", "C3+", "C6+", "O4+", "O8+")


def _build_normalized_cie_table():
    """Create smooth, normalized test fractions for several elements."""
    log_temperature_grid = np.linspace(
        np.log10(CIE_TEMPERATURE_MIN),
        np.log10(CIE_TEMPERATURE_MAX),
        CIE_GRID_POINT_COUNT,
    )
    ion_columns = [
        f"{element.lower()}_{stage + 1}"
        for element, atomic_number in CIE_ELEMENT_CHARGES.items()
        for stage in range(atomic_number + 1)
    ]
    table_rows = ["log_T " + " ".join(ion_columns)]

    for grid_index, log_temperature in enumerate(log_temperature_grid):
        ionization_progress = grid_index / (CIE_GRID_POINT_COUNT - 1)
        ion_fractions = []
        for atomic_number in CIE_ELEMENT_CHARGES.values():
            ion_fractions.extend(
                comb(atomic_number, charge)
                * ionization_progress**charge
                * (1.0 - ionization_progress) ** (atomic_number - charge)
                for charge in range(atomic_number + 1)
            )
        table_rows.append(
            f"{log_temperature:.12g} "
            + " ".join(f"{fraction:.16e}" for fraction in ion_fractions)
        )

    return "# normalized multi-element CIE test table\n" + "\n".join(
        table_rows
    ) + "\n"


@pytest.fixture
def cie_database(tmp_path, monkeypatch):
    ion_balance = tmp_path / "IonBalance"
    ion_balance.mkdir()
    (ion_balance / "CIE.txt").write_text(
        _build_normalized_cie_table(),
        encoding="utf-8",
    )
    monkeypatch.setenv("NEBULAPYDB", str(tmp_path))
    return tmp_path


def test_cie_end_to_end(cie_database, record_property):
    cie_ion_balance = cieMode()

    # Lazy loading, interpolation, and multidimensional shape preservation.
    temperature_grid = np.logspace(
        np.log10(CIE_TEMPERATURE_MIN),
        np.log10(CIE_TEMPERATURE_MAX),
        CIE_GRID_POINT_COUNT,
    ).reshape(2, 5)
    hydrogen_ion_fractions = cie_ion_balance.get_cie_fraction(
        "H1+", temperature_grid
    )
    expected_ionization_progress = np.linspace(0.0, 1.0, 10).reshape(2, 5)
    np.testing.assert_allclose(
        hydrogen_ion_fractions, expected_ionization_progress
    )
    assert hydrogen_ion_fractions.shape == temperature_grid.shape

    # Additional light and metal ions interpolate on the same arbitrary grid.
    for ion in ("He1+", "C3+", "O4+"):
        interpolated_ion_fractions = cie_ion_balance.get_cie_fraction(
            ion, temperature_grid
        )
        assert interpolated_ion_fractions.shape == temperature_grid.shape
        assert np.all(interpolated_ion_fractions >= 0.0)

    # Invalid physical input is rejected.
    with pytest.raises(ValueError, match="finite, positive"):
        cie_ion_balance.get_cie_fraction("H", 0.0)

    # Scalar number densities use mass density, mass fraction, and atomic mass.
    gas_mass_density = 2.0 * const.ATOMIC_MASS["H"]
    ion_number_densities = cie_ion_balance.build_cie_number_densities(
        {"H": 0.5}, CIE_TEMPERATURE_MIN, gas_mass_density
    )
    assert ion_number_densities["H"] == pytest.approx(1.0)
    assert ion_number_densities["H1+"] == pytest.approx(0.0)
    assert ion_number_densities["He"] == 0.0

    # Multidimensional inputs broadcast and conserve each element's density.
    spatial_mass_density = np.array([[1.0e-24], [2.0e-24]])
    element_mass_fractions = {"H": 0.70, "C": 0.02, "O": 0.03}
    broadcast_ion_number_densities = cie_ion_balance.build_cie_number_densities(
        element_mass_fractions,
        temperature_grid[0:1, :],
        spatial_mass_density,
    )
    assert broadcast_ion_number_densities["O4+"].shape == (2, 5)

    for element, atomic_number in CIE_ELEMENT_CHARGES.items():
        element_ions = [
            element if charge == 0 else f"{element}{charge}+"
            for charge in range(atomic_number + 1)
        ]
        summed_element_number_density = sum(
            broadcast_ion_number_densities[ion] for ion in element_ions
        )
        expected_element_number_density = (
            spatial_mass_density
            * element_mass_fractions.get(element, 0.0)
            / const.ATOMIC_MASS[element]
        )
        np.testing.assert_allclose(
            summed_element_number_density,
            np.broadcast_to(expected_element_number_density, (2, 5)),
            rtol=1.0e-12,
            atol=1.0e-14,
        )

    # Malformed database rows produce a useful error.
    cie_file = cie_database / "IonBalance" / "CIE.txt"
    cie_file.write_text("log_T h_1 h_2\n4.0 1.0\n", encoding="utf-8")
    with pytest.raises(NebulaError, match="row 2"):
        cieMode().load_cie_file()

    record_property(
        "test_summary",
        "  Model      : Collisional ionization equilibrium"
        "\n  Table      : Synthetic normalized ion-fraction grid"
        f"\n  Temperature: T={CIE_TEMPERATURE_MIN:.0e}-"
        f"{CIE_TEMPERATURE_MAX:.0e} K, {CIE_GRID_POINT_COUNT} points"
        f"\n  Input grid : {hydrogen_ion_fractions.shape}"
        "\n  Elements   : H, He, C, O"
        "\n  Density    : Scalar and 2D broadcast inputs"
        f"\n  Species    : {', '.join(CIE_VERIFIED_SPECIES)}",
    )
