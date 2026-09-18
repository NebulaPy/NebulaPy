"""Compare a NebulaPy CIE X-ray continuum with AtomDB/APEC 3.1.3."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import NebulaPy.src as nebula
from NebulaPy.src.LoggingConfig import configure_logging, get_logger
from NebulaPy.src.Utils import get_element_symbol


logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Matched physical and numerical configuration
# -----------------------------------------------------------------------------
elements = ["O", "Ne", "Fe"]
temperature = 5.0e6  # K
electron_density = 1.0e9  # cm^-3
hydrogen_density = 1.0  # cm^-3; normalization reference
emission_measure = 1.0e27  # cm^-3 = n_e n_H V

minimum_wavelength = 5.0  # Angstrom
maximum_wavelength = 30.0  # Angstrom
number_of_bins = 10000

atomdb_directory = Path("/Users/tony/Desktop/XrayTest/atomdb")
nebulapy_database = Path(__file__).resolve().parents[2] / "nebulapy-db-1.0.0"
output_directory = Path("/Users/tony/Desktop/XrayTest/xray-validation")
output_plot = output_directory / "continuum_APEC_NebulaPy.png"
output_table = output_directory / "continuum_APEC_NebulaPy.txt"

# Anders & Grevesse (1989) elemental number abundances relative to hydrogen,
# matching the SABUND_SOURCE=AG89 metadata in the AtomDB 3.1.3 APEC files.
ag89_abundance = {
    "O": 8.51138038e-4,
    "Ne": 1.23026877e-4,
    "Fe": 4.67735141e-5,
}

KEV_ANGSTROM = 12.398419843320026
KEV_TO_ERG = 1.602176634e-9


def wavelength_edges_and_centres():
    edges = np.linspace(
        minimum_wavelength,
        maximum_wavelength,
        number_of_bins + 1,
        dtype=np.float64,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    return edges, centres


def cie_ion_densities(temperature_grid):
    """Return every O, Ne, and Fe ion density for the CIE plasma."""
    os.environ.setdefault("NEBULAPY_DB", str(nebulapy_database))
    cie = nebula.cieMode()
    cie.load_cie_file()
    densities = {}

    for ion in cie.AllSpecies:
        element = get_element_symbol(ion)
        if element not in elements:
            continue
        densities[ion] = (
            hydrogen_density
            * ag89_abundance[element]
            * cie.get_cie_fraction(ion, temperature_grid)
        )

    return densities


def calculate_nebulapy_continuum(wavelength_centres):
    """Calculate free-free, free-bound, and two-photon CIE continuum."""
    temperature_grid = np.asarray(
        [[[temperature, temperature]]],
        dtype=np.float64,
    )
    electron_density_grid = np.full(
        temperature_grid.shape,
        electron_density,
    )
    species_densities = cie_ion_densities(temperature_grid)

    cell_volume = emission_measure / (
        electron_density * hydrogen_density
    )
    grid_volume = np.full(temperature_grid.shape, cell_volume)
    grid_mask = np.asarray([[[1.0, 0.0]]], dtype=np.float64)

    model = nebula.spectrum(
        min_wavelength=wavelength_centres[0],
        max_wavelength=wavelength_centres[-1],
        elements=elements,
        ion_list=None,
        doBremsstrahlung=True,
        doFreebound=True,
        doLine=False,
        doTwophoton=True,
        userGrid=True,
        gridSize=wavelength_centres.size,
        MPNcores=1,
        progress=False,
    )
    model.generate_spectrum(
        temperature=temperature_grid,
        ne=electron_density_grid,
        species_densities=species_densities,
        grid_volume=grid_volume,
        grid_mask=grid_mask,
    )
    return np.asarray(model.Spectrum, dtype=np.float64)


def calculate_apec_continuum(wavelength_edges):
    """Calculate the independent AtomDB/APEC CIE continuum reference."""
    os.environ["ATOMDB"] = str(atomdb_directory)
    import pyatomdb

    # APEC requires monotonically increasing energy-bin edges. Reversing the
    # wavelength edges provides that order because E is proportional to 1/lambda.
    energy_edges = KEV_ANGSTROM / wavelength_edges[::-1]
    energy_centres = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    wavelength_widths_descending = np.diff(wavelength_edges)[::-1]

    session = pyatomdb.spectrum.CIESession(
        linefile=str(atomdb_directory / "apec_line.fits"),
        cocofile=str(atomdb_directory / "apec_coco.fits"),
        elements=[8, 10, 26],
        # None retains SABUND_SOURCE from the APEC files: AG89 in AtomDB 3.1.3.
        abundset=None,
    )
    session.set_response(energy_edges, raw=True)
    session.set_eebrems(False)
    session.dolines = False
    session.docont = True
    session.dopseudo = False

    photon_emissivity_per_bin = session.return_spectrum(
        temperature,
        teunit="K",
    )

    # APEC returns photons cm^3 s^-1 bin^-1 for a raw unit-area response.
    # Convert to energy luminosity per Angstrom and per steradian, apply EM,
    # then reverse back to increasing wavelength.
    energy_power_descending = (
        photon_emissivity_per_bin
        * emission_measure
        * energy_centres
        * KEV_TO_ERG
        / (4.0 * np.pi)
        / wavelength_widths_descending
    )
    return np.asarray(energy_power_descending[::-1], dtype=np.float64)


def main():
    """Run and save the continuum-spectrum validation comparison."""
    configure_logging(level="DEBUG", log_to_file=True)

    wavelength_edges, wavelength = wavelength_edges_and_centres()
    logger.info(
        "NebulaPy RUN START: CIE continuum, %.3e K, %.1f-%.1f Angstrom",
        temperature,
        minimum_wavelength,
        maximum_wavelength,
    )
    nebulapy_continuum = calculate_nebulapy_continuum(wavelength)
    logger.info("NebulaPy RUN COMPLETE: %s wavelength bins", wavelength.size)

    logger.info(
        "APEC RUN START: AtomDB 3.1.3 CIE continuum, %.3e K, "
        "%.1f-%.1f Angstrom",
        temperature,
        minimum_wavelength,
        maximum_wavelength,
    )
    apec_continuum = calculate_apec_continuum(wavelength_edges)
    logger.info("APEC RUN COMPLETE: %s wavelength bins", wavelength.size)

    expected_shape = wavelength.shape
    for name, values in (
            ("NebulaPy", nebulapy_continuum),
            ("APEC", apec_continuum),
    ):
        if values.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected {name} shape {values.shape}; "
                f"expected {expected_shape}."
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} continuum contains non-finite values.")

    positive_reference = apec_continuum > 0.0
    ratio = np.full_like(apec_continuum, np.nan)
    ratio[positive_reference] = (
        nebulapy_continuum[positive_reference]
        / apec_continuum[positive_reference]
    )

    nebula_integral = np.trapezoid(nebulapy_continuum, wavelength)
    apec_integral = np.trapezoid(apec_continuum, wavelength)
    integrated_ratio = nebula_integral / apec_integral
    median_ratio = np.nanmedian(ratio)
    maximum_fractional_deviation = np.nanmax(np.abs(ratio - 1.0))
    logger.info(
        "COMPARISON COMPLETE: integrated NebulaPy/APEC=%.8f, "
        "median pointwise ratio=%.8f, maximum fractional deviation=%.6f%%",
        integrated_ratio,
        median_ratio,
        100.0 * maximum_fractional_deviation,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_table,
        np.column_stack((
            wavelength,
            nebulapy_continuum,
            apec_continuum,
            ratio,
        )),
        header=(
            "Wavelength[A] NebulaPy APEC NebulaPy/APEC "
            "[spectra: erg s^-1 sr^-1 A^-1]"
        ),
        fmt="%.10e",
    )

    figure, (spectrum_axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    spectrum_axis.plot(
        wavelength,
        apec_continuum,
        color="#D55E00",
        linewidth=1.5,
        label="AtomDB/APEC 3.1.3",
    )
    spectrum_axis.plot(
        wavelength,
        nebulapy_continuum,
        color="#0072B2",
        linewidth=1.2,
        linestyle="--",
        label=f"NebulaPy {nebula.__version__}",
    )
    spectrum_axis.set_yscale("log")
    spectrum_axis.set_ylabel(
        r"$\dfrac{dL}{d\Omega\,d\lambda}$ "
        r"[$\mathrm{erg\,s^{-1}\,sr^{-1}\,\AA^{-1}}$]"
    )
    spectrum_axis.legend()
    spectrum_axis.grid(alpha=0.2)

    information = "\n".join((
        "Single-cell emission with CIE ion fractions",
        rf"$T={temperature:.2e}$ K",
        rf"$n_e={electron_density:.2e}$ cm$^{{-3}}$",
        (
            rf"EM = $n_e n_H V={emission_measure:.2e}$ "
            rf"cm$^{{-3}}$"
        ),
        (
            rf"Band: {minimum_wavelength:.0f}--{maximum_wavelength:.0f} $\AA$ "
            f"({number_of_bins} bins)"
        ),
        "Elements: O, Ne, Fe (all CIE ion stages)",
        "Abundances: Anders & Grevesse (1989)",
        "Continuum: free-free, free-bound, two-photon",
        "Electron-electron bremsstrahlung: excluded",
        f"Integrated NebulaPy/APEC: {integrated_ratio:.5f}",
    ))
    spectrum_axis.text(
        0.98,
        0.78,
        information,
        transform=spectrum_axis.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.92,
        },
    )

    ratio_axis.plot(wavelength, ratio, color="black", linewidth=0.9)
    ratio_axis.axhline(1.0, color="0.5", linewidth=0.8)
    ratio_axis.set_xlabel(r"Wavelength [$\AA$]")
    ratio_axis.set_ylabel("NebulaPy/APEC")
    ratio_axis.set_xlim(minimum_wavelength, maximum_wavelength)
    ratio_axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_plot, bbox_inches="tight", dpi=300)
    plt.close(figure)

    print(f"AtomDB version: {(atomdb_directory / 'VERSION').read_text().strip()}")
    print(f"Temperature: {temperature:.6e} K")
    print(f"Integrated NebulaPy/APEC ratio: {integrated_ratio:.8f}")
    print(f"Median pointwise ratio: {median_ratio:.8f}")
    print(
        "Maximum pointwise fractional deviation: "
        f"{100.0 * maximum_fractional_deviation:.6f}%"
    )
    print(f"Saved comparison table: {output_table}")
    print(f"Saved comparison plot: {output_plot}")


if __name__ == "__main__":
    main()
