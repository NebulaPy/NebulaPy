"""Build a Pollock et al. (2021)-like three-temperature X-ray template.

This is an intrinsic NebulaPy spectrum, not a detector count spectrum.  It
uses the three temperatures, relative emission measures, common velocity
shift, velocity width, and the NebulaPy-supported subset of the abundances in
Table 3 of Pollock et al. (2021).  TBabs absorption and ARF/RMF response
folding are deliberately left for a later observational-analysis stage.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import NebulaPy.src as nebula
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import configure_logging, get_logger


# Pollock et al. (2021), Table 3.
COMPONENTS = (
    {"temperature_kev": 3.68, "emission_measure": 3.87e53},
    {"temperature_kev": 1.14, "emission_measure": 1.42e53},
    {"temperature_kev": 0.41, "emission_measure": 1.54e53},
)

VELOCITY_SHIFT_KM_S = -1365.0
VELOCITY_SIGMA_KM_S = 578.0
KELVIN_PER_KEV = 1.160451812e7

# Mass fractions from Table 3 for elements currently supported by NebulaPy.
# The omitted Mg, Al, Ar, Ca, and Ni fractions account for less than one per
# cent of the mixture and are not redistributed among the retained elements.
ELEMENT_MASS_FRACTIONS = {
    "H": 0.0,
    "He": 0.548,
    "C": 0.398,
    "N": 0.0,
    "O": 3.70e-2,
    "Ne": 1.09e-2,
    "Si": 1.23e-3,
    "S": 5.18e-4,
    "Fe": 1.09e-3,
}

MINIMUM_ENERGY_KEV = 0.6
MAXIMUM_ENERGY_KEV = 10.0
NUMBER_OF_WAVELENGTHS = 4000
REFERENCE_MASS_DENSITY = 1.0e-14  # g cm^-3; normalization is set by EM.

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output" / "Pollock2021"
OUTPUT_PLOT = OUTPUT_DIRECTORY / "Pollock_three_temperature_template.png"
OUTPUT_DATA = OUTPUT_DIRECTORY / "Pollock_three_temperature_template.txt"


def fully_ionized_electron_density(mass_density):
    """Estimate n_e for the adopted hot WC composition."""
    electrons_per_gram = sum(
        mass_fraction
        * const.ATOMIC_NUMBER[element]
        / const.ATOMIC_MASS[element]
        for element, mass_fraction in ELEMENT_MASS_FRACTIONS.items()
    )
    return mass_density * electrons_per_gram


def main():
    """Calculate, save, and plot the three Pollock-like components."""
    configure_logging(
        level="INFO",
        file_level="DEBUG",
        log_file=OUTPUT_DIRECTORY / "Pollock.log",
        log_to_file=True,
    )
    logger = get_logger(__name__)

    minimum_wavelength = const.KEV_ANGSTROM / MAXIMUM_ENERGY_KEV
    maximum_wavelength = const.KEV_ANGSTROM / MINIMUM_ENERGY_KEV

    spectrum_model = nebula.spectrum(
        min_wavelength=minimum_wavelength,
        max_wavelength=maximum_wavelength,
        elements=list(const.SUPPORTED_ELEMENTS),
        doBremsstrahlung=True,
        doFreebound=True,
        doLine=True,
        doTwophoton=True,
        allLines=True,
        userGrid=True,
        gridSize=NUMBER_OF_WAVELENGTHS,
        MPNcores=1,
        progress=False,
    )

    spectrum_model.initialize_global_line_profile(
        resolving_power=0,
        global_velocity=VELOCITY_SHIFT_KM_S,
        global_velocity_sigma=VELOCITY_SIGMA_KM_S,
        relativistic=False,
    )

    cie = nebula.cieMode()
    component_spectra = []
    wavelength = None

    for component in COMPONENTS:
        temperature_kelvin = (
            component["temperature_kev"] * KELVIN_PER_KEV
        )

        # Two identical cells avoid a ChiantiPy singleton-temperature issue;
        # the second cell is masked and contributes no emission.
        temperature = np.full((1, 1, 2), temperature_kelvin, dtype=np.float64,)

        mass_density = np.full_like(
            temperature,
            REFERENCE_MASS_DENSITY,
        )

        electron_density_value = fully_ionized_electron_density(REFERENCE_MASS_DENSITY)
        electron_density = np.full_like(temperature, electron_density_value,)

        species_densities = cie.build_cie_number_densities(
            element_mass_fractions=ELEMENT_MASS_FRACTIONS,
            temperature=temperature,
            density=mass_density,
        )

        # NebulaPy weights each cell by n_e*n_ion*V.  Using V=EM/n_e^2
        # makes Pollock's tabulated EM control the component normalization.
        # This is a generalized EM scaling; XSPEC's hydrogen-referenced APEC
        # normalization is not identical for a hydrogen-free WC composition.
        emitting_volume = (
            component["emission_measure"] / electron_density_value**2
        )
        grid_volume = np.full_like(temperature, emitting_volume)
        grid_mask = np.asarray([[[1.0, 0.0]]], dtype=np.float64)

        spectrum_model.generate_spectrum(
            temperature=temperature,
            ne=electron_density,
            species_densities=species_densities,
            grid_volume=grid_volume,
            grid_mask=grid_mask,
        )

        wavelength = np.asarray(spectrum_model.WavelengthGrid,
                                dtype=np.float64,)

        component_spectrum = np.asarray(
            spectrum_model.Spectrum,
            dtype=np.float64,
        ).copy()

        component_spectra.append(component_spectrum)

        logger.info(
            "Calculated kT=%.2f keV component with EM=%.3e cm^-3",
            component["temperature_kev"],
            component["emission_measure"],
        )

    component_spectra = np.asarray(component_spectra)
    combined_spectrum = np.sum(component_spectra, axis=0)

    photon_energy_kev = const.KEV_ANGSTROM / wavelength
    photon_energy_erg = photon_energy_kev * 1.602176634e-9
    wavelength_per_energy = (
        const.KEV_ANGSTROM / photon_energy_kev**2
    )
    component_photon_energy_spectra = (
        component_spectra
        * wavelength_per_energy[None, :]
        / photon_energy_erg[None, :]
    )
    combined_photon_energy_spectrum = (
        combined_spectrum
        * wavelength_per_energy
        / photon_energy_erg
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT_DATA,
        np.column_stack((
            wavelength,
            component_photon_energy_spectra.T,
            combined_photon_energy_spectrum,
        )),
        header=(
            "Wavelength[A] kT3.68keV kT1.14keV kT0.41keV "
            "Combined[photon_s^-1_sr^-1_keV^-1]"
        ),
        fmt="%.8e",
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    colors = ("tab:blue", "tab:green", "tab:cyan")
    for component, values, color in zip(
        COMPONENTS,
        component_photon_energy_spectra,
        colors,
    ):
        axis.plot(
            wavelength,
            values,
            color=color,
            linewidth=1.0,
            label=(
                rf"$kT={component['temperature_kev']:.2f}$ keV, "
                rf"$EM={component['emission_measure'] / 1.0e53:.2f}"
                r"\times10^{53}$ cm$^{-3}$"
            ),
        )

    axis.plot(
        wavelength,
        combined_photon_energy_spectrum,
        color="black",
        linewidth=1.4,
        label="Combined three-temperature template",
    )
    axis.set_yscale("log")
    axis.set_xlim(minimum_wavelength, maximum_wavelength)
    axis.set_xlabel(r"Wavelength [$\AA$]")
    axis.set_ylabel(
        r"$dN_E/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ keV$^{-1}$]"
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)

    information = "\n".join((
        rf"Velocity shift: {VELOCITY_SHIFT_KM_S:.0f} km s$^{{-1}}$ (fixed)",
        rf"Line broadening: {VELOCITY_SIGMA_KM_S:.0f} km s$^{{-1}}$ (fixed)",
        "WC abundances: Pollock et al. (2021), Table 3",
        "Model: CIE",
    ))
    axis.text(
        0.98,
        0.97,
        information,
        transform=axis.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )

    positive = combined_photon_energy_spectrum[
        np.isfinite(combined_photon_energy_spectrum)
        & (combined_photon_energy_spectrum > 0.0)
    ]
    if positive.size == 0:
        raise RuntimeError("The combined template contains no emission")
    axis.set_ylim(positive.min() * 0.5, positive.max() * 2.0)

    figure.tight_layout()
    figure.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(figure)

    logger.info("Saved template data: %s", OUTPUT_DATA)
    logger.info("Saved template plot: %s", OUTPUT_PLOT)


if __name__ == "__main__":
    main()
