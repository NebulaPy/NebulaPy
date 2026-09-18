"""Plot a user-defined Gaussian line profile initialized by ``spectrum``.

The spectrum object owns the LineProfile configuration and passes that profile
to the current NebulaPy CHIANTI line-coefficient calculation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import NebulaPy.src as nebula
from NebulaPy.src.LoggingConfig import configure_logging
from NebulaPy.src.Utils import get_spectroscopic_symbol


# -----------------------------------------------------------------------------
# Calculation configuration
# -----------------------------------------------------------------------------
pion_ion = "Fe13+"  # Fe XIV in spectroscopic notation

temperature = 2.0e6  # K
electron_density = 1.0e9  # cm^-3

minimum_wavelength = 200.0  # Angstrom
maximum_wavelength = 300.0  # Angstrom
number_of_wavelengths = 3000

resolving_power = 0
VELOCITY_SHIFT_KM_S = -1365.0
VELOCITY_SIGMA_KM_S = 578.0
relativistic = False


wavelength = np.linspace(
    minimum_wavelength,
    maximum_wavelength,
    number_of_wavelengths,
    dtype=np.float64,
)

output_directory = Path(
    "/Users/tony/Desktop/XrayTest/line-emission-coefficients"
)
output_filename = "Fe13+_line_emission_coefficients.png"


def main():

    """Calculate Fe13+ coefficients and save their temperature comparison."""
    configure_logging(level="DEBUG", log_to_file=True)

    # user defined
    spectrum_model = nebula.spectrum(
        min_wavelength=minimum_wavelength,
        max_wavelength=maximum_wavelength,
        elements=["Fe"],
        ion_list=[pion_ion],
        doBremsstrahlung=True,
        doFreebound=True,
        doLine=True,
        doTwophoton=True,
        userGrid=True,
        gridSize=number_of_wavelengths,
        MPNcores=1,
        progress=False,
    )

    # ChiantiPy 0.15.2 fails for a singleton temperature vector. Use two
    # identical cells and mask the second one so only one cell contributes.
    temperature_grid = np.asarray(
        [[[temperature, temperature]]],
        dtype=np.float64,
    )
    electron_density_grid = np.full(
        temperature_grid.shape,
        electron_density,
    )
    species_densities = {
        pion_ion: np.ones_like(temperature_grid),
    }
    grid_volume = np.ones_like(temperature_grid)
    grid_mask = np.asarray([[[1.0, 0.0]]], dtype=np.float64)

    # default ChiantiPy line profile (no user-defined profile)
    spectrum_model.generate_spectrum(
        temperature=temperature_grid,
        ne=electron_density_grid,
        species_densities=species_densities,
        grid_volume=grid_volume,
        grid_mask=grid_mask,
    )
    wavelength = spectrum_model.WavelengthGrid.copy()
    default_spectrum = spectrum_model.Spectrum.copy()

    # Snapshot-global line shift and Gaussian broadening.
    spectrum_model.initialize_global_line_profile(
        resolving_power=resolving_power,
        global_velocity=VELOCITY_SHIFT_KM_S,
        global_velocity_sigma=VELOCITY_SIGMA_KM_S,
        relativistic=relativistic,
    )

    spectrum_model.generate_spectrum(
        temperature=temperature_grid,
        ne=electron_density_grid,
        species_densities=species_densities,
        grid_volume=grid_volume,
        grid_mask=grid_mask,
    )

    user_spectrum = spectrum_model.Spectrum.copy()
    spectroscopic_name = get_spectroscopic_symbol(pion_ion)

    expected_shape = wavelength.shape
    for name, spectrum_values in (
            ("default", default_spectrum),
            ("user-defined", user_spectrum),
    ):
        if spectrum_values.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected {name} spectrum shape: "
                f"{spectrum_values.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(spectrum_values)):
            raise RuntimeError(
                f"{name} spectrum contains non-finite values."
            )

    output_directory.mkdir(parents=True, exist_ok=True)




    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        wavelength,
        default_spectrum,
        color="#8C8C8C",
        linewidth=1.3,
        alpha=0.4,
        label=f"{spectroscopic_name} Default ChiantiPy Profile",
    )
    axis.plot(
        wavelength,
        user_spectrum,
        color="#0072B2",
        linewidth=1.3,
        label=f"{spectroscopic_name} Global Velocity Profile",
        zorder=3,
    )

    axis.set_xlabel(r"Wavelength ($\AA$)", fontsize=12)
    axis.set_ylabel(
        r"Line spectral power (erg s$^{-1}$ sr$^{-1}$ $\AA^{-1}$)",
        fontsize=12,
    )
    axis.set_xlim(minimum_wavelength, maximum_wavelength)
    #axis.set_yscale("log")
    axis.legend()
    axis.grid(alpha=0.2)

    information = "\n".join((
        rf"$T={temperature:.2e}$ K",
        rf"$n_e={electron_density:.2e}$ cm$^{{-3}}$",
        rf"$n_{{\mathrm{{Fe\,XIV}}}}=1$ cm$^{{-3}}$",
        r"Cell volume = 1 cm$^3$",
        "Process: line emission",
        (
            rf"Grid: {minimum_wavelength:.0f}--{maximum_wavelength:.0f} "
            rf"$\AA$ ({number_of_wavelengths} points)"
        ),
        "Default: grid-limited ChiantiPy gaussianR",
        rf"Resolving Power = {resolving_power:.0f}",
        rf"Global velocity: ${VELOCITY_SHIFT_KM_S:.0f}$ km s$^{{-1}}$",
        (
            rf"Global velocity sigma: ${VELOCITY_SIGMA_KM_S:.0f}$ "
            rf"km s$^{{-1}}$"
        ),
        f"Relativistic shift: {relativistic}",
    ))
    axis.text(
        0.02,
        0.97,
        information,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )

    spectrum_maximum = max(
        default_spectrum.max(),
        user_spectrum.max(),
    )
    axis.set_ylim(0.0, spectrum_maximum * 1.03)

    figure.tight_layout()

    output_path = output_directory / output_filename
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)

    print(f"Ion: {spectroscopic_name} ({pion_ion})")
    print(f"Spectrum shape: {default_spectrum.shape}")
    print(
        "Default spectrum range: "
        f"{default_spectrum.min():.6e} to "
        f"{default_spectrum.max():.6e} per Angstrom"
    )
    print(
        "User spectrum range: "
        f"{user_spectrum.min():.6e} to "
        f"{user_spectrum.max():.6e} per Angstrom"
    )
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
