"""Generate a single-cell X-ray spectrum using CIE ion densities."""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script runs without opening a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import NebulaPy.src as nebula
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import configure_logging, get_logger


def main():
    """Calculate, save, and plot a process-resolved CIE spectrum."""
    configure_logging(level="DEBUG", log_to_file=True)
    logger = get_logger(__name__)

    ###########################################################################
    # User controls
    ###########################################################################

    # Select the radiative processes included in the calculation.
    do_bremsstrahlung = True
    do_freebound = True
    do_line = False
    do_twophoton = True

    # Plot-axis options.
    # x_axis: "wavelength" or "energy"
    # y_axis: "energy_per_wavelength", "energy_per_energy",
    #         "photon_per_wavelength", or "photon_per_energy"
    x_axis = "wavelength"
    y_axis = "energy_per_wavelength"

    # Set the logarithmic y-axis range relative to the spectrum's peak value.
    y_axis_minimum_factor = 1.0e-5
    y_axis_maximum_factor = 2.0

    # Solar-like composition used to convert mass density to ion density.
    element_mass_fractions = {
        "H": 7.1125e-1,
        "He": 2.7169e-1,
        "C": 3.0989e-3,
        "N": 9.0813e-4,
        "O": 8.4361e-3,
        "Ne": 1.8940e-3,
        "Si": 7.7755e-4,
        "S": 5.3521e-4,
        "Fe": 1.4082e-3,
    }

    # With ion_list=None, Spectrum uses every available ion belonging to the
    # selected elements.
    spectrum_elements = list(element_mass_fractions)
    ion_list = ['Fe24+']

    ###########################################################################
    # Single-active-cell physical inputs
    ###########################################################################

    # One active physical cell plus one identical masked padding cell.
    # The padding avoids ChiantiPy's one-sample array limitation and contributes
    # exactly zero to the integrated spectrum.
    temperature = np.asarray([[[3.0e7, 3.0e6]]], dtype=np.float64)
    electron_number_density = np.asarray(
        [[[1.0e9, 1.0e9]]],
        dtype=np.float64,
    )


    # Fully ionized electrons per gram:
    # sum(X_element * atomic_number / atomic_mass).
    fully_ionized_electrons_per_gram = sum(
        element_mass_fractions[element]
        * const.ATOMIC_NUMBER[element]
        / const.ATOMIC_MASS[element]
        for element in element_mass_fractions
    )
    gas_mass_density = (
        electron_number_density / fully_ionized_electrons_per_gram
    )

    # Distribute each elemental number density among its CIE ion stages:
    # n_ion = density * element_mass_fraction * ion_fraction / atomic_mass.
    cie_ion_balance = nebula.cieMode()
    species_densities = cie_ion_balance.build_cie_number_densities(
        element_mass_fractions=element_mass_fractions,
        temperature=temperature,
        density=gas_mass_density,
    )

    # Baseline volumes; the temporary validation block below overrides only
    # the active cell. Removing that block restores a 1 cm^3 active cell.
    grid_volume = np.asarray([[[1.0, 1.0]]], dtype=np.float64)
    grid_mask = np.asarray([[[1.0, 0.0]]], dtype=np.float64)

    # --- Start removable CHIANTI EM-normalization block. ---
    # Delete this block to restore the baseline 1 cm^3 active-cell volume.
    #
    # Match CHIANTI's volumetric emission-measure convention:
    # EM_H = n_e * n_H * V, where n_H includes neutral and ionized hydrogen.
    # NebulaPy applies the elemental abundance and CIE fraction through the
    # physical ion densities supplied to Spectrum.
    active_cell = (0, 0, 0)
    target_emission_measure = 1.0e27
    hydrogen_number_density = (
        gas_mass_density
        * element_mass_fractions["H"]
        / const.ATOMIC_MASS["H"]
    )
    grid_volume[active_cell] = target_emission_measure / (
        electron_number_density[active_cell]
        * hydrogen_number_density[active_cell]
    )
    calculated_emission_measure = (
        electron_number_density[active_cell]
        * hydrogen_number_density[active_cell]
        * grid_volume[active_cell]
    )
    logger.info(
        "CHIANTI EM normalization: n_e=%.6e cm^-3, n_H=%.6e cm^-3, "
        "V=%.12e cm^3, EM=%.6e cm^-3",
        electron_number_density[active_cell],
        hydrogen_number_density[active_cell],
        grid_volume[active_cell],
        calculated_emission_measure,
    )
    # --- End removable CHIANTI EM-normalization block. ---

    ###########################################################################
    # Independent radiative-process calculations
    ###########################################################################

    # Each enabled entry runs separately so its contribution can be saved and
    # plotted independently before constructing the total spectrum.
    available_processes = {
        "Free-free": (do_bremsstrahlung, {
            "doBremsstrahlung": True,
            "doFreebound": False,
            "doLine": False,
            "doTwophoton": False,
        }),
        "Free-bound": (do_freebound, {
            "doBremsstrahlung": False,
            "doFreebound": True,
            "doLine": False,
            "doTwophoton": False,
        }),
        "Lines": (do_line, {
            "doBremsstrahlung": False,
            "doFreebound": False,
            "doLine": True,
            "doTwophoton": False,
        }),
        "Two-photon": (do_twophoton, {
            "doBremsstrahlung": False,
            "doFreebound": False,
            "doLine": False,
            "doTwophoton": True,
        }),
    }
    process_settings = {
        process_name: settings
        for process_name, (enabled, settings) in available_processes.items()
        if enabled
    }
    if not process_settings:
        raise ValueError("Enable at least one radiative process")

    # Every process uses the same wavelength grid and physical-cell inputs.
    process_wavelength_luminosities = {}
    wavelength = None
    for process_name, enabled_process in process_settings.items():
        logger.info("Calculating %s contribution", process_name)
        spectrum_model = nebula.spectrum(
            min_wavelength=0.5,
            max_wavelength=20.0,
            elements=spectrum_elements,
            ion_list=ion_list,
            userGrid=True,
            gridSize=3000,
            allLines=False,
            MPNcores=1,
            progress=True,
            **enabled_process,
        )
        spectrum_model.generate_spectrum(
            temperature=temperature,
            ne=electron_number_density,
            species_densities=species_densities,
            grid_volume=grid_volume,
            grid_mask=grid_mask,
        )
        wavelength = np.asarray(spectrum_model.WavelengthGrid)
        process_wavelength_luminosities[process_name] = np.asarray(
            spectrum_model.Spectrum
        )

    # Independent radiative contributions add linearly.
    total_wavelength_luminosity = np.sum(
        np.stack(list(process_wavelength_luminosities.values())),
        axis=0,
    )

    ###########################################################################
    # Save process-resolved wavelength spectra
    ###########################################################################

    output_directory = Path("/Users/tony/Desktop/XrayTest")
    output_directory.mkdir(parents=True, exist_ok=True)
    spectrum_file = output_directory / "CIE_single_cell_spectrum.txt"
    plot_file = output_directory / "CIE_single_cell_spectrum.png"

    np.savetxt(
        spectrum_file,
        np.column_stack([
            wavelength,
            *process_wavelength_luminosities.values(),
            total_wavelength_luminosity,
        ]),
        header=(
            "Wavelength[A] "
            + " ".join(
                process_name.replace("-", "")
                for process_name in process_wavelength_luminosities
            )
            + " Total "
            "[erg s^-1 sr^-1 A^-1]"
        ),
        fmt="%.8e",
    )

    ###########################################################################
    # Convert the spectrum to the selected axes and plot
    ###########################################################################

    kev_angstrom = 12.39841984
    kev_to_erg = 1.602176634e-9
    energy = kev_angstrom / wavelength

    if x_axis == "wavelength":
        plot_x = wavelength
        plot_order = np.arange(wavelength.size)
        xlabel = r"Wavelength [$\AA$]"
    elif x_axis == "energy":
        plot_order = np.argsort(energy)
        plot_x = energy[plot_order]
        xlabel = "Photon energy [keV]"
    else:
        raise ValueError(
            "x_axis must be either 'wavelength' or 'energy'."
        )

    figure, axis = plt.subplots(figsize=(12, 7))
    plot_styles = {
        "Free-free": {"color": "tab:blue", "linestyle": "--"},
        "Free-bound": {"color": "tab:orange", "linestyle": "-."},
        "Lines": {"color": "tab:green", "linestyle": ":"},
        "Two-photon": {"color": "tab:red", "linestyle": "--"},
        "Total": {"color": "black", "linestyle": "-", "linewidth": 1.8},
    }
    plotted_spectra = {
        **process_wavelength_luminosities,
        "Total": total_wavelength_luminosity,
    }
    maximum_plot_value = 0.0
    for spectrum_name, wavelength_luminosity in plotted_spectra.items():
        if y_axis == "energy_per_wavelength":
            y_values = wavelength_luminosity
            ylabel = r"$dL_\lambda/d\Omega$ [erg s$^{-1}$ sr$^{-1}$ $\AA^{-1}$]"
        elif y_axis == "energy_per_energy":
            y_values = (
                wavelength_luminosity * kev_angstrom / energy**2
            )
            ylabel = r"$dL_E/d\Omega$ [erg s$^{-1}$ sr$^{-1}$ keV$^{-1}$]"
        elif y_axis == "photon_per_wavelength":
            y_values = wavelength_luminosity / (energy * kev_to_erg)
            ylabel = r"$dN_\lambda/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ $\AA^{-1}$]"
        elif y_axis == "photon_per_energy":
            energy_luminosity = (
                wavelength_luminosity * kev_angstrom / energy**2
            )
            y_values = energy_luminosity / (energy * kev_to_erg)
            ylabel = r"$dN_E/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ keV$^{-1}$]"
        else:
            raise ValueError(
                "y_axis must be 'energy_per_wavelength', "
                "'energy_per_energy', 'photon_per_wavelength', "
                "or 'photon_per_energy'."
            )

        plot_y = y_values[plot_order]
        maximum_plot_value = max(
            maximum_plot_value,
            float(np.max(plot_y)),
        )
        axis.plot(
            plot_x,
            plot_y,
            label=spectrum_name,
            linewidth=plot_styles[spectrum_name].get("linewidth", 1.2),
            color=plot_styles[spectrum_name]["color"],
            linestyle=plot_styles[spectrum_name]["linestyle"],
        )

    # Ignore numerical underflow far below the useful plotted dynamic range.
    # The text file still retains the original, untruncated values.
    axis.set_yscale("linear")
    axis.set_ylim(
        maximum_plot_value * y_axis_minimum_factor,
        maximum_plot_value * y_axis_maximum_factor,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(which="both", alpha=0.2)

    # Combine the curve labels and calculation settings in one box so
    # Matplotlib can place everything at the least obstructive location.
    ion_description = (
        f"all available ions of {', '.join(spectrum_elements)}"
        if ion_list is None
        else ", ".join(ion_list)
    )
    plot_information = "\n".join((
        "Model: Collisional ionization equilibrium",
        f"Temperature: {temperature[0, 0, 0]:.3e} K",
        (
            "Electron density: "
            f"{electron_number_density[0, 0, 0]:.3e} cm$^{{-3}}$"
        ),
        f"Mass density: {gas_mass_density[0, 0, 0]:.3e} g cm$^{{-3}}$",
        f"Cell volume: {grid_volume[0, 0, 0]:.3e} cm$^3$",
        f"Processes: {', '.join(process_settings)}",
        f"Axes: {x_axis}, {y_axis}",
        f"Wavelength: {wavelength[0]:.2f}-{wavelength[-1]:.2f} Å",
        f"Ions: {ion_description}",
    ))
    information_legend = axis.legend(
        loc="best",
        title=plot_information,
        fontsize=9,
        title_fontsize=8.5,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
    )
    information_legend.get_title().set_multialignment("left")
    information_legend.get_frame().set_edgecolor("0.4")

    # Save directly to disk using the non-interactive Matplotlib backend.
    figure.tight_layout()
    figure.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(figure)

    logger.info("Saved spectrum: %s", spectrum_file)
    logger.info("Saved plot: %s", plot_file)


if __name__ == "__main__":
    main()
