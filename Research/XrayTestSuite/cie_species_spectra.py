"""Generate individual and integrated single-cell CIE ion spectra."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import NebulaPy.src as nebula
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import (
    NebulaError,
    configure_logging,
    get_logger,
)
from NebulaPy.src.Utils import get_element_symbol


def main():
    """Save each ion spectrum and the sum over all calculated CIE ions."""
    configure_logging(level="DEBUG", log_to_file=True)
    logger = get_logger(__name__)

    ###########################################################################
    # User controls
    ###########################################################################

    do_bremsstrahlung = True
    do_freebound = True
    do_line = True
    do_twophoton = True

    # x_axis: "wavelength" or "energy"
    # y_axis: "energy_per_wavelength", "energy_per_energy",
    #         "photon_per_wavelength", or "photon_per_energy"
    x_axis = "wavelength"
    y_axis = "energy_per_wavelength"
    y_axis_minimum_factor = 1.0e-5
    y_axis_maximum_factor = 2.0

    minimum_wavelength = 0.5
    maximum_wavelength = 20.0
    wavelength_grid_size = 3000
    number_of_processes = 1
    include_all_lines = False

    output_directory = Path(
        "/Users/tony/Desktop/XrayTest/cie_all_species_spectra"
    )

    element_mass_fractions = {
        "H": 0.0,
        "He": 5.48e-1,
        "C": 3.98e-1,
        "N": 0.0,
        "O": 3.70e-2,
        "Ne": 1.09e-2,
        "Si": 1.23e-3,
        "S": 5.18e-4,
        "Fe": 1.09e-3,
    }

    ###########################################################################
    # Single-active-cell physical inputs
    ###########################################################################

    # The second, masked cell avoids ChiantiPy's one-sample array limitation.
    temperature = np.asarray([[[4.3e7, 3.0e6]]], dtype=np.float64)
    electron_number_density = np.asarray(
        [[[1.0e9, 1.0e9]]],
        dtype=np.float64,
    )
    grid_volume = np.asarray([[[1.0, 1.0]]], dtype=np.float64)
    grid_mask = np.asarray([[[1.0, 0.0]]], dtype=np.float64)

    fully_ionized_electrons_per_gram = sum(
        element_mass_fractions[element]
        * const.ATOMIC_NUMBER[element]
        / const.ATOMIC_MASS[element]
        for element in element_mass_fractions
    )
    gas_mass_density = (
        electron_number_density / fully_ionized_electrons_per_gram
    )

    # Build the complete 99-entry CIE ion-density dictionary. Each ion is then
    # selected independently below to reproduce the original Fe24+ workflow.
    cie_ion_balance = nebula.cieMode()
    species_densities = cie_ion_balance.build_cie_number_densities(
        element_mass_fractions=element_mass_fractions,
        temperature=temperature,
        density=gas_mass_density,
    )

    ###########################################################################
    # Independent radiative-process controls
    ###########################################################################

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

    output_directory.mkdir(parents=True, exist_ok=True)
    skipped_species = []
    completed_species = 0
    integrated_process_luminosities = {
        process_name: None for process_name in process_settings
    }
    integrated_wavelength = None

    ###########################################################################
    # Calculate and save a separate spectrum for every CIE ion
    ###########################################################################

    for species_index, ion in enumerate(species_densities, start=1):
        element = get_element_symbol(ion)
        filename_ion = ion.replace("+", "p")
        logger.info(
            "Calculating CIE spectrum %s/%s for %s",
            species_index,
            len(species_densities),
            ion,
        )

        process_wavelength_luminosities = {}
        wavelength = None

        try:
            for process_name, enabled_process in process_settings.items():
                logger.info(
                    "Calculating %s contribution for %s",
                    process_name,
                    ion,
                )
                spectrum_model = nebula.spectrum(
                    min_wavelength=minimum_wavelength,
                    max_wavelength=maximum_wavelength,
                    elements=[element],
                    ion_list=[ion],
                    userGrid=True,
                    gridSize=wavelength_grid_size,
                    allLines=include_all_lines,
                    MPNcores=number_of_processes,
                    progress=False,
                    **enabled_process,
                )
                spectrum_model.generate_spectrum(
                    temperature=temperature,
                    ne=electron_number_density,
                    species_densities=species_densities,
                    grid_volume=grid_volume,
                    grid_mask=grid_mask,
                )
                wavelength = np.asarray(
                    spectrum_model.WavelengthGrid,
                    dtype=np.float64,
                )
                process_wavelength_luminosities[process_name] = np.asarray(
                    spectrum_model.Spectrum,
                    dtype=np.float64,
                )
        except NebulaError as error:
            skipped_species.append(ion)
            logger.warning(
                "Skipping %s because CHIANTI cannot calculate its spectrum: %s",
                ion,
                error,
            )
            continue

        total_wavelength_luminosity = np.sum(
            np.stack(list(process_wavelength_luminosities.values())),
            axis=0,
        )

        # Accumulate every process independently. Their final sum is the
        # integrated spectrum over all successfully calculated CIE ions.
        if integrated_wavelength is None:
            integrated_wavelength = wavelength.copy()
        elif not np.array_equal(integrated_wavelength, wavelength):
            raise ValueError(
                f"Wavelength grid for {ion} differs from earlier species"
            )

        for process_name, wavelength_luminosity in (
            process_wavelength_luminosities.items()
        ):
            if integrated_process_luminosities[process_name] is None:
                integrated_process_luminosities[process_name] = (
                    wavelength_luminosity.copy()
                )
            else:
                integrated_process_luminosities[process_name] += (
                    wavelength_luminosity
                )

        #######################################################################
        # Save the numerical spectrum for this ion
        #######################################################################

        spectrum_file = output_directory / (
            f"CIE_{filename_ion}_spectrum.txt"
        )
        plot_file = output_directory / (
            f"CIE_{filename_ion}_spectrum.png"
        )
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
                + " Total [erg s^-1 sr^-1 A^-1]"
            ),
            fmt="%.8e",
        )

        #######################################################################
        # Convert and plot this ion's process contributions
        #######################################################################

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
                "x_axis must be either 'wavelength' or 'energy'"
            )

        figure, axis = plt.subplots(figsize=(12, 7))
        plot_styles = {
            "Free-free": {"color": "tab:blue", "linestyle": "--"},
            "Free-bound": {"color": "tab:orange", "linestyle": "-."},
            "Lines": {"color": "tab:green", "linestyle": ":"},
            "Two-photon": {"color": "tab:red", "linestyle": "--"},
            "Total": {
                "color": "black",
                "linestyle": "-",
                "linewidth": 1.8,
            },
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
                y_values = wavelength_luminosity / (
                    energy * kev_to_erg
                )
                ylabel = r"$dN_\lambda/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ $\AA^{-1}$]"
            elif y_axis == "photon_per_energy":
                energy_luminosity = (
                    wavelength_luminosity * kev_angstrom / energy**2
                )
                y_values = energy_luminosity / (
                    energy * kev_to_erg
                )
                ylabel = r"$dN_E/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ keV$^{-1}$]"
            else:
                raise ValueError(
                    "y_axis must be 'energy_per_wavelength', "
                    "'energy_per_energy', 'photon_per_wavelength', "
                    "or 'photon_per_energy'"
                )

            plot_y = y_values[plot_order]
            finite_plot_y = plot_y[np.isfinite(plot_y)]
            if finite_plot_y.size:
                maximum_plot_value = max(
                    maximum_plot_value,
                    float(np.max(finite_plot_y)),
                )
            axis.plot(
                plot_x,
                plot_y,
                label=spectrum_name,
                linewidth=plot_styles[spectrum_name].get(
                    "linewidth",
                    1.2,
                ),
                color=plot_styles[spectrum_name]["color"],
                linestyle=plot_styles[spectrum_name]["linestyle"],
            )

        axis.set_yscale("linear")
        if maximum_plot_value > 0.0:
            axis.set_ylim(
                maximum_plot_value * y_axis_minimum_factor,
                maximum_plot_value * y_axis_maximum_factor,
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(which="both", alpha=0.2)

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
            (
                f"Wavelength: {wavelength[0]:.2f}-"
                f"{wavelength[-1]:.2f} Å"
            ),
            f"Ion: {ion}",
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

        figure.tight_layout()
        figure.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(figure)

        completed_species += 1
        logger.info("Saved %s spectrum: %s", ion, spectrum_file)
        logger.info("Saved %s plot: %s", ion, plot_file)

    ###########################################################################
    # Save and plot the spectrum integrated over all calculated CIE ions
    ###########################################################################

    if completed_species == 0:
        raise NebulaError(
            "No CIE species spectrum was calculated; "
            "the integrated spectrum cannot be created"
        )

    integrated_total_luminosity = np.sum(
        np.stack(list(integrated_process_luminosities.values())),
        axis=0,
    )
    integrated_spectrum_file = (
        output_directory / "CIE_all_species_integrated_spectrum.txt"
    )
    integrated_plot_file = (
        output_directory / "CIE_all_species_integrated_spectrum.png"
    )
    np.savetxt(
        integrated_spectrum_file,
        np.column_stack([
            integrated_wavelength,
            *integrated_process_luminosities.values(),
            integrated_total_luminosity,
        ]),
        header=(
            "Wavelength[A] "
            + " ".join(
                process_name.replace("-", "")
                for process_name in integrated_process_luminosities
            )
            + " Total [erg s^-1 sr^-1 A^-1]"
        ),
        fmt="%.8e",
    )

    energy = kev_angstrom / integrated_wavelength
    if x_axis == "wavelength":
        integrated_plot_x = integrated_wavelength
        integrated_plot_order = np.arange(integrated_wavelength.size)
        xlabel = r"Wavelength [$\AA$]"
    else:
        integrated_plot_order = np.argsort(energy)
        integrated_plot_x = energy[integrated_plot_order]
        xlabel = "Photon energy [keV]"

    figure, axis = plt.subplots(figsize=(12, 7))
    integrated_plotted_spectra = {
        **integrated_process_luminosities,
        "Total": integrated_total_luminosity,
    }
    maximum_plot_value = 0.0

    for spectrum_name, wavelength_luminosity in (
        integrated_plotted_spectra.items()
    ):
        if y_axis == "energy_per_wavelength":
            y_values = wavelength_luminosity
            ylabel = r"$dL_\lambda/d\Omega$ [erg s$^{-1}$ sr$^{-1}$ $\AA^{-1}$]"
        elif y_axis == "energy_per_energy":
            y_values = wavelength_luminosity * kev_angstrom / energy**2
            ylabel = r"$dL_E/d\Omega$ [erg s$^{-1}$ sr$^{-1}$ keV$^{-1}$]"
        elif y_axis == "photon_per_wavelength":
            y_values = wavelength_luminosity / (energy * kev_to_erg)
            ylabel = r"$dN_\lambda/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ $\AA^{-1}$]"
        else:
            energy_luminosity = (
                wavelength_luminosity * kev_angstrom / energy**2
            )
            y_values = energy_luminosity / (energy * kev_to_erg)
            ylabel = r"$dN_E/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ keV$^{-1}$]"

        integrated_plot_y = y_values[integrated_plot_order]
        finite_plot_y = integrated_plot_y[
            np.isfinite(integrated_plot_y)
        ]
        if finite_plot_y.size:
            maximum_plot_value = max(
                maximum_plot_value,
                float(np.max(finite_plot_y)),
            )
        axis.plot(
            integrated_plot_x,
            integrated_plot_y,
            label=spectrum_name,
            linewidth=plot_styles[spectrum_name].get("linewidth", 1.2),
            color=plot_styles[spectrum_name]["color"],
            linestyle=plot_styles[spectrum_name]["linestyle"],
        )

    axis.set_yscale("linear")
    if maximum_plot_value > 0.0:
        axis.set_ylim(
            maximum_plot_value * y_axis_minimum_factor,
            maximum_plot_value * y_axis_maximum_factor,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(which="both", alpha=0.2)

    integrated_plot_information = "\n".join((
        "Model: Integrated collisional ionization equilibrium",
        f"Temperature: {temperature[0, 0, 0]:.3e} K",
        (
            "Electron density: "
            f"{electron_number_density[0, 0, 0]:.3e} cm$^{{-3}}$"
        ),
        f"Mass density: {gas_mass_density[0, 0, 0]:.3e} g cm$^{{-3}}$",
        f"Cell volume: {grid_volume[0, 0, 0]:.3e} cm$^3$",
        f"Processes: {', '.join(process_settings)}",
        f"Axes: {x_axis}, {y_axis}",
        f"Included ions: {completed_species}",
        f"Skipped ions: {len(skipped_species)}",
    ))
    information_legend = axis.legend(
        loc="best",
        title=integrated_plot_information,
        fontsize=9,
        title_fontsize=8.5,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
    )
    information_legend.get_title().set_multialignment("left")
    information_legend.get_frame().set_edgecolor("0.4")

    figure.tight_layout()
    figure.savefig(
        integrated_plot_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    logger.info(
        "Saved integrated all-species spectrum: %s",
        integrated_spectrum_file,
    )
    logger.info(
        "Saved integrated all-species plot: %s",
        integrated_plot_file,
    )

    logger.info(
        "Completed %s of %s CIE species; output: %s",
        completed_species,
        len(species_densities),
        output_directory,
    )
    if skipped_species:
        logger.warning(
            "Skipped %s unavailable species: %s",
            len(skipped_species),
            ", ".join(skipped_species),
        )


if __name__ == "__main__":
    main()
