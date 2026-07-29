"""Generate a CIE X-ray spectrum from the last WIND-WIND snapshot."""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import NebulaPy.src as nebula
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import configure_logging, get_logger


def main():
    """Build 99 CIE ion-density grids and calculate one integrated spectrum."""
    configure_logging(level="INFO", log_to_file=False)
    logger = get_logger(__name__)

    ###########################################################################
    # WIND-WIND input and output controls
    ###########################################################################

    output_directory = Path(
        "/Users/tony/Desktop/XrayTest/wind-wind"
    )
    silo_directory = Path("/Users/tony/Desktop/XrayTest/wind-wind/silo")
    filebase = "e7_WRwind_d2l5n128_v1500"
    start_time = 23.154  # kyr
    finish_time = None
    time_unit = 'kyr'
    output_frequency = None

    # Spectrum and plot controls.
    # Photon-energy range 0.25-1.5 keV converted using
    # wavelength [Angstrom] = 12.39841875 / energy [keV].
    minimum_wavelength = const.KEV_ANGSTROM / 1.5
    maximum_wavelength = const.KEV_ANGSTROM / 0.25
    wavelength_grid_size = 3000
    number_of_processes = 8
    x_axis = "energy"  # Options: "wavelength" or "energy"
    y_axis = "photon_per_energy"
    y_axis_minimum_factor = 2.0
    y_axis_maximum_factor = 1.0e-2

    # Radiative-process controls.
    do_bremsstrahlung = True
    do_freebound = True
    do_line = True
    do_twophoton = True

    ###########################################################################
    # Select the final snapshot and load its physical grids
    ###########################################################################

    batched_silos = nebula.Silo.batch(
        silo_directory,
        filebase,
        start_time=start_time,
        finish_time=finish_time,
        time_unit=time_unit,
        out_frequency=output_frequency,
    )
    if not batched_silos:
        raise RuntimeError("No WIND-WIND snapshots matched the selection")

    last_snapshot = batched_silos[-1]
    pion = nebula.pion([last_snapshot], progress=False)
    pion.load_chemistry()
    pion.load_geometry(scale="cm")

    simulation_time = pion.get_simulation_time(last_snapshot,time_unit=time_unit,)
    temperature = np.asarray(pion.get_parameter("Temperature", last_snapshot), dtype=np.float64,)
    mass_density = np.asarray(pion.get_parameter("Density", last_snapshot),dtype=np.float64,)
    grid_volume = np.asarray(pion.get_grid_volumes_2D(),dtype=np.float64,)
    grid_mask = np.asarray(pion.geometry_container["mask"],dtype=np.float64,)

    # Deliberately overwrite the elemental mass-fraction grids stored in the
    # SILO snapshot with the uniform WC9 wind composition in Table 2 of
    # Mathew et al. (2025), adopted there from Eatson et al. (2022a) for
    # H, He, C, N, and O, with solar values for Ne, Si, S, and Fe:
    # https://doi.org/10.1051/0004-6361/202452373
    #
    # Each scalar is broadcast over the full simulation grid. The dictionary
    # defines the parent composition for all 99 CIE ion stages; hydrogen and
    # nitrogen ion densities are correctly zero for this adopted WC9 mixture.
    element_mass_fractions = {
        "H": 0.0,
        "He": 0.546,
        "C": 0.4,
        "N": 0.0,
        "O": 0.05,
        "Ne": 1.258e-3,
        "Si": 6.656e-4,
        "S": 3.096e-4,
        "Fe": 1.293e-3,
    }

    ###########################################################################
    # Approximate electron density without calling pion.get_ne()
    ###########################################################################
    # At X-ray temperatures the plasma is approximately fully ionized:
    #
    #   n_e = rho * sum_element(X_element * Z_element / m_element)
    #
    # where rho is in g cm^-3 and m_element is the atomic mass in grams.
    fully_ionized_electrons_per_gram = np.zeros_like(mass_density)
    for element, mass_fraction in element_mass_fractions.items():
        fully_ionized_electrons_per_gram += (
            mass_fraction
            * const.ATOMIC_NUMBER[element]
            / const.ATOMIC_MASS[element]
        )
    electron_number_density = np.maximum(
        mass_density * fully_ionized_electrons_per_gram,
        const.ELECTRON_DENSITY_FLOOR,
    )

    ###########################################################################
    # Distribute elemental densities among all 99 CIE ion stages
    ###########################################################################
    cie_ion_balance = nebula.cieMode()
    species_densities = cie_ion_balance.build_cie_number_densities(
        element_mass_fractions=element_mass_fractions,
        temperature=temperature,
        density=mass_density,
    )
    if len(species_densities) != 99:
        raise RuntimeError(
            "Expected 99 CIE ion-density grids, "
            f"but received {len(species_densities)}"
        )

    logger.info(
        "Last snapshot: %.6g %s; approximate electron density and "
        "99 CIE ion-density grids are ready",
        simulation_time.value,
        simulation_time.unit,
    )

    ###########################################################################
    # Calculate the integrated X-ray spectrum
    ###########################################################################

    spectrum_start_time = perf_counter()
    spectrum_model = nebula.spectrum(
        min_wavelength=minimum_wavelength,
        max_wavelength=maximum_wavelength,
        elements=list(const.SUPPORTED_ELEMENTS),
        ion_list=['Fe24+'],
        doBremsstrahlung=do_bremsstrahlung,
        doFreebound=do_freebound,
        doLine=do_line,
        doTwophoton=do_twophoton,
        filtername=None,
        filterfactor=None,
        userGrid=True,
        gridSize=wavelength_grid_size,
        allLines=False,
        MPNcores=number_of_processes,
        progress=True,
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
    wavelength_luminosity = np.asarray(
        spectrum_model.Spectrum,
        dtype=np.float64,
    )
    spectrum_runtime = perf_counter() - spectrum_start_time

    ###########################################################################
    # Convert to the selected plotting axes
    ###########################################################################

    photon_energy = const.KEV_ANGSTROM / wavelength

    if y_axis == "energy_per_wavelength":
        y_values = wavelength_luminosity
        ylabel = r"$dL_\lambda/d\Omega$ [erg s$^{-1}$ sr$^{-1}$ $\AA^{-1}$]"
    elif y_axis == "energy_per_energy":
        y_values = (
            wavelength_luminosity
            * const.KEV_ANGSTROM
            / photon_energy**2
        )
        ylabel = r"$dL_E/d\Omega$ [erg s$^{-1}$ sr$^{-1}$ keV$^{-1}$]"
    elif y_axis == "photon_per_wavelength":
        y_values = wavelength_luminosity / (
            photon_energy * 1.602176634e-9
        )
        ylabel = r"$dN_\lambda/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ $\AA^{-1}$]"
    elif y_axis == "photon_per_energy":
        energy_luminosity = (
            wavelength_luminosity
            * const.KEV_ANGSTROM
            / photon_energy**2
        )
        y_values = energy_luminosity / (
            photon_energy * 1.602176634e-9
        )
        ylabel = r"$dN_E/d\Omega$ [photons s$^{-1}$ sr$^{-1}$ keV$^{-1}$]"
    else:
        raise ValueError(
            "y_axis must be 'energy_per_wavelength', "
            "'energy_per_energy', 'photon_per_wavelength', "
            "or 'photon_per_energy'"
        )

    if x_axis == "wavelength":
        plot_order = np.arange(wavelength.size)
        plot_x = wavelength
        xlabel = r"Wavelength [$\AA$]"
    elif x_axis == "energy":
        plot_order = np.argsort(photon_energy)
        plot_x = photon_energy[plot_order]
        xlabel = "Photon energy [keV]"
    else:
        raise ValueError(
            "x_axis must be either 'wavelength' or 'energy'"
        )
    plot_y = y_values[plot_order]

    ###########################################################################
    # Save the numerical spectrum and plot
    ###########################################################################

    output_directory.mkdir(parents=True, exist_ok=True)
    spectrum_file = output_directory / (
        f"{filebase}_CIE_spectrum_last_snapshot.txt"
    )
    plot_file = output_directory / (
        f"{filebase}_CIE_spectrum_last_snapshot.png"
    )
    np.savetxt(
        spectrum_file,
        np.column_stack((wavelength, wavelength_luminosity)),
        header="Wavelength[A] Spectrum[erg s^-1 sr^-1 A^-1]",
        fmt="%.8e",
    )

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(plot_x, plot_y, color="black", linewidth=1.2)
    axis.set_yscale("log")
    positive_values = plot_y[np.isfinite(plot_y) & (plot_y > 0.0)]
    if positive_values.size == 0:
        raise RuntimeError("The calculated CIE spectrum contains no emission")
    maximum_plot_value = float(np.max(positive_values))
    axis.set_ylim(
        maximum_plot_value * y_axis_minimum_factor,
        maximum_plot_value * y_axis_maximum_factor,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(which="both", alpha=0.2)
    axis.minorticks_on()

    enabled_processes = [
        name
        for enabled, name in (
            (do_bremsstrahlung, "free-free"),
            (do_freebound, "free-bound"),
            (do_line, "lines"),
            (do_twophoton, "two-photon"),
        )
        if enabled
    ]
    plot_information = "\n".join((
        f"Filebase: {filebase}",
        f"Simulation time: {simulation_time.value:.5g} "
        f"{simulation_time.unit}",
        "Ionization: collisional ionization equilibrium",
        "Composition: uniform WC9 wind override",
        f"Elements: {', '.join(const.SUPPORTED_ELEMENTS)}",
        "Ion-density grids: 99",
        (
            r"$n_e \approx \rho\sum_X X_X Z_X/m_X$ "
            "(fully ionized approximation)"
        ),
        f"Processes: {', '.join(enabled_processes)}",
        f"Spectrum runtime: {spectrum_runtime:.1f} s",
    ))
    information_box = axis.legend(
        [],
        [],
        title=plot_information,
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )
    information_box.get_title().set_fontsize(9)
    information_box.get_title().set_multialignment("left")
    information_box.get_frame().set_edgecolor("0.4")

    figure.tight_layout()
    figure.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(figure)

    logger.info("Saved CIE spectrum data: %s", spectrum_file)
    logger.info("Saved CIE spectrum plot: %s", plot_file)


if __name__ == "__main__":
    main()
