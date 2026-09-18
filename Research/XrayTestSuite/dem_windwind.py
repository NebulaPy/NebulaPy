"""Calculate and plot differential emission measures for WIND-WIND."""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import NebulaPy.src as nebula
from NebulaPy.src.LoggingConfig import configure_logging, get_logger


def main():
    """Generate species and total DEM plots for each selected snapshot."""
    configure_logging(level="INFO", log_to_file=False)
    logger = get_logger(__name__)

    ###########################################################################
    # Colliding wind binary paths and snapshot selection
    ###########################################################################

    output_directory = Path(
        "/Users/tony/Desktop/XrayTest/wind-wind/dem-plots"
    )
    silo_directory = Path("/Users/tony/Desktop/XrayTest/wind-wind/silo")
    filebase = "e7_WRwind_d2l5n128_v1500"
    start_time = None
    finish_time = None
    time_unit = "kyr"
    output_frequency = None

    minimum_temperature = 1.0e2
    maximum_temperature = 1.0e9
    number_of_temperature_bins = 200

    ###########################################################################
    # Batch snapshots and load the simulation metadata
    ###########################################################################

    batched_silos = nebula.Silo.batch(
        silo_directory,
        filebase,
        start_time=start_time,
        finish_time=finish_time,
        time_unit=time_unit,
        out_frequency=output_frequency,
    )

    pion = nebula.pion(batched_silos, progress=False)

    nemo = nebula.NEMO(pion)
    nemo.load_chemistry()
    # DEM uses number densities in cm^-3, so cell volumes must remain in cm^3.
    # There is no spatial axis in this plot; its x-axis is log10 temperature.
    pion.load_geometry(scale="cm")

    geometry = pion.geometry_container
    elements_present = ", ".join(
        nemo.chemistry_container["tracer_elements"]
    )
    number_of_grid_cells = geometry["Ngrid"]
    grid_edges_min = geometry["edges_min"]
    grid_edges_max = geometry["edges_max"]
    grid_mask = geometry["mask"]
    cell_volume = pion.get_grid_volumes_2D()

    emission_measure = nebula.emissionMeasure(
        Tmin=minimum_temperature,
        Tmax=maximum_temperature,
        Nbins=number_of_temperature_bins,
        progress=False,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    total_runtime = 0.0

    ###########################################################################
    # Calculate and plot the DEM for every selected snapshot
    ###########################################################################

    for snapshot_index, silo_snapshot in enumerate(batched_silos):
        snapshot_start_time = perf_counter()
        simulation_time = pion.get_simulation_time(
            silo_snapshot,
            time_unit=time_unit,
        )

        temperature = pion.get_parameter("Temperature", silo_snapshot)
        mass_density = pion.get_parameter("Density", silo_snapshot)
        electron_number_density = nemo.get_ne(
            silo_snapshot,
            progress=False,
        )
        species_number_densities = nemo.get_species_number_densities(
            silo_snapshot
        )

        emission_measure.DEM2D(
            temperature=temperature,
            ne=electron_number_density,
            speciesDensities=species_number_densities,
            volume=cell_volume,
            gridMask=grid_mask,
        )

        electron_dem = emission_measure.SAM_DEM(
            density=mass_density,
            temperature=temperature,
            ne=electron_number_density,
            mask=grid_mask,
            ngrid=number_of_grid_cells,
            volume=cell_volume,
            mesh_edges_min=grid_edges_min,
            mesh_edges_max=grid_edges_max,
            temp_bin=emission_measure.Tb,
            hw=emission_measure.half_bin_width,
        )

        total_species_dem = np.zeros_like(
            emission_measure.Tb,
            dtype=np.float64,
        )

        for species_dem_values in emission_measure.DEM.values():
            total_species_dem += np.asarray(
                species_dem_values,
                dtype=np.float64,
            )

        #######################################################################
        # Compare the summed ion DEM with the electron-density DEM
        #######################################################################

        positive_species_bins = total_species_dem > 0.0
        positive_electron_bins = electron_dem > 0.0

        figure, axis = plt.subplots(figsize=(8, 6))
        axis.plot(
            emission_measure.Tb[positive_species_bins],
            np.log10(total_species_dem[positive_species_bins]),
            linewidth=1.5,
            marker="o",
            markersize=3,
            color="black",
            label=(
                r"$\mathrm{DEM}_j=\sum_s\sum_{i\in j}"
                r"n_{\mathrm{e},i}n_{s,i}V_iM_i$"
            ),
        )
        axis.plot(
            emission_measure.Tb[positive_electron_bins],
            np.log10(electron_dem[positive_electron_bins]),
            linewidth=1.5,
            marker="o",
            markersize=3,
            color="red",
            label=(
                r"$\mathrm{DEM}_j=\sum_{i\in j}"
                r"n_{\mathrm{e},i}^{2}V_iM_i$"
            ),
        )
        axis.set_xlim(
            np.log10(minimum_temperature),
            np.log10(maximum_temperature),
        )
        axis.set_xlabel(r"$\log_{10}(T/\mathrm{K})$")
        axis.set_ylabel(r"$\log_{10}(\mathrm{DEM}/\mathrm{cm^{-3}})$")
        axis.text(
            0.03,
            0.95,
            (
                f"Filebase: {filebase}\n"
                "Plot: Differential Emission Measure\n"
                f"Simulation time: {simulation_time.value:.4g} "
                f"{simulation_time.unit}\n"
                f"Bins: {number_of_temperature_bins}\n"
                f"Elements: {elements_present}"
            ),
            transform=axis.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )
        axis.legend(loc="best")
        axis.grid(alpha=0.2)

        comparison_output_file = output_directory / (
            f"{filebase}_total_DEM_{snapshot_index:04d}.png"
        )
        figure.savefig(
            comparison_output_file,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(figure)

        snapshot_runtime = perf_counter() - snapshot_start_time
        total_runtime += snapshot_runtime
        logger.info(
            "Saved DEM snapshot %s/%s at %.4g %s in %.2f s",
            snapshot_index + 1,
            len(batched_silos),
            simulation_time.value,
            simulation_time.unit,
            snapshot_runtime,
        )

    logger.info(
        "Completed %s DEM snapshots in %.2f s; output: %s",
        len(batched_silos),
        total_runtime,
        output_directory,
    )


if __name__ == "__main__":
    main()
