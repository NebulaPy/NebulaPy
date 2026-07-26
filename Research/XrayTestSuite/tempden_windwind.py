"""Plot density and temperature maps for one WIND-WIND simulation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import NebulaPy.src as nebula
from NebulaPy.src.LoggingConfig import configure_logging, get_logger


def main():
    """Generate paired density-temperature plots for all selected snapshots."""
    configure_logging(level="INFO", log_to_file=False)
    logger = get_logger(__name__)

    ###########################################################################
    # Colliding wind binaries — MacBook paths and filenames
    ###########################################################################

    output_directory = Path("/Users/tony/Desktop/XrayTest/wind-wind/density-temperature-plots")
    silo_directory = Path("/Users/tony/Desktop/XrayTest/wind-wind/silo")
    filebase = "e7_WRwind_d2l5n128_v1500"
    start_time = None
    finish_time = None
    time_unit = "kyr"
    output_frequency = None
    simulation_name = "WIND-WIND"

    spatial_scale = "cm"
    density_limits = None
    temperature_limits = (3.0, 7.0)

    ###########################################################################
    # Batch SILO snapshots and load geometry
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
    pion.load_geometry(scale=spatial_scale)

    geometry = pion.geometry_container
    number_of_levels = geometry["Nlevel"]
    grid_edges_min = geometry["edges_min"]
    grid_edges_max = geometry["edges_max"]

    output_directory.mkdir(parents=True, exist_ok=True)

    ###########################################################################
    # Plot each selected snapshot
    ###########################################################################

    for snapshot_index, silo_snapshot in enumerate(batched_silos):
        simulation_time = pion.get_simulation_time(
            silo_snapshot,
            time_unit=time_unit,
        )
        density = pion.get_parameter("Density", silo_snapshot)
        temperature = pion.get_parameter("Temperature", silo_snapshot)

        log_density_by_level = [
            np.log10(np.maximum(values, np.finfo(np.float64).tiny))
            for values in density
        ]
        log_temperature_by_level = [
            np.log10(np.maximum(values, np.finfo(np.float64).tiny))
            for values in temperature
        ]

        # A single normalization must be shared by every refinement level.
        # Otherwise, identical values are assigned different colours on each
        # level and the rectangular refinement boundaries become visible.
        density_minimum = (
            min(np.nanmin(values) for values in log_density_by_level)
            if density_limits is None
            else density_limits[0]
        )
        density_maximum = (
            max(np.nanmax(values) for values in log_density_by_level)
            if density_limits is None
            else density_limits[1]
        )
        density_normalization = Normalize(
            vmin=density_minimum,
            vmax=density_maximum,
        )
        temperature_normalization = Normalize(
            vmin=temperature_limits[0],
            vmax=temperature_limits[1],
        )

        figure, (density_axis, temperature_axis) = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            sharex=True,
        )

        density_image = None
        temperature_image = None

        for level in range(number_of_levels):
            minimum_z = grid_edges_min[level][0].value
            maximum_z = grid_edges_max[level][0].value
            minimum_radius = grid_edges_min[level][1].value
            maximum_radius = grid_edges_max[level][1].value

            log_density = log_density_by_level[level]
            log_temperature = log_temperature_by_level[level]

            # Do not draw coarse cells that are covered by any finer level.
            # The finer image is plotted afterwards into the resulting opening.
            if level < number_of_levels - 1:
                number_of_radius_cells, number_of_z_cells = log_density.shape
                z_centres = np.linspace(
                    minimum_z,
                    maximum_z,
                    number_of_z_cells,
                    endpoint=False,
                )
                z_centres += (
                    maximum_z - minimum_z
                ) / (2.0 * number_of_z_cells)
                radius_centres = np.linspace(
                    minimum_radius,
                    maximum_radius,
                    number_of_radius_cells,
                    endpoint=False,
                )
                radius_centres += (
                    maximum_radius - minimum_radius
                ) / (2.0 * number_of_radius_cells)

                covered_by_finer_level = np.zeros(
                    log_density.shape,
                    dtype=bool,
                )
                for finer_level in range(level + 1, number_of_levels):
                    finer_minimum_z = grid_edges_min[finer_level][0].value
                    finer_maximum_z = grid_edges_max[finer_level][0].value
                    finer_minimum_radius = grid_edges_min[finer_level][1].value
                    finer_maximum_radius = grid_edges_max[finer_level][1].value
                    covered_by_finer_level |= (
                        (radius_centres[:, None] >= finer_minimum_radius)
                        & (radius_centres[:, None] <= finer_maximum_radius)
                        & (z_centres[None, :] >= finer_minimum_z)
                        & (z_centres[None, :] <= finer_maximum_z)
                    )

                log_density = np.ma.masked_where(
                    covered_by_finer_level,
                    log_density,
                )
                log_temperature = np.ma.masked_where(
                    covered_by_finer_level,
                    log_temperature,
                )

            density_image = density_axis.imshow(
                log_density,
                interpolation="nearest",
                cmap="viridis",
                extent=[
                    minimum_z,
                    maximum_z,
                    minimum_radius,
                    maximum_radius,
                ],
                origin="lower",
                aspect="auto",
                norm=density_normalization,
            )
            temperature_image = temperature_axis.imshow(
                log_temperature,
                interpolation="nearest",
                cmap="inferno",
                extent=[
                    minimum_z,
                    maximum_z,
                    -maximum_radius,
                    -minimum_radius,
                ],
                origin="upper",
                aspect="auto",
                norm=temperature_normalization,
            )

        density_axis.set_xlim(
            grid_edges_min[0][0].value,
            grid_edges_max[0][0].value,
        )
        density_axis.set_ylim(
            grid_edges_min[0][1].value,
            grid_edges_max[0][1].value,
        )
        temperature_axis.set_ylim(
            -grid_edges_max[0][1].value,
            -grid_edges_min[0][1].value,
        )

        density_axis.set_ylabel(f"R ({spatial_scale})", fontsize=12)
        temperature_axis.set_ylabel(f"R ({spatial_scale})", fontsize=12)
        temperature_axis.set_xlabel(f"z ({spatial_scale})", fontsize=12)
        density_axis.tick_params(axis="both", labelsize=11)
        temperature_axis.tick_params(axis="both", labelsize=11)

        density_axis.text(
            0.03,
            0.90,
            (
                f"{simulation_name}: "
                f"{simulation_time.value:.4g} {simulation_time.unit}"
            ),
            transform=density_axis.transAxes,
            fontsize=10,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round,pad=0.3",
            },
        )
        density_axis.text(
            0.97,
            0.90,
            r"$\log_{10}(\rho/\mathrm{g\,cm^{-3}})$",
            transform=density_axis.transAxes,
            horizontalalignment="right",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        )
        temperature_axis.text(
            0.97,
            0.08,
            r"$\log_{10}(T/\mathrm{K})$",
            transform=temperature_axis.transAxes,
            horizontalalignment="right",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        )

        if density_image is None or temperature_image is None:
            raise RuntimeError("No density-temperature data were available")

        density_divider = make_axes_locatable(density_axis)
        density_colorbar_axis = density_divider.append_axes(
            "right",
            size="5%",
            pad=0.05,
        )
        density_colorbar = figure.colorbar(
            density_image,
            cax=density_colorbar_axis,
        )
        density_colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())

        temperature_divider = make_axes_locatable(temperature_axis)
        temperature_colorbar_axis = temperature_divider.append_axes(
            "right",
            size="5%",
            pad=0.05,
        )
        temperature_colorbar = figure.colorbar(
            temperature_image,
            cax=temperature_colorbar_axis,
        )
        temperature_colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())

        figure.subplots_adjust(hspace=0.0)
        output_file = (
            output_directory
            / f"{filebase}_density_temperature_{snapshot_index:04d}.png"
        )
        figure.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(figure)

        logger.info(
            "Saved density-temperature snapshot %s/%s: %s",
            snapshot_index + 1,
            len(batched_silos),
            output_file,
        )


if __name__ == "__main__":
    main()
