"""Compare NEQ oxygen fractions with CIE fractions for one PION snapshot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

import NebulaPy.src as nebula
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import NebulaError, configure_logging, get_logger


def main():
    """Read one NEQ snapshot and plot NEQ versus temperature-derived CIE."""
    configure_logging(level="INFO", log_to_file=False)
    logger = get_logger(__name__)

    ###########################################################################
    # User controls
    ###########################################################################

    neq_silo_directory = Path("/Users/tony/Desktop/Multi-Ion-Bowshock/multi-ion-bowshock/high-res-silo-200kyr")
    neq_filebase = "Ostar_mhd-nemo-dep_d2n0384l3"
    start_time = None
    finish_time = None
    time_unit = "kyr"
    output_frequency = None

    output_directory = Path("/Users/tony/Desktop/XrayTest")
    spatial_scale = "pc"

    oxygen_ions = [
        "O",
        "O1+",
        "O2+",
        "O3+",
        "O4+",
        "O5+",
        "O6+",
        "O7+",
        "O8+",
    ]

    ###########################################################################
    # Load the NEQ snapshot with the current NebulaPy PION interface
    ###########################################################################

    neq_batched_silos = nebula.Silo.batch(
        neq_silo_directory,
        neq_filebase,
        start_time=start_time,
        finish_time=finish_time,
        time_unit=time_unit,
        out_frequency=output_frequency,
    )
    if len(neq_batched_silos) != 1:
        raise NebulaError(
            "Expected exactly one NEQ snapshot, but the batch selection "
            f"returned {len(neq_batched_silos)}. Set a narrower time range."
        )

    neq_silo = neq_batched_silos[0]
    neq_pion = nebula.pion(neq_batched_silos, progress=False)
    neq_pion.load_geometry(scale=spatial_scale)
    neq_pion.load_chemistry()

    geometry = neq_pion.geometry_container
    if geometry["coordinate_sys"] != "cylindrical":
        raise NebulaError(
            "Requires a 2D cylindrical snapshot"
        )

    number_of_levels = geometry["Nlevel"]
    grid_edges_min = geometry["edges_min"]
    grid_edges_max = geometry["edges_max"]
    grid_mask = geometry["mask"]

    temperature = neq_pion.get_parameter("Temperature", neq_silo)
    oxygen_mass_fraction = neq_pion.get_parameter(
        neq_pion.chemistry_container["mass_fractions"]["O"],
        neq_silo,
    )
    simulation_time = neq_pion.get_simulation_time(
        neq_silo,
        time_unit="kyr",
    )

    ###########################################################################
    # Build NEQ tracer fractions and CIE fractions on the same temperature grid
    ###########################################################################

    cie_ion_balance = nebula.cieMode()
    neq_ion_fractions = {}
    cie_ion_fractions = {}

    for ion in oxygen_ions:
        if ion in const.FULLY_IONIZED_IONS:
            ion_mass_fraction = neq_pion.get_top_ion_massfrac(ion, neq_silo)
        else:
            ion_mass_fraction = neq_pion.get_ion_values(ion, neq_silo)

        neq_ion_fractions[ion] = [
            np.divide(
                ion_mass_fraction[level],
                oxygen_mass_fraction[level],
                out=np.zeros_like(
                    ion_mass_fraction[level],
                    dtype=np.float64,
                ),
                where=oxygen_mass_fraction[level] > 0.0,
            )
            for level in range(number_of_levels)
        ]
        cie_ion_fractions[ion] = [
            cie_ion_balance.get_cie_fraction(ion, temperature[level])
            for level in range(number_of_levels)
        ]

    ###########################################################################
    # Original paired plotting layout
    ###########################################################################

    number_of_columns = 4
    first_ion_group = oxygen_ions[1:1 + number_of_columns]
    second_ion_group = oxygen_ions[
        1 + number_of_columns:1 + 2 * number_of_columns
    ]

    figure = plt.figure(figsize=(12.3, 7))
    grid = GridSpec(
        5,
        1,
        height_ratios=[1, 1, 0.5, 1, 1],
        hspace=0.0,
    )
    row_ions = [
        first_ion_group,
        first_ion_group,
        None,
        second_ion_group,
        second_ion_group,
    ]
    image = None

    for row, ions_in_row in enumerate(row_ions):
        if ions_in_row is None:
            spacer_axis = figure.add_subplot(grid[row])
            spacer_axis.axis("off")
            continue

        row_grid = grid[row].subgridspec(
            1,
            number_of_columns,
            wspace=0.0,
        )
        is_neq_row = row in (0, 3)

        for column, ion in enumerate(ions_in_row):
            axis = figure.add_subplot(row_grid[0, column])
            axis.set_xlim(
                grid_edges_min[0][0].value,
                grid_edges_max[0][0].value,
            )

            if is_neq_row:
                axis.axes.get_xaxis().set_visible(False)
                axis.set_ylim(
                    grid_edges_min[0][1].value,
                    grid_edges_max[0][1].value,
                )
            else:
                axis.set_xlabel(f"z ({spatial_scale})", fontsize=15)
                axis.set_ylim(
                    -grid_edges_max[0][1].value,
                    -grid_edges_min[0][1].value,
                )

            if column != 0:
                axis.axes.get_yaxis().set_visible(False)
            else:
                axis.set_ylabel(
                    f"R ({spatial_scale})",
                    fontsize=15,
                    labelpad=15 if is_neq_row else 4,
                )

            for level in range(number_of_levels):
                minimum_z = grid_edges_min[level][0].value
                maximum_z = grid_edges_max[level][0].value
                minimum_radius = grid_edges_min[level][1].value
                maximum_radius = grid_edges_max[level][1].value
                active_cells = np.asarray(grid_mask[level]) > 0.0

                selected_fraction = (
                    neq_ion_fractions[ion][level]
                    if is_neq_row
                    else cie_ion_fractions[ion][level]
                )
                plot_data = np.where(
                    active_cells,
                    selected_fraction,
                    np.nan,
                )
                radial_extents = (
                    [minimum_radius, maximum_radius]
                    if is_neq_row
                    else [-maximum_radius, -minimum_radius]
                )

                image = axis.imshow(
                    plot_data,
                    interpolation="nearest",
                    cmap="Reds",
                    extent=[
                        minimum_z,
                        maximum_z,
                        *radial_extents,
                    ],
                    origin="lower" if is_neq_row else "upper",
                    vmin=0.0,
                    vmax=1.0,
                    aspect="auto",
                )

            if is_neq_row:
                ion_label = rf"$\mathrm{{O}}^{{{ion[1:]}}}$"
                axis.text(
                    0.1,
                    0.8,
                    ion_label,
                    transform=axis.transAxes,
                    fontsize=14,
                    color="black",
                )

            if row == 0 and column == number_of_columns - 1:
                axis.text(
                    0.95,
                    0.82,
                    (
                        f"NEQ: {simulation_time.value:.3f} "
                        f"{simulation_time.unit}"
                    ),
                    transform=axis.transAxes,
                    horizontalalignment="right",
                    fontsize=9,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "black",
                        "boxstyle": "round,pad=0.3",
                    },
                )
            if row == 1 and column == number_of_columns - 1:
                axis.text(
                    0.95,
                    0.12,
                    "CIE from NEQ temperature",
                    transform=axis.transAxes,
                    horizontalalignment="right",
                    fontsize=9,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "black",
                        "boxstyle": "round,pad=0.3",
                    },
                )

            axis.tick_params(axis="both", which="major", labelsize=12)

    if image is None:
        raise NebulaError("No oxygen ion-fraction data were available to plot")

    colorbar_axis = figure.add_axes([0.125, 0.93, 0.775, 0.02])
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=MultipleLocator(0.2),
    )
    colorbar.set_label("Oxygen ion fraction")
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_directory / f"{neq_filebase}_oxygen_NEQ_CIE_fractions.png"
    )
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)

    logger.info("Saved oxygen NEQ-CIE comparison: %s", output_file)


if __name__ == "__main__":
    main()
