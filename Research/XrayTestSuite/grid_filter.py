"""Plot temperature-based PION grid filtering for visual inspection.

For every selected Silo snapshot, this script compares the original PION grid
mask with the mask returned by ``pion.apply_grid_filters``.  The temperature
map is included so that the selected cells can be checked against the requested
temperature interval.

Author: Arun Mathew
Date: 20 Aug 2026
"""

import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import NebulaPy.src as nebula


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="divide by zero encountered in log10",
)


# -----------------------------------------------------------------------------
# File and directory configuration
# -----------------------------------------------------------------------------
# Colliding wind binaries
#Macbook -> Set up paths and filenames
OutputDir = '/Users/tony/Desktop/CWBs-NEMOv1/Post-Processing/WR140Test'  # Output image directory
SiloDir = '/Users/tony/Desktop/CWBs-NEMOv1/Silo-n128'  # Directory containing silo files
Filebase = 'wr140_NEMO_d07e13_d2l6n128'  # Base name of the silo files
start_time = 14.35  # days
finish_time = None
time_unit = 'days'
out_frequency = None
SimulationName = "WR140"


# -----------------------------------------------------------------------------
# Optional temperature filters in K
# Set either value to None to leave that side of the interval unrestricted.
# Set both values to None to recover the original grid mask.
# -----------------------------------------------------------------------------
min_temperature = 1.0e2
max_temperature = 1.0e9


def add_colorbar(fig, ax, image, label):
    """Attach a colorbar to one map axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(label)


def main():
    """Plot the original and temperature-filtered mask for each snapshot."""
    print("Calculating temperature-filtered grid-mask maps")

    batched_silos = nebula.Silo.batch(
        SiloDir,
        Filebase,
        start_time=start_time,
        finish_time=finish_time,
        time_unit=time_unit,
        out_frequency=out_frequency,
    )

    number_of_snapshots = len(batched_silos)
    if number_of_snapshots == 0:
        raise RuntimeError("No Silo snapshots matched the requested selection.")

    pion = nebula.pion(batched_silos, progress=True)
    pion.load_geometry(scale="pc")

    geometry = pion.geometry_container
    number_of_levels = geometry["Nlevel"]
    mesh_edges_min = geometry["edges_min"]
    mesh_edges_max = geometry["edges_max"]

    os.makedirs(OutputDir, exist_ok=True)

    runtime = 0.0
    for step, silo_instant in enumerate(batched_silos):
        snapshot_start_time = time.time()

        sim_time = pion.get_simulation_time(
            silo_instant,
            time_unit=time_unit,
        )
        print(
            f"Step: {step}/{number_of_snapshots - 1} | "
            f"Simulation Time: {sim_time:.6e}"
        )

        temperature = np.asarray(
            pion.get_parameter("Temperature", silo_instant),
            dtype=np.float64,
        )

        # Calling without filters returns a fresh copy of the snapshot's NG_Mask.
        original_mask = pion.apply_grid_filters(silo_instant)

        # This call independently reloads NG_Mask before applying the temperature
        # bounds, so filtering cannot accumulate between snapshots.
        filtered_mask = pion.apply_grid_filters(
            silo_instant,
            temperature=temperature,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
        )

        original_cells = np.count_nonzero(original_mask)
        selected_cells = np.count_nonzero(filtered_mask)
        selected_fraction = (
            selected_cells / original_cells
            if original_cells > 0
            else 0.0
        )
        print(
            f"Selected cells: {selected_cells}/{original_cells} "
            f"({100.0 * selected_fraction:.2f}%)"
        )

        active_temperatures = temperature[original_mask != 0]
        positive_temperatures = active_temperatures[active_temperatures > 0.0]
        if positive_temperatures.size == 0:
            raise RuntimeError(
                f"Snapshot {silo_instant} contains no positive temperatures "
                "inside the original grid mask."
            )
        temperature_normalization = Normalize(
            vmin=np.log10(np.nanmin(positive_temperatures)),
            vmax=np.log10(np.nanmax(positive_temperatures)),
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.25))
        titles = (
            "Original grid mask",
            r"Temperature $\log_{10}(T/\mathrm{K})$",
            "Temperature-filtered mask",
        )

        for ax, title in zip(axes, titles):
            ax.set_title(title)
            ax.set_xlim(mesh_edges_min[0][0].value, mesh_edges_max[0][0].value)
            ax.set_ylim(mesh_edges_min[0][1].value, mesh_edges_max[0][1].value)
            ax.set_xlabel("z (pc)")
            ax.set_ylabel("R (pc)")
            ax.set_aspect("equal", adjustable="box", anchor="N")
            ax.tick_params(axis="both", which="major", labelsize=11)

        original_image = None
        temperature_image = None
        filtered_image = None

        for level in range(number_of_levels):
            extents = [
                mesh_edges_min[level][0].value,
                mesh_edges_max[level][0].value,
                mesh_edges_min[level][1].value,
                mesh_edges_max[level][1].value,
            ]

            original_image = axes[0].imshow(
                original_mask[level],
                interpolation="nearest",
                cmap="Greys",
                norm=Normalize(vmin=0.0, vmax=1.0),
                extent=extents,
                origin="lower",
            )

            log_temperature = np.ma.masked_where(
                (original_mask[level] == 0)
                | (temperature[level] <= 0.0),
                np.log10(temperature[level]),
            )
            temperature_image = axes[1].imshow(
                log_temperature,
                interpolation="nearest",
                cmap="inferno",
                norm=temperature_normalization,
                extent=extents,
                origin="lower",
            )

            filtered_image = axes[2].imshow(
                filtered_mask[level],
                interpolation="nearest",
                cmap="Greys",
                norm=Normalize(vmin=0.0, vmax=1.0),
                extent=extents,
                origin="lower",
            )

        add_colorbar(fig, axes[0], original_image, "Mask value")
        add_colorbar(fig, axes[1], temperature_image, r"$\log_{10}(T/\mathrm{K})$")
        add_colorbar(fig, axes[2], filtered_image, "Mask value")

        lower_label = "None" if min_temperature is None else f"{min_temperature:.2e} K"
        upper_label = "None" if max_temperature is None else f"{max_temperature:.2e} K"
        fig.suptitle(
            f"time = {sim_time.value:.2f} {time_unit} | "
            f"temperature filter = [{lower_label}, {upper_label}] | "
            f"selected = {100.0 * selected_fraction:.2f}%",
            fontsize=13,
            y=0.975,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), w_pad=1.5)
        fig.subplots_adjust(top=0.90)

        filename = (
            f"{Filebase}_grid-filter_"
            f"{sim_time.value:.2f}{time_unit}.png"
        )
        filepath = os.path.join(OutputDir, filename)
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        plt.close(fig)

        elapsed = time.time() - snapshot_start_time
        runtime += elapsed
        print(f"Saved: {filepath}")
        print(f"Runtime: {runtime:.4e} s | dt: {elapsed:.4e} s")


if __name__ == "__main__":
    main()
