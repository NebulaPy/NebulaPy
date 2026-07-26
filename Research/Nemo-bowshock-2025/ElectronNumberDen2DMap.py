"""
2D Electron Number Density Map Generator from PION Silo Files
Description:
    This script generates 2D emissivity maps for electron number
    density using data from 2D PION simulation silo files.
    The output plots for visual inspection.

Author: Arun Mathew
Date: 07 Sep 2025
"""

# --- Import Required Libraries ---
import os  # File system operations
import time  # For tracking runtime
import warnings  # Suppress specific runtime warnings
import numpy as np  # Numerical array operations
import matplotlib.pyplot as plt  # Plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For attaching colorbars to axes
from matplotlib.ticker import MultipleLocator, ScalarFormatter  # For controlling tick formatting

# --- Import Project Modules ---
import NebulaPy.src as nebula  # NebulaPy simulation interface
from NebulaPy.tools import util  # Utility functions
from pypion.ReadData import ReadData  # Silo data reader

# --- Suppress warnings for log10 of zero ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")

# --- File and Directory Configuration ---
output_dir = '/home/tony/Desktop/multi-ion-bowshock/sim-output/ne_map'
silo_dir = '/home/tony/Desktop/multi-ion-bowshock/high-res-silos-200kyr'
filebase = 'Ostar_mhd-nemo-dep_d2n0384l3'

# --- Simulation Time Parameters ---
start_time = None  # kyr
finish_time = None  # kyr
out_frequency = None  # Use all available outputs

print(rf" calculating electron number denisty map")

# --- Get List of Silo Files Based on Time Constraints ---
batched_silos = util.batch_silos(
    silo_dir,
    filebase,
    start_time=start_time,
    finish_time=finish_time,
    time_unit='kyr',
    out_frequency=out_frequency
)

# Total number of time instant
N_time_instant = len(batched_silos)

# --- Load Simulation Data ---
pion = nebula.pion(batched_silos, verbose=True)
pion.load_chemistry()  # Load ion and reaction network data
pion.load_geometry(scale='pc')  # Load spatial grid configuration

# --- Extract Geometry Info ---
geometry = pion.geometry_container
coordinate_sys = geometry['coordinate_sys']
N_grid_level = geometry['Nlevel']
mesh_edges_min = geometry['edges_min']
mesh_edges_max = geometry['edges_max']

# --- Prepare Output Path ---
electron_tolerance = 1.E-08  # Floor value for electron density
os.makedirs(output_dir, exist_ok=True)


# --- Loop Through Time Steps ---
runtime = 0.0
for step, silo_instant in enumerate(batched_silos):
    silo_instant_start_time = time.time()

    # --- Get Simulation Time ---
    sim_time = pion.get_simulation_time(silo_instant, time_unit='kyr')
    print(f" ---------------------------")
    print(f" Step: {step}/{N_time_instant - 1} | Simulation Time: {sim_time:.6e} kyr")

    # --- Retrieve electron number density ---
    ne = pion.get_ne(silo_instant)

    # --- Generate electron number density Map Plots ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.05, 0.9, f'time = {sim_time.value:5.2f} kyr', transform=ax.transAxes, fontsize=12, color='white')

    ax.set_xlim(mesh_edges_min[0][0].value, mesh_edges_max[0][0].value)
    ax.set_ylim(mesh_edges_min[0][1].value, mesh_edges_max[0][1].value)

    for level in range(N_grid_level):
        plot_data = np.log10(ne[level])
        extents = [
            mesh_edges_min[level][0].value, mesh_edges_max[level][0].value,
            mesh_edges_min[level][1].value, mesh_edges_max[level][1].value
        ]
        image = ax.imshow(plot_data, interpolation='nearest', cmap='inferno',
                          extent=extents, origin='lower',
                          vmin=-3,
                          vmax=2
                          )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(image, cax=cax, ticks=MultipleLocator(1))
    colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    colorbar.ax.yaxis.get_major_formatter().set_scientific(False)
    colorbar.ax.yaxis.get_major_formatter().set_useOffset(False)

    ax.set_xlabel('z (pc)', fontsize=12)
    ax.set_ylabel('R (pc)', fontsize=12)
    ax.text(0.65, 0.9, r'Electron Density $n_e$', transform=ax.transAxes, fontsize=12, color='white')
    ax.tick_params(axis='both', which='major', labelsize=13)

    filename = f"{filebase}_nemap_{sim_time.value:.2f}kyr.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)


    # --- Track Runtime ---
    dt = time.time() - silo_instant_start_time
    runtime += dt
    print(f" Runtime: {runtime:.4e} s | Δt: {dt:.4e} s")
