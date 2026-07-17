"""
Number Density Map for Fe
2D PION simulation:
Author: Arun Mathew
Date: 11 Dec 2025
"""

import numpy as np
import os
import time
from NebulaPy.tools import util
from NebulaPy.tools import constants as const
import NebulaPy.src as nebula
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")
cm2au = 6.68459e-14  # cm to au conversion factor

# Mimir -> Set up paths and filenames
OutputDir = '/Users/tony/Desktop/CWBs-NEMOv1/Silo-n128'  # Output image directory
SiloDir = '/mnt/massive-stars/data/wr140-nemo/wr140_NEMO_d07e13_d2l6n128'  # Directory containing silo files
Filebase = 'wr140_NEMO_d07e13_d2l6n128'  # Base name of the silo files
start_time = 1.0e6  # in sec
finish_time = None
time_unit = 'sec'
out_frequency = None
ion_list = ['Fe24+', 'Fe25+', 'Fe26+']

#Razer Blade
#OutputDir = '/home/tony/Desktop/multi-ion-bowshock/sim-output/coolmap'  # Output image directory
# Set up paths and filenames
#SiloDir = '/home/tony/Desktop/multi-ion-bowshock/high-res-silos-200kyr'  # Directory containing silo files
#Filebase = 'Ostar_mhd-nemo-dep_d2n0384l3'  # Base name of the silo files
#start_time = None
#finish_time = None
#time_unit = None
#out_frequency = None
#ion_list = ['Fe25+', 'Fe26+']

# Batch the silo files according to the time instant
batched_silos = nebula.Silo.batch(
    SiloDir,
    Filebase,
    start_time=start_time,
    finish_time=finish_time,
    time_unit=time_unit,
    out_frequency=out_frequency
)

# Initialize the Pion class from NebulaPy, which handles the simulation data
pion = nebula.pion(batched_silos, verbose=True)

# Calculates and stores geometric grid parameters.
# For example, in a spherical geometry, it extracts radius and shell volumes
# from the first silo file in the batch and saves them into a geometry container.
pion.load_geometry(scale='cm')
N_grid_level = pion.geometry_container['Nlevel']
mesh_edges_min = pion.geometry_container['edges_min'] * cm2au
mesh_edges_max = pion.geometry_container['edges_max'] * cm2au

print(mesh_edges_min)
print(mesh_edges_max)

# Extract all chemistry information from the silo files into a chemistry container
# This uses the first time instant's silo file to initialize
pion.load_chemistry()

runtime = 0.0
# Loop over each time instant in the batched silo files
for step, silo_instant in enumerate(batched_silos):
    silo_instant_start_time = time.time()

    print(f" ---------------------------")
    sim_time = pion.get_simulation_time(silo_instant, time_unit='kyr')
    print(f" step: {step} | simulation time: {sim_time:.6e}")

    # rows = hemispheres (upper, lower), cols = ions
    fig, axes = plt.subplots(
        2, len(ion_list),
        figsize=(4.0, 3.45),
        sharex=True,
        sharey=False
    )

    last_image = None
    v_min, v_max = 0.0, 4.0

    xlim = (-4.7, 1.34)
    ylim = 8.0

    for j, ion in enumerate(ion_list):
        n_ion = pion.get_ion_number_density(ion, silo_instant)

        axU = axes[0, j]  # upper hemisphere
        axL = axes[1, j]  # lower hemisphere (mirror)

        # Axis limits
        axU.set_xlim(*xlim)
        axL.set_xlim(*xlim)
        axU.set_ylim(0.0, ylim)
        axL.set_ylim(-ylim, 0.0)

        axU.tick_params(labelsize=11)
        axL.tick_params(labelsize=11)

        for level in range(N_grid_level):
            plot_data = np.log10(n_ion[level] + 1e-99)

            x_min = mesh_edges_min[level][0].value
            x_max = mesh_edges_max[level][0].value
            y_min = mesh_edges_min[level][1].value
            y_max = mesh_edges_max[level][1].value

            # ---- Upper hemisphere ----
            extU = [x_min, x_max, y_min, y_max]
            last_image = axU.imshow(
                plot_data,
                interpolation='nearest',
                cmap='plasma',
                extent=extU,
                origin='lower',
                vmin=v_min, vmax=v_max
            )

            # ---- Lower hemisphere (mirrored) ----
            plot_data_mirror = np.flipud(plot_data)
            extL = [x_min, x_max, -y_max, -y_min]
            last_image = axL.imshow(
                plot_data_mirror,
                interpolation='nearest',
                cmap='plasma',
                extent=extL,
                origin='lower',
                vmin=v_min, vmax=v_max
            )

        # Column (ion name)
        axU.text(0.1, 0.85, f"log n({ion})", transform=axU.transAxes, fontsize=10, color='white')

        # X labels
        axU.get_xaxis().set_visible(False)
        axL.set_xlabel(f'z (au)', fontsize=10)

        # Y labels
        if j == 0:
            axU.set_ylabel(f'R (au)', fontsize=10)
            axL.set_ylabel(f'R (au)', fontsize=10)
        else:
            axU.get_yaxis().set_visible(False)
            axL.get_yaxis().set_visible(False)


    # Shared colorbar
    cbar_ax = fig.add_axes([0.125, 0.97, 0.775, 0.03])
    fig.colorbar(last_image, cax=cbar_ax, orientation='horizontal')

    plt.subplots_adjust(hspace=0.0, wspace=0.0, top=0.90)

    Filename = f"{Filebase}_FeNumDen_{sim_time.value:.2f}kyr.png"
    plt.savefig(os.path.join(OutputDir, Filename), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f" time: {sim_time:.6e}, Saved snapshot {step} to {Filename}")

    dt = time.time() - silo_instant_start_time
    runtime += dt
    print(f" runtime: {runtime:.4e} s | dt: {dt:.4e} s")



