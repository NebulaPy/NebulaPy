import NebulaPy.src as nebula
from NebulaPy.tools import util
import time
# from pypion.ReadData import ReadData
import matplotlib.pyplot as plt
import numpy as np
# import astropy.units as unit
import os
# from NebulaPy.tools import constants as const
# import pandas as pd
import ChiantiPy.tools.filters as chfilters
import matplotlib.pyplot as plt  # Plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For attaching colorbars to axes
from matplotlib.ticker import MultipleLocator, ScalarFormatter  # For controlling tick formatting
from scipy.ndimage import gaussian_filter1d

# constants
cm2au = 6.68459e-14  # cm to au conversion factor

# Macbook
#OutputDir = '/Users/tony/Desktop/CWBs-NEMOv1/Post-Processing/XraySpectrum'  # Output image directory

#Razer Blade -> Set up paths and filenames
OutputDir = '/home/tony/Desktop/CWBs-2026/Postprocessing/X-raySpectrum'  # Output image directory
SiloDir = '/home/tony/Desktop/CWBs-2026/Silo-n128'  # Directory containing silo files
Filebase = 'wr140_NEMO_d07e13_d2l6n128'  # Base name of the silo files
start_time = 1.24e6  # in sec
finish_time = None
time_unit = 'sec'
out_frequency = None


# Batch the silo files according to the time instant
batched_silos = nebula.Silo.batch(
    SiloDir,
    Filebase,
    start_time=start_time,
    finish_time=finish_time,
    time_unit=time_unit,
    out_frequency=out_frequency
)

key = input(" Press 'y' to continue, anything else to exit: ").strip().lower()

if key == "y":
    print(" Continuing execution...")
else:
    nebula.get_logger(__name__).info("Resetting parameters before the next run")
    exit(0)


# Initialize the Pion class from NebulaPy, which handles the simulation data
pion = nebula.pion(batched_silos, progress=True)

# load chemistry
pion.load_chemistry()

# Calculates and stores geometric grid parameters.
# For example, in a spherical geometry, it extracts radius and shell volumes
# from the first silo file in the batch and saves them into a geometry container.
pion.load_geometry(scale='cm')
N_grid_level = pion.geometry_container['Nlevel']
#mesh_edges_min = pion.geometry_container['edges_min'] * cm2au
#mesh_edges_max = pion.geometry_container['edges_max'] * cm2au
mesh_edges_min = pion.geometry_container['edges_min']
mesh_edges_max = pion.geometry_container['edges_max']
N_grid = pion.geometry_container['Ngrid']
cell_volume = pion.get_grid_volumes_2D()



EM = nebula.emissionMeasure(Tmin=100, Tmax=1.e9, Nbins=200, progress=True)

runtime = 0.0
# Loop over each time instant in the batched silo files
for step, silo_instant in enumerate(batched_silos):
    silo_instant_start_time = time.time()

    print(f" ---------------------------")
    sim_time = pion.get_simulation_time(silo_instant, time_unit='sec')
    print(f" step: {step} | simulation time: {sim_time:.6e}")

    # Extract temperature and electron number density
    temperature = pion.get_parameter('Temperature', silo_instant)
    density = pion.get_parameter('Density', silo_instant)
    ne = pion.get_ne(silo_instant)
    grid_mask = pion.geometry_container['mask']
    number_densities = pion.get_species_number_densities(silo_instant)

    EM.DEM2D(temperature=temperature, ne=ne,
             speciesDensities=number_densities,
             volume=cell_volume,
             gridMask=grid_mask)

    Bin_temperature = EM.Tb
    half_width = EM.half_bin_width

    SAMDEM = EM.SAM_DEM(density=density, temperature=temperature, ne=ne,
                        mask=grid_mask, ngrid=N_grid,
                        mesh_edges_min=mesh_edges_min, mesh_edges_max=mesh_edges_max,
                        volume=cell_volume, temp_bin=Bin_temperature,
                        hw=half_width
                        )

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.text(0.05, 0.90, f"time = {sim_time:5.2f}", fontsize=12, color='black')
    # Total DEM
    total_DEM = np.zeros_like(EM.Tb, dtype=np.float64)


    # Plot DEM for each species separately
    for species in number_densities:
        fig_species, ax_species = plt.subplots(figsize=(8, 6))

        fig_species.text(0.05, 0.90, f"time = {sim_time:5.2f}",
                         fontsize=12, color='black')

        species_dem = np.asarray(EM.DEM[species], dtype=np.float64)


        # Add unsmoothed DEM to total DEM
        total_DEM += species_dem

        # Smooth only for plotting
        #species_dem_smooth = gaussian_filter1d(species_dem, sigma=1.0)

        #valid = species_dem > 0.0

        ax_species.plot(
            EM.Tb,
            np.log10(species_dem),
            linewidth=1.5,
            color='black',
            label=rf'$\mathrm{{DEM}}_{{{species}}}$'
        )

        ax_species.set_title(f"{species} Differential Emission Measure")
        ax_species.set_xlabel(r'$\log(T/\mathrm{K})$', fontsize=12)
        ax_species.set_ylabel(r'$\log(\mathrm{DEM})$', fontsize=12)
        ax_species.legend(fontsize=10)


        species_label = species.replace('+', 'p')
        Filename = f"DEM_{species_label}_{sim_time.value:.2f}kyr.png"
        Filepath = os.path.join(OutputDir, Filename)
        plt.savefig(Filepath, bbox_inches="tight", dpi=300)
        print(f" Saving {species} differential emission measure to {Filename}")
        plt.close(fig_species)


    ax.set_title("       Total DEM of the Colliding-Wind Binary WR140")
    ax.plot(EM.Tb, np.log10(total_DEM), linewidth=1.5, color='black',
            label=r'$\sum_i \, \mathrm{DEM}_{X_i}$')
    ax.plot(EM.Tb, np.log10(SAMDEM), linewidth=1.5, color='red', label="SAM's DEM")
    ax.set_xlabel(r'$\log(T/\mathrm{K})$', fontsize=12)
    ax.set_ylabel(r'$\log(\mathrm{DEM})$', fontsize=12)
    ax.legend()
    Filename = f"{Filebase}_DEM_{sim_time.value:.2f}kyr.png"
    Filepath = os.path.join(OutputDir, Filename)
    plt.savefig(Filepath, bbox_inches="tight", dpi=300)
    print(f" Saving DEM comparison: SAM calculation vs. sum of all species DEMs to {Filename}")
    plt.close(fig)

    dt = time.time() - silo_instant_start_time
    runtime += dt
    print(f" runtime: {runtime:.4e} s | dt: {dt:.4e} s")
