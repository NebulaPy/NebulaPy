"""
2D Emissivity Map Generator FOR Hydrogen Lines
Description:
    This script produces two-dimensional emissivity maps for hydrogen emission
    lines, covering both collisionally de-excited transitions and recombination
    lines. It processes data from 2D PION simulation SILO files, which encode
    the physical conditions across the computational domain.

    Outputs include:
    HDF5 (.h5) files with the computed emissivity data for further analysis.
    PNG figures that provide a visual representation of the spatial distribution
    of the line emissivities.

Author: Arun Mathew
Date: 19 Sep 2025

About Hα Line:
    Hα is usually quoted as a single line at 6562.8 Å in air (6564.6 Å in vacuum),
    the transition actually consists of several closely spaced subcomponents arising
    from different sub-levels (e.g., 3s, 3p, 3d → 2s, 2p). These small wavelength
    differences reflect the effects of fine-structure splitting in the hydrogen atom.
"""

import os  # File system operations
import time  # For tracking runtime
import warnings  # Suppress specific runtime warnings
import h5py  # For saving data in HDF5 format
import numpy as np  # Numerical array operations
import matplotlib.pyplot as plt  # Plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For attaching colorbars to axes
from matplotlib.ticker import MultipleLocator, ScalarFormatter  # For controlling tick formatting
import astropy.units as unit  # Astropy units for physical quantities
import NebulaPy.src as nebula  # NebulaPy simulation interface
from NebulaPy.tools import util  # Utility functions
from pypion.ReadData import ReadData  # Silo data reader

# --- Suppress warnings for log10 of zero ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")

# --- File and Directory Configuration on Macbook ---
output_dir = '/home/tony/Desktop/multi-ion-bowshock/sim-output/emiss_map_new'
silo_dir = '/home/tony/Desktop/multi-ion-bowshock/high-res-silos-200kyr'
filebase = 'Ostar_mhd-nemo-dep_d2n0384l3'
start_time = None # kyr
finish_time = None  # kyr
out_frequency = None  # Use all available outputs

# --- File and Directory Configuration on Mimir ---
output_dir = '/mnt/massive-stars/data/arun_simulations/Nemo_BowShock/high-res/emissivity-map'
silo_dir = '/mnt/massive-stars/data/arun_simulations/Nemo_BowShock/high-res/silo'
filebase = 'Ostar_mhd-nemo-dep_d2n0384l3'
start_time = 200 # kyr
finish_time = 201  # kyr
out_frequency = None  # Use all available outputs

# PION Ion Name
pion_ion = 'H'

print(rf" task: calculating relevant H lines emissivity map")

# Chianti H-alpha lines 6564.538, 6564.564, 6564.523, 6564.665, 6564.722 (Å)
# and the repeated entries near 6564 Å correspond to the fine-structure
# components of the hydrogen Hα line (Balmer α, 3 → 2 transition).

# collisional lines
Chianti_HAlphaLines = [6564.538, 6564.564, 6564.523, 6564.665, 6564.722]  # in Angstroms
print(f" Hα collisional de-excited chianti lines: {', '.join(map(str, Chianti_HAlphaLines))}  \u212B")
Chianti_HBetaLines = [4862.733, 4862.72]  # in Angstroms
print(f" Hβ collisional de-excited chianti lines: {', '.join(map(str, Chianti_HBetaLines))}  \u212B")
Chianti_HLyAlphaLines = [1215.674, 1215.668]  # in Angstroms
print(f" H Lyman α collisional de-excited chianti lines: {', '.join(map(str, Chianti_HLyAlphaLines))}  \u212B")

# recombination lines
PyNeb_HAlphaLines = [6562.816]
print(f" Hα recombination pyneb lines: {', '.join(map(str, PyNeb_HAlphaLines))}  \u212B")
PyNeb_HBetaLines = [4861.332]
print(f" Hβ recombination pyneb lines: {', '.join(map(str, PyNeb_HBetaLines))}  \u212B")
PyNeb_HPaAlphaLines = [18750.960]
print(f" H Paschen α recombination pyneb lines: {', '.join(map(str, PyNeb_HPaAlphaLines))}  \u212B")
PyNeb_HBrGammaLines = [21655.283]
print(f" H Brackett gamma recombination pyneb lines: {', '.join(map(str, PyNeb_HBrGammaLines))}  \u212B")
PyNeb_HLyAlphaLines = [1215.670]
print(f" H Lyman α recombination pyneb lines: {', '.join(map(str, PyNeb_HLyAlphaLines))}  \u212B")

IncludeCollisionalDeExcitation = True
IncludeRecombination = True

# Batching silo files based on time range
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

# Initialize the PION class to handle simulation data
pion = nebula.pion(batched_silos, verbose=True)

# Calculates and stores geometric grid parameters.
# For example, in a spherical geometry, it extracts radius and shell volumes
# from the first silo file in the batch and saves them into a geometry container.
pion.load_geometry(scale='pc')
N_grid_level = pion.geometry_container['Nlevel']
mesh_edges_min = pion.geometry_container['edges_min']
mesh_edges_max = pion.geometry_container['edges_max']

grid_mask = pion.geometry_container['mask']
cell_volume = pion.get_cylindrical_cell_volume().value

# h5 header and metadata
heading = f'Generated by {util.nebula_version()}'
metadata = {'minimum_mesh_edges': mesh_edges_min,
            'maximum_mesh_edges': mesh_edges_max,
            'Nlevels': N_grid_level,
            'grid_mask': grid_mask,
            'cell_volume': cell_volume}

# Extract all chemistry information from the silo files into a chemistry container
# This uses the first time instant's silo file to initialize
pion.load_chemistry()

electron_tolerance = 1.E-08  # Floor value for electron density
ion_name = pion_ion.replace('+', 'p')
ion_output_dir = os.path.join(output_dir, ion_name)
os.makedirs(ion_output_dir, exist_ok=True)

data_title = f"Relevant H lines emissivity map"

# Initialize Line Emission Module
line_emission = nebula.line_emission(pion_ion, verbose=True)

# Loop Through Silo Time Instants
runtime = 0.0
for step, silo_instant in enumerate(batched_silos):
    silo_instant_start_time = time.time()

    # --- Get Simulation Time ---
    sim_time = pion.get_simulation_time(silo_instant, time_unit='kyr')
    print(f" ---------------------------")
    print(f" Step: {step}/{N_time_instant - 1} | Simulation Time: {sim_time:.6e} kyr")

    # Extract temperature and electron number density
    temperature = pion.get_parameter('Temperature', silo_instant)
    ne = pion.get_ne(silo_instant)

    # h5 metadata
    metadata['step'] = step
    metadata['simulation-time'] = sim_time.value
    metadata['time-unit'] = str(sim_time.unit)

    # Retrieve Physical Parameters
    temperature = pion.get_parameter('Temperature', silo_instant)
    ne = pion.get_ne(silo_instant)

    # Create a zero emissivity map with the same shape as the temperature grids
    Halpha_coll_emissivity_map = np.array([np.zeros(arr.shape) for arr in temperature])
    # Create a zero emissivity map with the same shape as the temperature grids
    Halpha_recomb_emissivity_map = np.array([np.zeros(arr.shape) for arr in temperature])

    # Create a zero emissivity map with the same shape as the temperature grids
    Hbeta_coll_emissivity_map = np.array([np.zeros(arr.shape) for arr in temperature])
    # Create a zero emissivity map with the same shape as the temperature grids
    Hbeta_recomb_emissivity_map = np.array([np.zeros(arr.shape) for arr in temperature])

    Lyalpha_coll_emissivity_map = np.array([np.zeros(arr.shape) for arr in temperature])
    Lyalpha_recomb_emissivity_map = np.array([np.zeros(arr.shape) for arr in temperature])

    Paalpha_recomb_emissivity_map = np.array([np.zeros(arr.shape) for arr in temperature])
    Brgamma_recomb_emissivity_map = np.array([np.zeros(arr.shape) for arr in temperature])

    # Sum collisional emissivities
    if IncludeCollisionalDeExcitation:

        # get collisional Emissivity Maps for all Hα lines
        Halpha_coll_emiss_map_dict = line_emission.line_emissivity_2D_map(
            Chianti_HAlphaLines, temperature, ne, progress_bar=True)
        # Sum all line emissivity map for all Hα lines
        for line in Halpha_coll_emiss_map_dict:
            for level in range(N_grid_level):
                Halpha_coll_emissivity_map[level] += Halpha_coll_emiss_map_dict[line][level]
        del Halpha_coll_emiss_map_dict

        # get collisional Emissivity Maps for all Hβ lines
        Hbeta_coll_emiss_map_dict = line_emission.line_emissivity_2D_map(
            Chianti_HBetaLines, temperature, ne, progress_bar=True)
        # Sum all line emissivity map for all Hβ lines
        for line in Hbeta_coll_emiss_map_dict:
            for level in range(N_grid_level):
                Hbeta_coll_emissivity_map[level] += Hbeta_coll_emiss_map_dict[line][level]
        del Hbeta_coll_emiss_map_dict

        # get collisional Emissivity Maps for all Lyman-α lines
        Lyalpha_coll_emiss_map_dict = line_emission.line_emissivity_2D_map(
            Chianti_HLyAlphaLines, temperature, ne, progress_bar=True)
        # Sum all line emissivity map for all Lyman-α lines
        for line in Lyalpha_coll_emiss_map_dict:
            for level in range(N_grid_level):
                Lyalpha_coll_emissivity_map[level] += Lyalpha_coll_emiss_map_dict[line][level]
        del Lyalpha_coll_emiss_map_dict


    # Sum recombination emissivities
    if IncludeRecombination:

        # get recombination Emissivity Maps for all Hα lines
        Halpha_recomb_emiss_map_dict = line_emission.recombination_line_emissivity_2D_map(
            PyNeb_HAlphaLines, temperature, ne, progress_bar=True)
        # Sum all line emissivity map for all Hα lines
        for line in Halpha_recomb_emiss_map_dict:
            for level in range(N_grid_level):
                Halpha_recomb_emissivity_map[level] += Halpha_recomb_emiss_map_dict[line][level]
        del Halpha_recomb_emiss_map_dict

        # get recombination Emissivity Maps for all Hβ lines
        Hbeta_recomb_emiss_map_dict = line_emission.recombination_line_emissivity_2D_map(
            PyNeb_HBetaLines, temperature, ne, progress_bar=True)
        # Sum all line emissivity map for all Hβ lines
        for line in Hbeta_recomb_emiss_map_dict:
            for level in range(N_grid_level):
                Hbeta_recomb_emissivity_map[level] += Hbeta_recomb_emiss_map_dict[line][level]
        del Hbeta_recomb_emiss_map_dict

        # get recombination Emissivity Maps for Lyman alpha lines
        Lyalpha_recomb_emiss_map_dict = line_emission.recombination_line_emissivity_2D_map(
            PyNeb_HLyAlphaLines, temperature, ne, progress_bar=True)
        # Sum all line emissivity map for Lyman lines
        for line in Lyalpha_recomb_emiss_map_dict:
            for level in range(N_grid_level):
                Lyalpha_recomb_emissivity_map[level] += Lyalpha_recomb_emiss_map_dict[line][level]
        del Lyalpha_recomb_emiss_map_dict

        # get recombination Emissivity Maps for all Paschen α lines
        Paalpha_recomb_emiss_map_dict = line_emission.recombination_line_emissivity_2D_map(
            PyNeb_HPaAlphaLines, temperature, ne, progress_bar=True)
        # Sum all line emissivity map for all Paα lines
        for line in Paalpha_recomb_emiss_map_dict:
            for level in range(N_grid_level):
                Paalpha_recomb_emissivity_map[level] += Paalpha_recomb_emiss_map_dict[line][level]
        del Paalpha_recomb_emiss_map_dict

        # get recombination Emissivity Maps for all Brackett γ lines
        Brgamma_recomb_emiss_map_dict = line_emission.recombination_line_emissivity_2D_map(
            PyNeb_HBrGammaLines, temperature, ne, progress_bar=True)
        # Sum all line emissivity map for all Hβ lines
        for line in Brgamma_recomb_emiss_map_dict:
            for level in range(N_grid_level):
                Brgamma_recomb_emissivity_map[level] += Brgamma_recomb_emiss_map_dict[line][level]
        del Brgamma_recomb_emiss_map_dict

    # Store the total in a dictionary with key "Halpha_coll"
    Halpha_coll_emiss_map_main_dict = {'H I 6564.': Halpha_coll_emissivity_map}
    # Store the total in a dictionary with key "Halpha_recomb"
    Halpha_recomb_emiss_map_main_dict = {'H I 6562.816': Halpha_recomb_emissivity_map}

    # Store the total in a dictionary with key "Hbeta_coll"
    Hbeta_coll_emiss_map_main_dict = {'H I 4862.': Hbeta_coll_emissivity_map}
    # Store the total in a dictionary with key "Hbeta_recomb"
    Hbeta_recomb_emiss_map_main_dict = {'H I 4861.332': Hbeta_recomb_emissivity_map}

    Lyalpha_coll_emiss_map_dict = {'H I 1215.': Lyalpha_coll_emissivity_map}
    Lyalpha_recomb_emiss_map_dict = {'H I 1215.670': Lyalpha_recomb_emissivity_map}

    Paalpha_recomb_emiss_map_dict = {'H I 18750.960': Paalpha_recomb_emissivity_map}
    Brgamma_recomb_emiss_map_dict = {'H I 21655.283': Brgamma_recomb_emissivity_map}

    # get neutral hydrogen number density
    HIden = pion.get_ion_number_density(pion_ion, silo_instant)
    # get ionised hydrogen number density
    HIIden = pion.get_ion_number_density('H1+', silo_instant)

    # --- Save HAlpha Emissivity Maps to HDF5 ---
    h5_filename = f"{filebase}_emiss_{ion_name}_{str(step).zfill(4)}.h5"
    h5_file = os.path.join(ion_output_dir, h5_filename)

    print(f" saving H recombination line emissivity maps and H number density into {h5_filename}")
    with h5py.File(h5_file, "w") as file:
        # Add heading and title
        file.attrs['head'] = heading
        file.attrs['title'] = data_title

        # Save metadata
        meta = file.create_group("metadata")
        for key, value in metadata.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                meta.create_dataset(key, data=np.array(value))
            else:
                meta.attrs[key] = value

        # Save ion number density
        ion_number_density_group = file.create_group("ion_number_density")
        ion_number_density_group.create_dataset(str(pion_ion), data=np.array(HIden, dtype=np.float32))
        ion_number_density_group.create_dataset('H1+', data=np.array(HIIden, dtype=np.float32))

        # Save emissivity function map
        emissivity_map_group = file.create_group("emissivity_map")

        for key, value in Halpha_coll_emiss_map_main_dict.items():
            emissivity_map_group.create_dataset(str(key), data=np.array(value, dtype=np.float32))
        for key, value in Halpha_recomb_emiss_map_main_dict.items():
            emissivity_map_group.create_dataset(str(key), data=np.array(value, dtype=np.float32))

        for key, value in Hbeta_coll_emiss_map_main_dict.items():
            emissivity_map_group.create_dataset(str(key), data=np.array(value, dtype=np.float32))
        for key, value in Hbeta_recomb_emiss_map_main_dict.items():
            emissivity_map_group.create_dataset(str(key), data=np.array(value, dtype=np.float32))

        for key, value in Lyalpha_coll_emiss_map_dict.items():
            emissivity_map_group.create_dataset(str(key), data=np.array(value, dtype=np.float32))
        for key, value in Lyalpha_recomb_emiss_map_dict.items():
            emissivity_map_group.create_dataset(str(key), data=np.array(value, dtype=np.float32))

        for key, value in Paalpha_recomb_emiss_map_dict.items():
            emissivity_map_group.create_dataset(str(key), data=np.array(value, dtype=np.float32))
        for key, value in Brgamma_recomb_emiss_map_dict.items():
            emissivity_map_group.create_dataset(str(key), data=np.array(value, dtype=np.float32))


    '''
    # --- Generate and Save Emissivity Map Plots ---
    for line in Halpha_emiss_map_dict:
        line_name = line.replace(' ', '')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.05, 0.9, f'time = {sim_time.value:5.2f} kyr', transform=ax.transAxes, fontsize=12, color='white')

        ax.set_xlim(mesh_edges_min[0][0].value, mesh_edges_max[0][0].value)
        ax.set_ylim(mesh_edges_min[0][1].value, mesh_edges_max[0][1].value)

        for level in range(N_grid_level):

            plot_data = np.log10(Halpha_emiss_map_dict[line][level]*HIIden[level])
            extents = [
                mesh_edges_min[level][0].value, mesh_edges_max[level][0].value,
                mesh_edges_min[level][1].value, mesh_edges_max[level][1].value
            ]
            image = ax.imshow(plot_data, interpolation='nearest', cmap='inferno',
                              extent=extents, origin='lower', vmin=-27, vmax=-22)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = plt.colorbar(image, cax=cax, ticks=MultipleLocator(1))
        colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())
        colorbar.ax.yaxis.get_major_formatter().set_scientific(False)
        colorbar.ax.yaxis.get_major_formatter().set_useOffset(False)

        ax.set_xlabel('z (cm)', fontsize=12)
        ax.set_ylabel('R (cm)', fontsize=12)
        ax.text(0.65, 0.9, line, transform=ax.transAxes, fontsize=12, color='white')
        ax.tick_params(axis='both', which='major', labelsize=13)

        filename = f"{filebase}_emiss_{line_name}_{sim_time.value:.2f}kyr.png"
        filepath = os.path.join(ion_output_dir, filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f" saving {line} emissivity map to {filename}")
        '''

    # Track Runtime
    dt = time.time() - silo_instant_start_time
    runtime += dt
    print(f" Runtime: {runtime:.4e} s | Δt: {dt:.4e} s")
