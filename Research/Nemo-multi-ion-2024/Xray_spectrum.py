import NebulaPy.src as nebula
from NebulaPy.tools import util
import time
from pypion.ReadData import ReadData
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as unit
import os
from NebulaPy.tools import constants as const
import pandas as pd

# MIMIR: Set up paths and filenames
#silo_dir = '/mnt/massive-stars/data/arun_simulations/wind-wind-equi/'  # Directory containing silo files
#output_path = '/net/maedoc.ap.dias.ie/maedoc/home_cr/arun/Desktop/plots/'  # Output file for results

# RAZERBLADE: Set up paths and filenames
silo_dir = '/home/tony/Desktop/Equi_NonEqui/nonequi-equi/'  # Directory containing silo files
output_path = '/home/tony/Desktop/Equi_NonEqui/nonequi-equi/'  # Output file for results

filebase = 'e7_WRwind_d1l5n256_v1500'  # Base name of the silo files
# Set up paths and filenames

# Batch the silo files according to the time instant
batched_silos = util.batch_silos(silo_dir, filebase)

# Initialize the Pion class from NebulaPy, which handles the simulation data
pion = nebula.pion(batched_silos, verbose=True)

# Extract all chemistry information from the silo files into a chemistry container
# This uses the first time instant's silo file to initialize
pion.load_chemistry()

# Initialize spherical grid parameters (e.g., radius, shell volumes)
# This sets up the grid using the first silo file in the batch
pion.spherical_grid(batched_silos[0])

# Retrieve the radius and shell volumes from the geometry container
radius = pion.geometry_container['radius']
shell_volume_original = pion.geometry_container['shell_volumes']
elements = pion.get_elements()

xray_emission = nebula.xray(
    min_photon_energy=0.5,  # Minimum photon energy in keV
    max_photon_energy=10.0,  # Maximum photon energy in keV
    energy_point_count=1000,
    elements=elements,
    bremsstrahlung=False,
    freebound=True,
    lines=False,
    twophoton=False,
    multiprocessing=True,
    ncores=12,
    verbose=True
)

runtime = 0.0
# Loop over each time instant in the batched silo files
for step, silo_instant in enumerate(batched_silos):
    silo_instant_start_time = time.time()  # Record the start time

    # Read the data from the current silo file
    dataio = ReadData(silo_instant)
    basic = dataio.get_1Darray('Density')  # Retrieve basic simulation data, such as density
    sim_time = (basic['sim_time'] * unit.s).to(unit.kyr)  # Convert simulation time to kiloyears
    dataio.close()  # Close the data file

    # Print the current time instant
    print(f" ---------------------------")
    print(f" step: {step} | simulation time: {sim_time:.6e}")

    temperature = pion.get_parameter('Temperature', silo_instant)
    density = pion.get_parameter('Density', silo_instant)
    ne = pion.get_ne(silo_instant)
    elemental_mass_fraction = pion.get_elemental_mass_frac(silo_instant)
    ion_fractions = pion.get_tracer_values(silo_instant)

    Tmin = 1e+5
    Tmax = 1e+9

    ##
    #hack
    relevant_grid_points = np.where((temperature >= Tmin) & (temperature <= Tmax))[0]
    temperature = temperature[relevant_grid_points]
    density = density[relevant_grid_points]
    ne = ne[relevant_grid_points]
    shell_volume = shell_volume_original[relevant_grid_points]
    filtered_elem_mass_frac = []
    for e in range(len(elemental_mass_fraction)):
        # Ensure the slice operation is valid for each elemental_mass_fraction[e]
        filtered_elem_mass_frac.append(elemental_mass_fraction[e][relevant_grid_points])

    filtered_ion_frac = []
    for e in range(len(ion_fractions)):
        element_wise = []
        for i in range(len(ion_fractions[e])):
            element_wise.append(ion_fractions[e][i][relevant_grid_points])
        filtered_ion_frac.append(element_wise)
    ##

    dem = pion.generate_dem_indices(temperature=temperature, Tmin=Tmin, Tmax=Tmax, Nbins=100)
    dem_indices = dem['indices']

    spectrum = xray_emission.xray_intensity(
        temperature=temperature,
        density=density, ne=ne,
        elemental_abundances=filtered_elem_mass_frac,
        ion_fractions=filtered_ion_frac,
        shell_volume=shell_volume,
        dem_indices=dem_indices
    )


    silo_instant_finish_time = time.time()  # Record the finish time
    # Calculate the time spent on the current step
    dt = silo_instant_finish_time - silo_instant_start_time
    # Update the runtime with the time spent on the current step
    runtime += dt
    print(f" runtime: {runtime:.4e} s | dt: {dt:.4e} s")

    # calculate differential emission measure
    DEM = pion.DEM(
        dem_indices=dem_indices,
        ne=ne,
        shellvolume=shell_volume,
    )

    # making DEM plot
    dem_filename = filebase + f"_t{int(sim_time.value)}_dem.png"
    dem_file = os.path.join(output_path, dem_filename)
    plt.plot(dem['Tb'], np.log10(DEM), linestyle='-', color='b', label=f'time = {sim_time.value:.4e} kyr')
    plt.xlabel(r'log(T$_b$) K', fontsize=14)
    plt.ylabel(r'log(DEM) cm$^{-3}$', fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.savefig(dem_file)  # Save as a PNG file
    plt.close()  # Close the plot to free memory


    # making plot
    generated_wvl_array = xray_emission.xray_containter['wvl_array']
    d = 10 * 3.086e+18 # cm
    xray_spectrum = 4.0 * const.pi * np.sum(spectrum, axis=0) / (d*d)

    plt.figure(figsize=(8, 6))  # Set the figure size
    # Format and replace '.' with 'p' to avoid issues in filenames
    out_filename = filebase + f"_t{int(sim_time.value)}_fb.png"
    out_file = os.path.join(output_path, out_filename)

    # Plot the spectrum with the corresponding temperature
    plt.plot(generated_wvl_array, xray_spectrum, linestyle='-', color='b', label=f'time = {sim_time.value:.4e} kyr')
    plt.xlabel(r'$\lambda \, (\AA)$', fontsize=14)
    plt.ylabel(r'$F_{\lambda} \, (egr \, cm^{-2} \, s^{-1} \, \AA^{-1}$)', fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.savefig(out_file)  # Save as a PNG file
    plt.close()

    # Create a DataFrame with wavelength (x) and spectrum (y)
    df = pd.DataFrame({
        'Wavelength': generated_wvl_array,
        'Spectrum': xray_spectrum
    })

    # Specify the output path and filename
    filename = os.path.join(output_path, f'xray_spectrum_t{sim_time.value:.4e}_fb.txt')
    # Save the DataFrame to a text file
    print(f"Saving X-ray spectrum into file")
    df.to_csv(filename, sep='\t', index=False)


















