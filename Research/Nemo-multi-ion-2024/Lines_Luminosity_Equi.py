import NebulaPy.src as nebula
from NebulaPy.tools import util
from pypion.ReadData import ReadData
import astropy.units as unit
import numpy as np
import time

# MIMIR: Set up paths and filenames
silo_dir = '/mnt/massive-stars/data/arun_simulations/wind-wind-equi/silo'  # Directory containing silo files
output_file = '/net/maedoc.ap.dias.ie/maedoc/home_cr/arun/Desktop/plots/line_luminosity_equi.txt'  # Output file for results

# RAZERBLADE: Set up paths and filenames
#silo_dir = '/home/tony/Desktop/OIV_Luminosity/mpv10-wind-test'  #Directory containing silo files
#output_file = '/home/tony/Desktop/OIV_Luminosity/lines_luminosity.txt'  # Output file for results

filebase = 'e7_WRwind_d1l5n256_v1500'  # Base name of the silo files

# Set up the ion and line emission parameters
O2P_pion_ion = 'O2+'  # The ion of interest (Oxygen IV)
mass_O = 2.6567628e-23  # Mass of an Oxygen atom (in grams)
O2P_lines = [4960.295, 5008.240]  # Emission line(s) of interest
print(rf" calculating line luminosity of {O2P_pion_ion} lines: {O2P_lines} Angstrom")

# Set up the ion and line emission parameters
O3P_pion_ion = 'O3+'  # The ion of interest (Oxygen IV)
O3P_lines = [2.589332e+05]  # Emission line of interest
print(rf" calculating line luminosity of {O3P_pion_ion} lines: {O3P_lines} Angstrom")

# Set up the ion and line emission parameters
O5P_pion_ion = 'O5+'  # The ion of interest (Oxygen IV)
O5P_lines = [1031.9120, 1037.6130]  # Emission line(s) of interest
print(rf" calculating line luminosity of {O5P_pion_ion} lines: {O5P_lines} Angstrom")

# Batch the silo files according to the time instant
start_time = 0.0
finish_time = 1000.0
batched_silos = util.batch_silos(
    silo_dir,
    filebase,
    start_time=None,
    finish_time=finish_time,
    time_unit='kyr',
    out_frequency=6
)

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
shell_volume = pion.geometry_container['shell_volumes']

O2P_line_emission = nebula.line_emission(O2P_pion_ion, verbose=True)  # Initialize the emission line calculation
O3P_line_emission = nebula.line_emission(O3P_pion_ion, verbose=True)  # Initialize the emission line calculation
O5P_line_emission = nebula.line_emission(O5P_pion_ion, verbose=True)  # Initialize the emission line calculation

# Open the output file and write the header
with open(output_file, "w") as file:
    file.write("# " + util.nebula_version() + '\n')
    file.write("# line luminosity(erg/s) for flow velocity: 1500 km/s \n")
    file.write("# time(kyr)\t")
    file.write("\t".join(f"OIII{line}" for line in O2P_lines))
    file.write("\t")
    file.write("\t".join(f"OIV{line}" for line in O3P_lines))
    file.write("\t")
    file.write("\t".join(f"OVI{line}" for line in O5P_lines))
    file.write("\n")

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

    # Extract necessary physical parameters for the current time instant
    density = pion.get_parameter('Density', silo_instant)  # Retrieve density
    temperature = pion.get_parameter('Temperature', silo_instant)  # Retrieve temperature
    O2P_massfrac = pion.get_ion_values(O2P_pion_ion, silo_instant)  # Retrieve ion mass fraction
    O3P_massfrac = pion.get_ion_values(O3P_pion_ion, silo_instant)  # Retrieve ion mass fraction
    O5P_massfrac = pion.get_ion_values(O5P_pion_ion, silo_instant)  # Retrieve ion mass fraction

    # Calculate ion number density (number of ions per unit volume)
    O2P_num_density = O2P_massfrac * density / mass_O
    O3P_num_density = O3P_massfrac * density / mass_O
    O5P_num_density = O5P_massfrac * density / mass_O

    # Retrieve the electron number density
    ne = pion.get_ne(silo_instant)

    # Calculate the line luminosity for the specific emission line
    O2P_line_emission.lineluminosity_spherical(
        lines=O2P_lines,
        temperature=temperature,
        ne=ne,
        species_density=O2P_num_density,
        shell_volume=shell_volume
    )

    # Calculate the line luminosity for the specific emission line
    O3P_line_emission.lineluminosity_spherical(
        lines=O3P_lines,
        temperature=temperature,
        ne=ne,
        species_density=O3P_num_density,
        shell_volume=shell_volume
    )

    # Calculate the line luminosity for the specific emission line
    O5P_line_emission.lineluminosity_spherical(
        lines=O5P_lines,
        temperature=temperature,
        ne=ne,
        species_density=O5P_num_density,
        shell_volume=shell_volume
    )


    # Print the calculated luminosity for the current time instant
    #print(f" L_{lines} = {line_emission_obj.line_emission_container['luminosity']} erg/s")

    # Append the calculated luminosity to the output file
    with open(output_file, "a") as file:
        file.write(f"{sim_time.value:.6e}\t")
        file.write("\t".join(f"{O2P_line_emission.line_emission_container['luminosity'][i]:.6e}"
                            for i in range(len(O2P_lines))))
        file.write("\t")
        file.write("\t".join(f"{O3P_line_emission.line_emission_container['luminosity'][i]:.6e}"
                            for i in range(len(O3P_lines))))
        file.write("\t")
        file.write("\t".join(f"{O5P_line_emission.line_emission_container['luminosity'][i]:.6e}"
                             for i in range(len(O5P_lines))))
        file.write("\n")

    silo_instant_finish_time = time.time()  # Record the finish time
    # Calculate the time spent on the current step
    dt = silo_instant_finish_time - silo_instant_start_time
    # Update the runtime with the time spent on the current step
    runtime += dt
    print(f" runtime: {runtime:.4e} s | dt: {dt:.4e} s")

