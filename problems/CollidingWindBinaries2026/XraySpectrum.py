#import numpy as np
import NebulaPy.src as nebula
from NebulaPy.src import Utils as util
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# constants
cm2au = 6.68459e-14  # cm to au conversion factor


# Colliding wind binaries
#Razer Blade -> Set up paths and filenames
OutputDir = '/home/tony/Desktop/CWBs-2026/Postprocessing/X-raySpectrum'  # Output image directory
SiloDir = '/home/tony/Desktop/CWBs-2026/Silo-n128'  # Directory containing silo files
Filebase = 'wr140_NEMO_d07e13_d2l6n128'  # Base name of the silo files
start_time = 1.24e6  # in sec
finish_time = None
time_unit = 'sec'
out_frequency = None
SimulationName = "CWB"

# edit here for Mimir
'''
#MIMIR -> Set up paths and filenames
OutputDir = ''  # Output image directory
SiloDir = ''  # Directory containing silo files
Filebase = 'wr140_NEMO_d07e13_d2l6n128'  # Base name of the silo files
start_time = 1.24e6  # in sec
finish_time = None
time_unit = 'sec'
out_frequency = None
SimulationName = "CWB"
'''


# Bowshock
'''
#Razer Blade -> Set up paths and filenames
OutputDir = '/home/tony/Desktop/CWBs-2026/Postprocessing/X-raySpectrum'  # Output image directory
SiloDir = '/home/tony/Desktop/multi-ion-bowshock/sim-output/silo'  # Directory containing silo files
Filebase = 'Ostar_mhd-nemo-dep_d2n0128l3'  # Base name of the silo files
start_time = 161  # in kyr
finish_time = 161.5
time_unit = 'kyr'
out_frequency = None
SimulationName = "Bowshock_FF"
'''

# Batch the silo files according to the time instant
batched_silos = util.batch_silos(
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
    util.nebula_info("Resetting parameters before the next run")
    exit(0)


# Initialize the Pion class from NebulaPy, which handles the simulation data
pion = nebula.pion(batched_silos, verbose=True)

# loading geometry attributes from the first silo file in the batch
# and saves them into a geometry container.
pion.load_geometry(scale='cm')
N_grid_level = pion.geometry_container['Nlevel']
mesh_edges_min = pion.geometry_container['edges_min']
mesh_edges_max = pion.geometry_container['edges_max']
N_grid = pion.geometry_container['Ngrid']
grid_volume = pion.get_grid_volumes_2D()
grid_mask = pion.geometry_container['mask']

# loading chemistry container for pion simulation data
pion.load_chemistry()
elements = pion.get_elements()

# initializing spectrum class
NebulaSpectrum = nebula.spectrum(
    min_wavelength=0.5,  # Minimum wavelength in Angstroms
    max_wavelength=20.0,  # Maximum wavelength in Angstroms
    min_photon_energy=None,  # Minimum photon energy in keV # not implemented
    max_photon_energy=None,  # Maximum photon energy in keV # not implemented
    elements=elements,
    doBremsstrahlung=False,
    doFreebound=False,
    doLine=False,
    doTwophoton=True,
    CIE=False,
    filtername=None,
    filterfactor=None,
    userGrid=True,
    gridSize=3000,
    allLines=True,
    MPNcores=4,
    verbose=False
)

runtime = 0.0

# Loop over each time instant in the batched silo files
for step, silo_instant in enumerate(batched_silos):

    silo_instant_start_time = time.time()

    print(" [ SIMULATION SNAPSHOT ] " + "─" * 70)

    sim_time = pion.get_simulation_time(silo_instant, time_unit=time_unit)
    print(f" Step: {step} | Simulation time: {sim_time:.6e}")

    # Extract temperature and electron number density
    temperature = np.asarray(
        pion.get_parameter('Temperature', silo_instant),
        dtype=np.float64
    )

    ne = pion.get_ne(silo_instant)
    species_densities = pion.get_species_number_densities(silo_instant)

    NebulaSpectrum.generateSpectrum(
        temperature=temperature,
        ne=ne,
        species_densities=species_densities,
        grid_volume=grid_volume,
        grid_mask=grid_mask
    )

    wavelength = NebulaSpectrum.WavelengthGrid
    spectrum = NebulaSpectrum.Spectrum

    # Save spectrum to text file
    txtfile = os.path.join(
        OutputDir,
        f"{Filebase}_Spectrum_{sim_time.value:6e}.txt"
    )

    np.savetxt(
        txtfile,
        np.c_[wavelength, spectrum],
        header="Wavelength[A] Spectrum[erg s^-1 A^-1]",
        fmt="%.8e"
    )

    print(f" Saved spectrum data to {txtfile}")

    energy = 12.39841984 / wavelength
    idx = np.argsort(energy)
    energy = energy[idx]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=12)
    ax.set_ylabel(r"$L_\lambda$ [erg s$^{-1}$ $\AA^{-1}$]", fontsize=12)

    ax.plot(
        wavelength,
        spectrum,
        color="green",
        linewidth=1.4,
        label=f"NEQ {SimulationName} Spectrum"
    )

    ax.set_yscale("log")

    ax.minorticks_on()

    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        top=True,
        right=True,
        length=6,
        width=1.2,
        labelsize=11
    )

    ax.tick_params(
        axis='both',
        which='minor',
        direction='in',
        top=True,
        right=True,
        length=3,
        width=1.0
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.legend(loc='best', frameon=False, fontsize=10)

    fig.tight_layout()

    outfile = os.path.join(OutputDir, f"{Filebase}_Spectrum_{sim_time.value:6e}.png")

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f" sim time: {sim_time:.6e}, saved snapshot {step} to {outfile}")

    # Update runtime
    dt = time.time() - silo_instant_start_time
    runtime += dt

    print(f" runtime: {runtime:.4e} s | step runtime: {dt:.4e} s")