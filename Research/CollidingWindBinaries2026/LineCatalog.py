#import numpy as np
import NebulaPy.src as nebula
import time
#from pypion.ReadData import ReadData
import matplotlib.pyplot as plt
import numpy as np
#import astropy.units as unit
import os
#from NebulaPy.tools import constants as const
#import pandas as pd
import ChiantiPy.tools.filters as chfilters
import matplotlib.pyplot as plt  # Plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For attaching colorbars to axes
from matplotlib.ticker import MultipleLocator, ScalarFormatter  # For controlling tick formatting
from NebulaPy.src.CIE import cieMode

# info: code to test emission measure calculation.
import ChiantiPy.core as ch
import ChiantiPy.tools.data as chdata
from NebulaPy.src.EmissionMeasure import emissionMeasure

import NebulaPy.src.Chianti as nebula_chainti

# constants
cm2au = 6.68459e-14  # cm to au conversion factor

# Colliding wind binaries
# Macbook -> Set up paths and filenames
#OutputDir = '/Users/tony/Desktop/CWBs-NEMOv1/Post-Processing/XraySpectrum'  # Output image directory
#SiloDir = '/home/tony/Desktop/CWBs-2026/Silo-n128'  # Directory containing silo files
#Razer Blade -> Set up paths and filenames
OutputDir = '/home/tony/Desktop/CWBs-2026/Postprocessing/X-raySpectrum'  # Output image directory
SiloDir = '/home/tony/Desktop/CWBs-2026/Silo-n128'  # Directory containing silo files
Filebase = 'wr140_NEMO_d07e13_d2l6n128'  # Base name of the silo files
start_time = 1.24e6  # in sec
finish_time = None
time_unit = 'sec'
out_frequency = None
SimulationName = "CWB"


# Bowshock
#Razer Blade -> Set up paths and filenames
OutputDir = '/home/tony/Desktop/CWBs-2026/Postprocessing/X-raySpectrum'  # Output image directory
SiloDir = '/home/tony/Desktop/multi-ion-bowshock/sim-output/silo'  # Directory containing silo files
Filebase = 'Ostar_mhd-nemo-dep_d2n0128l3'  # Base name of the silo files
start_time = 161  # in kyr
finish_time = 161.5
time_unit = 'kyr'
out_frequency = None
SimulationName = "Bowshock"

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
    min_wavelength=1.0,  # Minimum wavelength in Angstroms
    max_wavelength=10.0,  # Maximum wavelength in Angstroms
    min_photon_energy=None,  # Minimum photon energy in keV
    max_photon_energy=None,  # Maximum photon energy in keV
    elements=elements,
    doBremsstrahlung=True,
    doFreebound=True,
    doLine=True,
    doTwophoton=True,
    CIE=False,
    filtername=None,
    filterfactor=None,
    userGrid=True,
    gridSize=3000,
    allLines=True,
    progress=True
)

wavelength = NebulaSpectrum.WavelengthGrid
NebulaSpectrum.line_cataloger(Filebase=Filebase, OutDir=OutputDir)
