"""
Line Luminosity

Description:
This script calculates the temporal evolution of luminosity for selected spectral lines
in an astrophysical simulation. The lines correspond to different ions,
including He1+, C2+, N1+, N2+, O1+, O2+, Ne1+, Ne2+, S1+, S2+, and S3+, and their
corresponding emission lines. The luminosity for each line is computed using the
simulation data and stored for later analysis.

Features:
- Reads silo files generated from the simulation to extract temperature, electron
density, and ion number densities.
- Computes the line luminosity for specific emission lines over a time range.
- Outputs the results to a text file, with each row representing a time step and the luminosities of different lines.

Notes:
Calculating line luminosity only from the shocked layer.  For the line luminosities. For this we exclude any gas that
satisfies the following:
(1) wind tracer value <0.5,
(2) density <1.1x the background inflow density from the parameter file,
(3) v_x within a few (5%) percent of the inflow value (-35 km/s?)

Author: Arun Mathew
Date: 01 Feb 2025
"""

import os
import time
import warnings
from NebulaPy.tools import util
import NebulaPy.src as nebula
import multiprocessing as mp
import time
import numpy as np
import queue  # Import queue globally

# remove this after coding shocked ISM mask
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, ScalarFormatter

#


# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")

# Set OMP_NUM_THREADS=1 to ensure single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"


def compute_luminosity(workerQ, doneQ):
    """Worker function to compute luminosity for each ion"""
    import queue  # Import inside function to avoid errors

    while True:
        try:
            task = workerQ.get(timeout=timeout)  # Get task from queue
            if task is None:
                # print("[Worker] Received stop signal, exiting.")
                break  # Exit if None is received (stop signal)

            process_id, line_emission, lines, species_density, temperature, ne, cell_volume, mask, ism_mask, recombination = task

            if recombination:
                print(f" multiprocessing – ID {process_id}: calculation of HI line(s) luminosities from recombination")
                luminosity = line_emission.recombination_line_luminosity_2D(
                    lines=lines, temperature=temperature, ne=ne,
                    species_density=species_density,
                    cell_volume=cell_volume, grid_mask=mask, progress_bar=True)
            else:
                print(f" multiprocessing – ID {process_id}: calculation of HI collisional line(s) luminosities")
                luminosity = line_emission.line_luminosity_2D(
                    lines=lines, temperature=temperature, ne=ne,
                    species_density=species_density,
                    cell_volume=cell_volume, grid_mask=mask, progress_bar=False)

            if recombination:
                print(f" multiprocessing – ID {process_id}: finished computing HI line(s) luminosities from recombination")
            else:
                print(f" multiprocessing – ID {process_id}: finished computing HI collisional line(s) luminosities")

            luminosity['ism_mask'] = ism_mask
            doneQ.put({process_id: luminosity})  # Store result
        except queue.Empty:  # Use correct exception for empty queue
            util.nebula_exit_with_error(f"multiprocessing - no task in queue")
            break  # Queue is empty, exit loop
        except Exception as e:
            util.nebula_exit_with_error(f'multiprocessing - working process {e}')
            break


def generate_shocked_ism_mask(pion, silo_instant):
    """
    Generate a mask for the shocked ISM region based on specific criteria:
    (1) wind tracer value < 0.5,
    (2) density < 1.1x the background inflow density from the parameter file,
    (3) v_x within a few (5%) percent of the inflow value (-35 km/s)

    Parameters:
    pion: nebula.pion object
    silo_instant: the current silo file being processed.

    Returns:
    numpy.ndarray
        A boolean mask array where True indicates shocked ISM regions.
    """

    print(f" generating shocked ISM mask")

    # Extract necessary parameters from the simulation
    density = pion.get_parameter('Density', silo_instant)
    grid_mask = pion.get_parameter('NG_Mask', silo_instant)
    shocked_ism_mask = [np.ones(shape) for shape in [arr.shape for arr in density]]
    wind_tracer = pion.get_parameter('Tr000_WIND', silo_instant)
    velocity_Y = pion.get_parameter('VelocityY', silo_instant)

    for level in range(N_grid_level):
        shocked_ism_mask[level][wind_tracer[level] > 0.5] = 0
        shocked_ism_mask[level][density[level] < 1.05 * 1.0E-23] = 0
        shocked_ism_mask[level][velocity_Y[level] <= 12.0E+03] = 0
        shocked_ism_mask[level] = shocked_ism_mask[level] * grid_mask[level]

    return shocked_ism_mask


if __name__ == "__main__":

    # Input-output file configuration for the low-resolution simulation on Razer Blade machine.
    output_dir = '/home/tony/Desktop/multi-ion-bowshock/sim-output/time-lines-luminosity'
    silo_dir = '/home/tony/Desktop/multi-ion-bowshock/high-res-silos-200kyr'
    filebase = 'Ostar_mhd-nemo-dep_d2n0384l3'  # Base name of the silo files
    ism_filename = filebase + '_HI_luminosity_ShockedISM_HighRes.txt'
    grid_filename = filebase + '_HI_luminosity_Grid_HighRes.txt'
    start_time = None
    finish_time = None
    out_frequency = None
    time_unit = 'kyr'


    # Input-output file configuration for the medium-resolution simulation on MIMIR
    # output_dir = '/mnt/massive-stars/data/arun_simulations/Nemo_BowShock/med-res/time-lines-luminosity'
    # silo_dir = '/mnt/massive-stars/data/arun_simulations/Nemo_BowShock/med-res/silo'
    # filebase = 'Ostar_mhd-nemo-dep_d2n0192l3'  # Base name of the silo files
    # filename = filebase + '_lines_luminosity_MedRes_3.txt'
    # start_time = 0
    # finish_time = 47
    # out_frequency = 2

    # Input-output file configuration for the high-resolution simulation on MIMIR
    #output_dir = '/mnt/massive-stars/data/arun_simulations/Nemo_BowShock/high-res/time-lines-luminosity'
    #silo_dir = '/mnt/massive-stars/data/arun_simulations/Nemo_BowShock/high-res/silo'
    #filebase = 'Ostar_mhd-nemo-dep_d2n0384l3'  # Base name of the silo files
    #filename = filebase + '_lines_luminosity_HighRes_ShockedISM.txt'
    #start_time = None
    #finish_time = 201
    #out_frequency = 2
    #time_unit = 'kyr'

    # Batch the silo files for analysis within the specified time range
    batched_silos = util.batch_silos(
        silo_dir,
        filebase,
        start_time=start_time,
        finish_time=finish_time,
        time_unit=time_unit,
        out_frequency=out_frequency
    )

    # Total number of time instant
    N_time_instant = len(batched_silos)

    # Initialize the PION class to handle simulation data
    pion = nebula.pion(batched_silos, verbose=True)
    # Load chemistry and geometry data
    pion.load_chemistry()
    pion.load_geometry(scale='cm')

    print(f" ---------------------------")
    print(f" task: multiprocessing for computing the temporal evolution of Hα, Hβ luminosities")
    print(" list of spectral lines to be processed:")

    # Set up the ion and line emission parameters
    # Define the ions and their respective emission lines
    # Chianti H-alpha lines 6564.538, 6564.564, 6564.523, 6564.665, 6564.722 (Å)
    # and the repeated entries near 6564 Å correspond to the fine-structure
    # components of the hydrogen Hα line (Balmer α, 3 → 2 transition).

    # pion ion name
    pion_ion = 'H'

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

    # Initialize the emission line calculations for each ion
    line_emission = nebula.line_emission(pion_ion, verbose=True)

    # Check the requested lines in the database for each ion
    print(f" ---------------------------")
    print(f" checking requested lines in CHIANTI database:")
    line_emission.chianti_line_batch_check(Chianti_HAlphaLines)
    line_emission.chianti_line_batch_check(Chianti_HBetaLines)

    print(f" ---------------------------")
    print(f" checking requested lines in PyNeb database:")
    line_emission.pyneb_line_batch_check(PyNeb_HAlphaLines)
    line_emission.pyneb_line_batch_check(PyNeb_HBetaLines)

    # Get geometry information
    geometry = pion.geometry_container
    N_grid_level = geometry['Nlevel']
    grid_mask = geometry['mask']
    cell_volume = pion.get_cylindrical_cell_volume().value

    mesh_edges_min = pion.geometry_container['edges_min']
    mesh_edges_max = pion.geometry_container['edges_max']

    # creating output file
    ism_outfile = os.path.join(output_dir, ism_filename)
    grid_outfile = os.path.join(output_dir, grid_filename)


    # Write initial header to the output file
    with open(ism_outfile, "w") as ism_file:
        ism_file.write(f"#File generated by {util.nebula_version()}\n\n")
        ism_file.write("#Task: Multiprocessing for Computing the Temporal Evolution of H Lines Luminosity\n\n")
        ism_file.write(f"#PION Simulation Reference: NEMO Bowshock 2025\n")
        ism_file.write(f"#PION Simulation FileBase: {filebase}\n")
        ism_file.write(f"#PION Simulation Mask: Shocked ISM Mask\n\n")
        ism_file.write("#Dataset Description:\n")
        ism_file.write("#This dataset provides the luminosity evolution of selected spectral lines over time.\n")
        ism_file.write("#Each row represents a different time step, with:\n")
        ism_file.write("# - The first column indicating time (in kyr).\n")
        ism_file.write("# - Subsequent columns representing the luminosity of specific spectral lines (in erg/s).\n\n")

        # Write initial header to the output file
    with open(grid_outfile, "w") as grid_file:
        grid_file.write(f"#File generated by {util.nebula_version()}\n\n")
        grid_file.write("#Task: Multiprocessing for Computing the Temporal Evolution of H Lines Luminosity\n\n")
        grid_file.write(f"#PION Simulation Reference: NEMO Bowshock 2025\n")
        grid_file.write(f"#PION Simulation FileBase: {filebase}\n")
        grid_file.write(f"#PION Simulation Mask: Grid Mask\n\n")
        grid_file.write("#Dataset Description:\n")
        grid_file.write("#This dataset provides the luminosity evolution of selected spectral lines over time.\n")
        grid_file.write("#Each row represents a different time step, with:\n")
        grid_file.write("# - The first column indicating time (in kyr).\n")
        grid_file.write("# - Subsequent columns representing the luminosity of specific spectral lines (in erg/s).\n\n")

    # Loop over each time instant in the batched silo files
    runtime = 0.0
    grid_write_heading = True
    ism_write_heading = True

    for step, silo_instant in enumerate(batched_silos):
        silo_instant_start_time = time.time()

        print(f" ---------------------------")
        sim_time = pion.get_simulation_time(silo_instant, time_unit='kyr')
        print(f" step: {step}/{N_time_instant - 1} | simulation time: {sim_time:.6e}")

        # get temperature
        temperature = pion.get_parameter('Temperature', silo_instant)
        # calculate electron number density
        ne = pion.get_ne(silo_instant)
        # generate ISM shocked mask
        shocked_ism_mask = generate_shocked_ism_mask(pion, silo_instant)

        # Retrieve species number density
        H_num_density = pion.get_ion_number_density(pion_ion, silo_instant)
        # note: require H1+ number density to calculate recombination line
        # luminosities, hence
        H1P_num_density = pion.get_ion_number_density('H1+', silo_instant)

        # list of all processes
        all_processes = {
            # luminosity from the  whole simulation domain
            "HalphCollGrid": (line_emission, Chianti_HAlphaLines, H_num_density, False, grid_mask, False),
            "HalphRecoGrid": (line_emission, PyNeb_HAlphaLines, H1P_num_density, True, grid_mask, False),
            "HbetaCollGrid": (line_emission, Chianti_HBetaLines, H_num_density, False, grid_mask, False),
            "HbetaRecoGrid": (line_emission, PyNeb_HBetaLines, H1P_num_density, True, grid_mask, False),

            "HLyalpCollGrid": (line_emission, Chianti_HLyAlphaLines, H_num_density, False, grid_mask, False),
            "HPaalpRecoGrid": (line_emission, PyNeb_HPaAlphaLines, H1P_num_density, True, grid_mask, False),
            "HBrgamRecoGrid": (line_emission, PyNeb_HBrGammaLines, H1P_num_density, True, grid_mask, False),
            "HLyalpRecoGrid": (line_emission, PyNeb_HLyAlphaLines, H1P_num_density, True, grid_mask, False),

            # luminosity from shocked ISM
            "HalphCollIsmM": (line_emission, Chianti_HAlphaLines, H_num_density, False, shocked_ism_mask, True),
            "HalphRecoIsmM": (line_emission, PyNeb_HAlphaLines, H1P_num_density, True, shocked_ism_mask, True),
            "HbetaCollIsmM": (line_emission, Chianti_HBetaLines, H_num_density, False, shocked_ism_mask, True),
            "HbetaRecoIsmM": (line_emission, PyNeb_HBetaLines, H1P_num_density, True, shocked_ism_mask, True),
            "HLyalpCollIsmM": (line_emission, Chianti_HLyAlphaLines, H_num_density, False, shocked_ism_mask, True),
            "HPaalpRecoIsmM": (line_emission, PyNeb_HPaAlphaLines, H1P_num_density, True, shocked_ism_mask, True),
            "HBrgamRecoIsmM": (line_emission, PyNeb_HBrGammaLines, H1P_num_density, True, shocked_ism_mask, True),
            "HLyalpRecoIsmM": (line_emission, PyNeb_HLyAlphaLines, H1P_num_density, True, shocked_ism_mask, True),
        }

        # get keys of ions
        process_IDs = all_processes.keys()
        computed_result = {}

        # Parameters for multiprocessing
        timeout = 0.1
        Ncores = len(all_processes)
        cpu_count = mp.cpu_count()
        proc = min(Ncores, cpu_count)
        print(f" multiprocessing: utilizing {proc} cores (one core per ion)")

        with mp.Manager() as manager:
            luminosity_workerQ = manager.Queue()
            luminosity_doneQ = manager.Queue()

            print(" multiprocessing: adding tasks to queue")
            # Populate worker queue
            for task, (line_emission, lines, species_density, recombination, mask, ism_mask) in all_processes.items():
                luminosity_workerQ.put((task, line_emission, lines, species_density, temperature, ne, cell_volume, mask, ism_mask, recombination))

            print(" multiprocessing: starting worker processes")
            # Start worker processes
            luminosity_processes = []
            for _ in range(proc):
                p = mp.Process(target=compute_luminosity, args=(luminosity_workerQ, luminosity_doneQ))
                p.start()
                luminosity_processes.append(p)

            # Allow some time for processing
            time.sleep(2)

            # Send stop signal (None) to workers
            for _ in range(proc):
                luminosity_workerQ.put(None)

            # Wait for processes to complete
            for p in luminosity_processes:
                p.join()

            # Collect results
            while not luminosity_doneQ.empty():
                computed_result.update(luminosity_doneQ.get())

        # Order dictionary entries according to `all_processes`# Order dictionary entries according to `all_processes`
        sorted_computed_result = {key: computed_result[key] for key in process_IDs}

        shocked_ism_luminosity = {}
        grid_mask_luminosity = {}
        for key, data in sorted_computed_result.items():
            if data.get('ism_mask', False):
                shocked_ism_luminosity[key] = data
                del shocked_ism_luminosity[key]['ism_mask']
            else:
                grid_mask_luminosity[key] = data
                del grid_mask_luminosity[key]['ism_mask']

        # combine all sub dictionary into a single dictionary
        shocked_ism_luminosity_dict = {key: value for subdict in shocked_ism_luminosity.values() for key, value in subdict.items()}
        grid_mask_luminosity_dict = {key: value for subdict in grid_mask_luminosity.values() for key, value in subdict.items()}

        print(f" saving shocked ISM luminosity data to file: {ism_filename}")
        if ism_write_heading:
            with open(ism_outfile, "a") as ism_file:
                # Write the time (or any other desired variable)
                ism_file.write(f"Time ")
                # Write the keys from Dictionaries
                ism_file.write(" ".join(f"{key}" for key in shocked_ism_luminosity_dict.keys()))

                ism_file.write("\n")
            write_heading = False

        with open(ism_outfile, "a") as ism_file:
            ism_file.write(f"{sim_time.value:.6e} ")
            ism_file.write(" ".join(f"{value:.6e}" for value in shocked_ism_luminosity_dict.values()))
            ism_file.write("\n")

        print(f" saving full domain luminosity data to file: {grid_filename}")
        if ism_write_heading:
            with open(grid_outfile, "a") as grid_file:
                # Write the time (or any other desired variable)
                grid_file.write(f"Time ")
                # Write the keys from Dictionaries
                grid_file.write(" ".join(f"{key}" for key in grid_mask_luminosity_dict.keys()))

                grid_file.write("\n")
            ism_write_heading = False

        with open(grid_outfile, "a") as grid_file:
            grid_file.write(f"{sim_time.value:.6e} ")
            grid_file.write(" ".join(f"{value:.6e}" for value in grid_mask_luminosity_dict.values()))
            grid_file.write("\n")

        del shocked_ism_luminosity
        del shocked_ism_luminosity_dict
        del grid_mask_luminosity
        del grid_mask_luminosity_dict

        '''
        # generating masked grid image
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        ax.text(0.05, 0.9, 'time = %5.2f kyr' % sim_time.value, transform=ax.transAxes,
                fontsize=12, color='white')
        ax.set_xlim(mesh_edges_min[0][0].value, mesh_edges_max[0][0].value)
        ax.set_ylim(mesh_edges_min[0][1].value, mesh_edges_max[0][1].value)

        # Assuming parameter_data, dims_min_1, dims_max_1, and cmap_value are defined
        # Assuming ax is already defined
        for level in range(N_grid_level):
            plot_data = shocked_ism_mask[level]
            extents = [mesh_edges_min[level][0].value, mesh_edges_max[level][0].value,
                       mesh_edges_min[level][1].value, mesh_edges_max[level][1].value]

            image = ax.imshow(plot_data, interpolation='nearest', cmap='inferno',
                              extent=extents, origin='lower')

        # Create divider for existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax1
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add colorbar with appropriate ticks
        colorbar = plt.colorbar(image, cax=cax,
                                #ticks=MultipleLocator(1)
                                )

        # Set the formatter for the colorbar
        colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())
        colorbar.ax.yaxis.get_major_formatter().set_scientific(False)  # Disable scientific notation
        colorbar.ax.yaxis.get_major_formatter().set_useOffset(False)  # Disable offset

        # Update the ticks for the colorbar
        colorbar.update_ticks()
        ax.set_xlabel('z (pc)', fontsize=12)
        ax.set_ylabel('R (pc)', fontsize=12)

        ax.text(0.65, 0.9, 'Shocked ISM', transform=ax.transAxes, fontsize=12, color='white')  # Ensure a string is provided here
        ax.tick_params(axis='both', which='major', labelsize=13)
        # ax.axes.get_xaxis().set_visible(False)  # Remove the x-axis
        plt.show()
        '''

        # Update runtime
        silo_instant_finish_time = time.time()
        dt = silo_instant_finish_time - silo_instant_start_time
        runtime += dt
        print(f" runtime: {runtime:.4e} s | step runtime: {dt:.4e} s")

