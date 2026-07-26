"""
Line Luminosity

Description:
This script calculates the temporal evolution of luminosity for selected spectral lines
in an astrophysical simulation. The lines correspond to different ions,
including He1+, C2+, N1+, N2+, O1+, O2+, Ne1+, Ne2+, S1+, S2+, and S3+, and their
corresponding emission lines. The luminosity for each line is computed using the
simulation data and stored for later analysis.

Features:
- Reads silo files generated from the simulation to extract temperature, electron density, and ion number densities.
- Computes the line luminosity for specific emission lines over a time range.
- Outputs the results to a text file, with each row representing a time step and the luminosities of different lines.

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
import queue  # Import queue globally
import time
from typing import Any, Dict, Tuple, Optional
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from matplotlib.gridspec import GridSpec

from pypion.ReadData import ReadData
import astropy.units as unit

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")

# Set OMP_NUM_THREADS=1 to ensure single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"

def compute_emissivity(workerQ, doneQ, timeout=0.1):
    """Worker function to compute luminosity for each ion"""
    import queue  # Import inside function to avoid errors

    while True:
        try:
            task = workerQ.get(timeout=timeout)  # Get task from queue
            if task is None:
                #print("[Worker] Received stop signal, exiting.")
                break  # Exit if None is received (stop signal)

            ion_name, line_emission, lines, temperature, ne = task

            print(f" Multiprocessing: computing the emissivity of {ion_name:<4} lines")

            emissivity = line_emission.line_emissivity_2D_map(lines=lines, temperature=temperature, ne=ne, progress_bar=True)

            #emissivity['He II 303.78'] = np.where(
            #    emissivity['He II 303.78'] < 1e-30, 1e-30, emissivity['He II 303.78'])


            print(f" Multiprocessing: finished computing emissivity for {ion_name:<4} lines")
            doneQ.put({ion_name: emissivity})  # Store result
        except queue.Empty:  # Use correct exception for empty queue
            util.nebula_exit_with_error(f"Multiprocessing - no task in queue")
            break  # Queue is empty, exit loop
        except Exception as e:
            util.nebula_exit_with_error(f'Multiprocessing - working process {e}')
            break

def print_spectral_lines(task_desc, ion_lines):
    width_ion = 6
    width_lines = 70

    print("─" * (width_ion + width_lines + 7))
    print(f" TASK : {task_desc}")
    print("─" * (width_ion + width_lines + 7))
    print(f" {'ION':<{width_ion}} | EMISSION LINES (Å)")
    print("─" * (width_ion + width_lines + 7))

    for ion, lines in ion_lines.items():
        line_str = ", ".join(f"{l:g}" for l in lines)
        print(f" {ion:<{width_ion}} | {line_str}")


def run_multiprocessing_task(
    task_packet: Dict[str, Tuple[Any, Any, Any, Any]],
    task,
    *,
    sleep_before_stop: float = 2.0,
    max_procs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    task_packet maps:
      ion_name -> (line_emission, lines, species_density, temperature, ne, cell_volume, grid_mask)

    compute_luminosity_func(workerQ, doneQ) must consume:
      (ion_name, line_emission, lines, species_density, temperature, ne, cell_volume, grid_mask)
    and put dict results into doneQ, e.g. {ion_name: {...}}.
    """
    keys = list(task_packet.keys())
    if not keys:
        return {}

    cpu_count = mp.cpu_count()
    proc = min(len(keys), cpu_count)
    if max_procs is not None:
        proc = max(1, min(proc, int(max_procs)))

    if verbose:
        print(f" Multiprocessing: utilizing {proc} core(s) (one core per ion)")
        print(" Multiprocessing: adding tasks to queue")

    species_lines_luminosity: Dict[str, Dict[str, Any]] = {}

    with mp.Manager() as manager:
        workerQ = manager.Queue()
        doneQ = manager.Queue()

        # Enqueue tasks (expand tuple)
        for ion_name, payload in task_packet.items():
            (line_emission, lines, temperature, ne) = payload

            workerQ.put((ion_name, line_emission, lines, temperature, ne))

        if verbose:
            print(" Multiprocessing: starting worker processes")

        procs = []
        for _ in range(proc):
            p = mp.Process(target=task, args=(workerQ, doneQ))
            p.start()
            procs.append(p)

        if sleep_before_stop and sleep_before_stop > 0:
            time.sleep(sleep_before_stop)

        # stop signals
        for _ in range(proc):
            workerQ.put(None)

        for p in procs:
            p.join()

        # collect results
        while not doneQ.empty():
            species_lines_luminosity.update(doneQ.get())

    # Preserve original ion order, then flatten
    sorted_species = {k: species_lines_luminosity[k] for k in keys if k in species_lines_luminosity}
    flattened = {lk: lv for sub in sorted_species.values() for lk, lv in sub.items()}
    return flattened


if __name__ == "__main__":

    # Input-output file configuration for the high-resolution simulation on Razer Blade machine.
    # NEQ configuration
    neq_silo_dir = '/home/tony/Desktop/multi-ion-bowshock/high-res-silo-200kyr'
    neq_filebase = 'Ostar_mhd-nemo-dep_d2n0384l3'  # Base name of the silo files
    neq_start_time = 200  # Start time for the simulation in kyr
    neq_finish_time = 201  # Finish time for the simulation in kyr
    # IEQ configuration
    ieq_silo_dir = '/home/tony/Desktop/ioneq'
    ieq_filebase = 'Ostar_mhd-nemo-dep-ioneq_d2n0384l3'  # Base name of the silo files
    ieq_start_time = 39645  # Start time for the simulation in kyr
    ieq_finish_time = None  # Finish time for the simulation in kyr
    # common configuration
    out_frequency = None  # Optional output frequency (default: None)
    time_unit = 'kyr'
    spatial_scale = 'cm' # Issue: While calculating line luminosities set spatial scale to 'cm'
    # Output configuration
    output_dir = '/home/tony/Desktop/multi-ion-bowshock/sim-output/Llines-NEQ-IEQ'
    filename = neq_filebase + '_OxyEmiss_NEQ_IEQ.png'
    outfile = os.path.join(output_dir, filename)

    # All input lines
    ion_lines = {
        #"He1+": [303.78],
        "O5+": [1031.912]
    }

    #####################################################
    # Batching silo files
    neq_batched_silos = util.batch_silos(
        neq_silo_dir,
        neq_filebase,
        start_time=neq_start_time,
        finish_time=neq_finish_time,
        time_unit=time_unit,
        out_frequency=out_frequency
    )
    print(" Finished batching non-equilibrium silo files")
    ieq_batched_silos = util.batch_silos(
        ieq_silo_dir,
        ieq_filebase,
        start_time=ieq_start_time,
        finish_time=ieq_finish_time,
        time_unit=time_unit,
        out_frequency=out_frequency
    )
    print(" Finished batching equilibrium silo files")

    key = input(" Press 'y' to continue, anything else to exit: ").strip().lower()

    if key == "y":
        print(" Continuing execution...")
    else:
        util.nebula_info("Resetting parameters before the next run")
        exit(0)

    if len(neq_batched_silos) != 1 or len(ieq_batched_silos) != 1:
        raise ValueError(
            f"Expected exactly one time instant, but found "
            f"{len(neq_batched_silos)} NEQ and {len(ieq_batched_silos)} EQI entries. "
            "Specify a single time instant before proceeding."
        )

    print_spectral_lines(
        "Multiprocessing computation of spectral-line emissivity",
        ion_lines
    )

    ###############################################################
    # Initialize the PION class to extract microphysics data
    neq_pion = nebula.pion(neq_batched_silos, verbose=True)
    # Load chemistry and geometry data
    neq_pion.load_chemistry()
    print(" Finished loading chemistry for non-equilibrium silo files")
    neq_pion.load_geometry(scale=spatial_scale)
    print(" Finished loading geometry for non-equilibrium silo files")

    # Initialize the PION class to extract microphysics data
    ieq_pion = nebula.pion(ieq_batched_silos, verbose=True)
    # Load chemistry and geometry data
    ieq_pion.load_chemistry()
    print(" Finished loading chemistry for equilibrium silo files")
    ieq_pion.load_geometry(scale=spatial_scale)
    print(" Finished loading geometry for equilibrium silo files")

    ###############################################################
    # creating line emission objects for each ions
    # note: common to both NEI and EQI
    print("─" * 85)
    print(" Checking requested lines in database:")
    line_emission_objects = {}
    for ion, lines in ion_lines.items():
        # Initialise emission object
        le = nebula.line_emission(ion, verbose=True)
        # availability check (if applicable)
        #if hasattr(le, "chianti_line_batch_check"):
        #    if ion == 'H':
        #        le.pyneb_line_batch_check(lines)
        #    else:
        #        le.chianti_line_batch_check(lines)

        # Store for later luminosity / time-evolution calculations
        line_emission_objects[ion] = le

    #######################################################
    # Get geometry information
    neq_geometry = neq_pion.geometry_container
    N_grid_level_neq = neq_geometry['Nlevel']
    grid_mask_neq = neq_geometry['mask']
    cell_volume_neq = neq_pion.get_2D_cell_volumes()

    ieq_geometry = ieq_pion.geometry_container
    N_grid_level_ieq = ieq_geometry['Nlevel']
    grid_mask_ieq = ieq_geometry['mask']
    cell_volume_ieq = ieq_pion.get_2D_cell_volumes()


    # Loop over each time instant in the batched silo files
    runtime = 0.0
    # Number of time instant in NEQ batched silo should be same as IEQ.
    N_time_instant = len(neq_batched_silos)
    for step, (neq_silo, ieq_silo) in enumerate(zip(neq_batched_silos, ieq_batched_silos)):

        run_start_time = time.time()

        print("─" * 85)
        # NEQ ################################################################
        neq_sim_time = neq_pion.get_simulation_time(neq_silo, time_unit='kyr')
        print(f" Step: {step}/{N_time_instant-1} | NEQ simulation time: {neq_sim_time:.6e}")

        # Extract temperature and electron number density
        neq_temperature = neq_pion.get_parameter('Temperature', neq_silo)
        neq_ne = neq_pion.get_ne(neq_silo)

        print(" Multiprocessing: preparing per-ion NEQ line-emission tasks")
        neq_task_packet = {
            ion: (
                line_emission_objects[ion],  # line emission object
                ion_lines[ion],  # list of wavelengths
                neq_temperature,
                neq_ne
            )
            for ion in ion_lines
        }

        neq_lines_emissivity_dict = run_multiprocessing_task(
            task_packet=neq_task_packet,
            task=compute_emissivity,
        )

        # IEQ ################################################################
        ieq_sim_time = ieq_pion.get_simulation_time(ieq_silo, time_unit='kyr')
        print(f" Step: {step}/{N_time_instant - 1} | IEQ simulation time: {ieq_sim_time:.6e}")

        # Extract temperature and electron number density
        ieq_temperature = ieq_pion.get_parameter('Temperature', ieq_silo)
        ieq_ne = ieq_pion.get_ne(ieq_silo)

        print(" Multiprocessing: preparing per-ion IEQ line-emission tasks")
        ieq_task_packet = {
            ion: (
                line_emission_objects[ion],  # line emission object
                ion_lines[ion],  # list of wavelengths
                ieq_temperature,
                ieq_ne,
            )
            for ion in ion_lines
        }

        ieq_lines_emissivity_dict = run_multiprocessing_task(
            task_packet=ieq_task_packet,
            task=compute_emissivity,
        )

        print(f" Plotting NEQ-IEQ Oxygen line")


        # Create a new figure with specific size and two subplots arranged vertically
        fig, (axs_neq, axs_ieq) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        # Extract the data for the current time instance from the appropriate resolution
        time_instances = neq_silo
        data = ReadData(time_instances)
        N_level = data.nlevels()  # Number of levels (e.g., ionization levels)
        BasicDate = data.get_2Darray('Density')  # Density array

        dims_max = (BasicDate['max_extents'] * unit.cm).to(unit.pc)
        dims_min = (BasicDate['min_extents'] * unit.cm).to(unit.pc)
        distance_unit_str = 'pc'

        # NEQ emissivity plot
        axs_neq.set_yticks([0.5, 1, 1.5, 2.0])  # Set specific tick positions
        axs_neq.tick_params(axis='y', labelsize=12)  # Adjust font size of y-axis ticks
        axs_neq.set_xlim(dims_min[0][0].value, dims_max[0][0].value)
        axs_neq.set_ylim(dims_min[0][1].value, dims_max[0][1].value)

        neq_nion = neq_pion.get_ion_number_density(ion, neq_silo)
        neq_norm_emissivity = np.zeros_like(neq_temperature)

        for l in range(N_level):
            for line in sorted(neq_lines_emissivity_dict.keys()):
                neq_norm_emissivity[l] = neq_lines_emissivity_dict[line][l] * neq_nion[l] * grid_mask_neq[l]
                neq_norm_emissivity[l] = np.where(neq_norm_emissivity[l] < 1e-30, 1e-30, neq_norm_emissivity[l])
                plot_data = np.log10(neq_norm_emissivity[l])
            extents = [dims_min[l][0].value, dims_max[l][0].value,
                       dims_min[l][1].value, dims_max[l][1].value]
            image = axs_neq.imshow(plot_data, interpolation='nearest', cmap='viridis',
                                       extent=extents, origin='lower', vmin=-27, vmax=-24)

        axs_neq.set_ylabel('R (' + distance_unit_str + ')', fontsize=13)
        axs_neq.text(0.1, 0.8, r'O VI 1032 $\mathrm{\AA}$',
                     transform=axs_neq.transAxes,
                     fontsize=12, color='white')
        axs_neq.text(0.88, 0.8, 'NEQ', transform=axs_neq.transAxes,
                         fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        axs_neq.tick_params(axis='both', which='major', labelsize=13)
        axs_neq.set_yticks([0.5, 1, 1.5, 2.0])  # Set specific tick positions
        axs_neq.tick_params(axis='y', labelsize=12)  # Adjust font size of y-axis ticks
        axs_neq.axes.get_xaxis().set_visible(False)  # Remove x-axis

        # IEQ plot
        axs_ieq.set_xlim(dims_min[0][0].value, dims_max[0][0].value)
        axs_ieq.set_ylim(-dims_max[0][1].value, -dims_min[0][1].value)
        axs_ieq.tick_params(axis='x', labelsize=12)  # Adjust font size of y-axis ticks


        ieq_nion = ieq_pion.get_ion_number_density(ion, ieq_silo)
        ieq_norm_emissivity = np.zeros_like(ieq_temperature)

        for l in range(N_level):
            for line in sorted(ieq_lines_emissivity_dict.keys()):
                ieq_norm_emissivity[l] = ieq_lines_emissivity_dict[line][l] * ieq_nion[l] * grid_mask_ieq[l]
                ieq_norm_emissivity[l] = np.where(ieq_norm_emissivity[l] < 1e-30, 1e-30, ieq_norm_emissivity[l])
                plot_data = np.log10(ieq_norm_emissivity[l])
            extents = [dims_min[l][0].value, dims_max[l][0].value,
                       -dims_max[l][1].value, -dims_min[l][1].value]
            image = axs_ieq.imshow(plot_data, interpolation='nearest', cmap='viridis',
                                    extent=extents, vmin=-27, vmax=-24)

        axs_ieq.text(0.88, 0.1, 'IEQ', transform=axs_ieq.transAxes,
                     color='black', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        axs_ieq.set_xlabel('z (' + distance_unit_str + ')', fontsize=13)
        axs_ieq.set_ylabel('R (' + distance_unit_str + ')', fontsize=13)

        # Add a color bar for the entire figure (to represent the cooling function intensity)
        cbar_ax = fig.add_axes([0.18, 0.93, 0.66, 0.02])  # Position of colorbar
        fig.colorbar(image, cax=cbar_ax, orientation='horizontal', ticks=MultipleLocator(1.0))

        data.close()  # Close the data to free resources

        # Save the figure for the current timestep
        plt.subplots_adjust(hspace=0.0)  # Set the vertical spacing to zero
        fig.savefig(outfile, bbox_inches='tight', dpi=100)
        print(f" Saved snapshot to {filename}")
        plt.close(fig)

        del neq_lines_emissivity_dict
        del ieq_lines_emissivity_dict

        # Update runtime
        run_finish_time = time.time()
        dt = run_finish_time - run_start_time
        runtime += dt
        print(f" runtime: {runtime:.4e} s | step runtime: {dt:.4e} s")
