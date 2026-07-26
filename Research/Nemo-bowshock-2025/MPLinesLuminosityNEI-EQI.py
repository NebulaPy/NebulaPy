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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")

# Set OMP_NUM_THREADS=1 to ensure single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"

def compute_luminosity(workerQ, doneQ, timeout=0.1):
    """Worker function to compute luminosity for each ion"""
    import queue  # Import inside function to avoid errors

    while True:
        try:
            task = workerQ.get(timeout=timeout)  # Get task from queue
            if task is None:
                #print("[Worker] Received stop signal, exiting.")
                break  # Exit if None is received (stop signal)

            ion_name, line_emission, lines, species_density, temperature, ne, cell_volume, mask = task

            print(f" Multiprocessing: computing the luminosity of {ion_name:<4} lines")

            if ion_name == 'H':

                luminosity = line_emission.recombination_line_luminosity_2D(
                    lines=lines,
                    temperature=temperature, ne=ne,
                    species_density=species_density,
                    cell_volume=cell_volume,
                    grid_mask=mask, progress_bar=False)
            else:
                luminosity = line_emission.line_luminosity_2D(
                    lines=lines,
                    temperature=temperature, ne=ne,
                    species_density=species_density,
                    cell_volume=cell_volume,
                    grid_mask=mask, progress_bar=False)

            print(f" Multiprocessing: finished computing the luminosity for {ion_name:<4} lines")
            doneQ.put({ion_name: luminosity})  # Store result
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
    task_packet: Dict[str, Tuple[Any, Any, Any, Any, Any, Any, Any]],
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
            (line_emission, lines, species_density,
             temperature, ne, cell_volume, grid_mask) = payload

            workerQ.put(
                (ion_name, line_emission, lines, species_density,
                 temperature, ne, cell_volume, grid_mask)
            )

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
    filename = neq_filebase + '_Llines_NEQ_IEQ.txt'
    outfile = os.path.join(output_dir, filename)



    # All input lines
    ion_lines = {
        "H": [6562.816, 4861.332, 18750.960, 21655.283, 1215.670], # Hydrogen recombination lines
        "He1+": [303.78, 303.786, 256.317, 256.318, 243.026, 243.027],
        "C2+": [1906.683, 1908.734, 977.02],
        "N1+": [6585.273, 6549.861, 1218026.8, 2053388.09],
        "N2+": [573394.5, 989.799, 1752.16, 1749.674, 1753.995, 1748.646],
        "O1+": [3729.844, 3727.092, 7331.722, 2470.97, 2471.094, 7321.094, 7322.177, 7332.808],
        "O2+": [5008.24, 883564.0, 518145.0, 4960.295, 1666.15, 4364.436, 832.929, 833.715, 1660.809],
        "Ne1+": [128139.42],
        "Ne2+": [155545.19, 3869.849, 3968.585, 360230.55, 489.495, 491.041],
        "S1+": [6718.295, 6732.674, 4069.749, 10323.317, 4077.5],
        "S2+": [335008.38, 9532.252, 187055.74, 9070.048, 6313.649, 3722.454],
        "S3+": [105104.95],
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
        "Multiprocessing computation of temporal evolution of spectral-line luminosities",
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
        if hasattr(le, "chianti_line_batch_check"):
            if ion == 'H':
                le.pyneb_line_batch_check(lines)
            else:
                le.chianti_line_batch_check(lines)

        # Store for later luminosity / time-evolution calculations
        line_emission_objects[ion] = le

    #######################################################
    # Get geometry information
    neq_geometry = neq_pion.geometry_container
    N_grid_level_neq = neq_geometry['Nlevel']
    grid_mask_neq = neq_geometry['mask']
    cell_volume_neq = neq_pion.get_cylindrical_cell_volume().value

    ieq_geometry = ieq_pion.geometry_container
    N_grid_level_ieq = ieq_geometry['Nlevel']
    grid_mask_ieq = ieq_geometry['mask']
    cell_volume_ieq = ieq_pion.get_cylindrical_cell_volume().value


    # Write initial header to the output file
    with open(outfile, "w") as file:
        file.write(f"#File generated by {util.nebula_version()}\n\n")
        file.write("#Task: Multiprocessing of Spectral-line luminosities\n")
        file.write("       Comparison: NEQ vs IEQ Emission\n")
        file.write(f"#PION Simulation: NEMO Bowshock 2025\n")
        file.write(f"#PION Simulation Filebase: {neq_filebase}\n")
        file.write(f"#PION Simulation Mask: Grid Mask\n")

        file.write("# Dataset Description:\n")
        file.write("# This dataset provides spectral-line luminosities computed under\n")
        file.write("# non-equilibrium (NEQ) and ionisation-equilibrium (IEQ) conditions.\n")
        file.write("# Each row corresponds to a specific spectral line, with:\n")
        file.write("#  - The first column listing the line wavelength (in Angstrom).\n")
        file.write("#  - The second column giving the line luminosity in the NEQ case (erg/s).\n")
        file.write("#  - The third column giving the corresponding line luminosity in the IEQ case (erg/s).\n\n")

    # Loop over each time instant in the batched silo files
    runtime = 0.0
    write_heading = True
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
                neq_pion.get_ion_number_density(
                    "H1+" if ion == "H" else ion,  # <-- remap only here
                    neq_silo
                ),
                neq_temperature,
                neq_ne,
                cell_volume_neq,
                grid_mask_neq
            )
            for ion in ion_lines
        }

        neq_lines_luminosity_dict = run_multiprocessing_task(
            task_packet=neq_task_packet,
            task=compute_luminosity,
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
                ieq_pion.get_ion_number_density(
                    "H1+" if ion == "H" else ion,  # <-- remap only here
                    ieq_silo
                ),
                ieq_temperature,
                ieq_ne,
                cell_volume_ieq,
                grid_mask_ieq
            )
            for ion in ion_lines
        }

        ieq_lines_luminosity_dict = run_multiprocessing_task(
            task_packet=ieq_task_packet,
            task=compute_luminosity,
        )

        print(f" Saving data to file: {filename}")
        # Column widths
        W_LINE = 18
        W_LUM = 18
        # Write header once
        if write_heading:
            with open(outfile, "a") as file:
                file.write(
                    f"{'#Spectral Line':<{W_LINE}}"
                    f"{'Luminosity NEQ':>{W_LUM}}"
                    f"{'Luminosity IEQ':>{W_LUM}}\n"
                )
            write_heading = False

        # Append data
        with open(outfile, "a") as file:
            for line in sorted(neq_lines_luminosity_dict.keys()):
                l_neq = neq_lines_luminosity_dict[line]
                l_ieq = ieq_lines_luminosity_dict.get(line, 0.0)

                file.write(
                    f"{line:<{W_LINE}}"
                    f"{l_neq:>{W_LUM}.6e}"
                    f"{l_ieq:>{W_LUM}.6e}\n"
                )

        del neq_lines_luminosity_dict
        del ieq_lines_luminosity_dict

        # Update runtime
        run_finish_time = time.time()
        dt = run_finish_time - run_start_time
        runtime += dt
        print(f" runtime: {runtime:.4e} s | step runtime: {dt:.4e} s")
