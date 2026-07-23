#import numpy as np
import NebulaPy.src as nebula
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import configure_logging, get_logger

# constants
cm2au = 6.68459e-14  # cm to au conversion factor
x_axis = "wavelength"  # Options: "wavelength" or "energy"
y_axis = "photon_per_energy"  # See plotting options below

'''
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
'''


# Colliding wind binaries
#Macbook -> Set up paths and filenames
OutputDir = '/Users/tony/Desktop/CWBs-NEMOv1/Post-Processing/WR140Test'  # Output image directory
SiloDir = '/Users/tony/Desktop/CWBs-NEMOv1/Silo-n128'  # Directory containing silo files
Filebase = 'wr140_NEMO_d07e13_d2l6n128'  # Base name of the silo files
start_time = 14.35  # days
finish_time = None
time_unit = 'days'
out_frequency = None
SimulationName = "WR140"


'''
# Bowshock
#Macbook -> Set up paths and filenames
OutputDir = '/Users/tony/Desktop/Bowshock-Xray/Post-Processing/XraySpecTest2'  # Output image directory
SiloDir = '/Users/tony/Desktop/Bowshock-Xray/silo'  # Directory containing silo files
Filebase = 'Ostar_mhd-nemo-dep_d2n0128l3'  # Base name of the silo files
start_time = 165  # days
finish_time = None
time_unit = 'kyr'
out_frequency = None
SimulationName = "Bowshock"
'''

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

def main():
    """Generate spectra without re-running this workflow in spawned workers."""

    configure_logging(level="Info", log_to_file=False, log_file=f"{Filebase}.log")
    logger = get_logger()
    logger.info(f"Running X-ray Spectrum Test Suite")
    logger.info(f"Filebase: {Filebase}")

    # Batch the silo files according to the time instant
    batched_silos = nebula.Silo.batch(
        SiloDir,
        Filebase,
        start_time=start_time,
        finish_time=finish_time,
        time_unit=time_unit,
        out_frequency=out_frequency
    )

    key = input("Press 'y' to continue, anything else to exit: ").strip().lower()

    if key == "y":
        pass
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
    ion_list = ['Fe25+']

    # initializing spectrum class
    NebulaSpectrum = nebula.spectrum(
        min_wavelength=1.0,  # Minimum wavelength in Angstroms
        max_wavelength=100,  # Maximum wavelength in Angstroms
        min_photon_energy=None,  # Minimum photon energy in keV # not implemented
        max_photon_energy=None,  # Maximum photon energy in keV # not implemented
        elements=elements,
        ion_list=None,
        doBremsstrahlung=True,
        doFreebound=True,
        doLine=True,
        doTwophoton=True,
        filtername=None,
        filterfactor=None,
        userGrid=True,
        gridSize=3000,
        allLines=True,
        MPNcores=8,
    )

    runtime = 0.0

    # Loop over each time instant in the batched silo files
    for step, silo_instant in enumerate(batched_silos):

        silo_instant_start_time = time.time()

        sim_time = pion.get_simulation_time(silo_instant, time_unit=time_unit)
        logger.info(
            "Simulation snapshot: step %s, time %.6e %s",
            step,
            sim_time.value,
            sim_time.unit,
        )

        # Extract temperature and electron number density
        '''
        temperature = np.asarray(
            pion.get_parameter('Temperature', silo_instant),
            dtype=np.float64
        )
        ne = pion.get_ne(silo_instant)
        species_densities = pion.get_species_number_densities(
            silo_instant,
            ion_list=NebulaSpectrum.required_density_ions,
        )
        '''

        temperature = np.asarray([[[1.0e7, 1.0e7]]])
        ne = np.asarray([[[1.0e9, 1.0e9]]])
        grid_volume = np.asarray([[[1.0, 1.0]]])
        grid_mask = np.asarray([[[1.0, 0.0]]])

        # Solar-like dummy composition. Convert the prescribed electron number
        # density to mass density using the fully ionized electron-per-gram
        # abundance, then distribute each element among its CIE ion stages.
        element_mass_fractions = {
            "H": 0.7381,
            "He": 0.2485,
            "C": 0.0024,
            "N": 0.0007,
            "O": 0.0057,
            "Ne": 0.0013,
            "Si": 0.0007,
            "S": 0.0004,
            "Fe": 0.0013,
        }

        fully_ionized_electrons_per_gram = sum(
            element_mass_fractions[element]
            * const.ATOMIC_NUMBER[element]
            / const.ATOMIC_MASS[element]
            for element in element_mass_fractions
        )
        gas_mass_density = ne / fully_ionized_electrons_per_gram

        cie_ion_balance = nebula.cieMode()
        species_densities = cie_ion_balance.build_cie_number_densities(
            element_mass_fractions=element_mass_fractions,
            temperature=temperature,
            density=gas_mass_density,
        )

        NebulaSpectrum.generate_spectrum(
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

        logger.info("Saved spectrum data to %s", txtfile)

        fig, ax = plt.subplots(figsize=(10, 5))

        kev_angstrom = 12.39841984
        kev_to_erg = 1.602176634e-9
        energy = kev_angstrom / wavelength

        if y_axis == "energy_per_wavelength":
            y_values = spectrum
            ylabel = r"$L_\lambda$ [erg s$^{-1}$ $\AA^{-1}$]"
        elif y_axis == "energy_per_energy":
            y_values = spectrum * kev_angstrom / energy**2
            ylabel = r"$L_E$ [erg s$^{-1}$ keV$^{-1}$]"
        elif y_axis == "photon_per_wavelength":
            y_values = spectrum / (energy * kev_to_erg)
            ylabel = r"$N_\lambda$ [photons s$^{-1}$ $\AA^{-1}$]"
        elif y_axis == "photon_per_energy":
            energy_luminosity = spectrum * kev_angstrom / energy**2
            y_values = energy_luminosity / (energy * kev_to_erg)
            ylabel = r"$N_E$ [photons s$^{-1}$ keV$^{-1}$]"
        else:
            raise ValueError(
                "y_axis must be 'energy_per_wavelength', "
                "'energy_per_energy', 'photon_per_wavelength', "
                "or 'photon_per_energy'."
            )

        if x_axis == "wavelength":
            plot_x = wavelength
            plot_y = y_values
            xlabel = r"Wavelength [$\AA$]"
        elif x_axis == "energy":
            energy_order = np.argsort(energy)
            plot_x = energy[energy_order]
            plot_y = y_values[energy_order]
            xlabel = "Photon energy [keV]"
        else:
            raise ValueError(
                "x_axis must be either 'wavelength' or 'energy'."
            )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        ax.plot(
            plot_x,
            plot_y,
            color="green",
            linewidth=1.4,
        )

        enabled_processes = [
            name
            for enabled, name in (
                (NebulaSpectrum.bremsstrahlung, "free-free"),
                (NebulaSpectrum.freebound, "free-bound"),
                (NebulaSpectrum.line, "lines"),
                (NebulaSpectrum.twophoton, "two-photon"),
            )
            if enabled
        ]
        ion_description = (
            "all ions"
            if NebulaSpectrum.ion_list is None
            else ", ".join(NebulaSpectrum.ion_list)
        )
        plot_information = "\n".join((
            f"Simulation: {SimulationName}",
            f"Time: {sim_time.value:.4g} {sim_time.unit}",
            f"Elements: {', '.join(elements)}",
            f"Ions: {ion_description}",
            f"Processes: {', '.join(enabled_processes)}",
            (
                f"Wavelength range: {wavelength[0]:.3g}–"
                f"{wavelength[-1]:.3g} Å"
            ),
        ))
        information_box = ax.legend(
            [],
            [],
            title=plot_information,
            loc="best",
            frameon=True,
            fancybox=True,
            framealpha=0.85,
        )
        information_box.get_title().set_fontsize(9)
        information_box.get_title().set_multialignment("left")
        information_box.get_frame().set_facecolor("white")
        information_box.get_frame().set_edgecolor("0.4")

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

        fig.tight_layout()

        outfile = os.path.join(OutputDir, f"{Filebase}_Spectrum_{sim_time.value:6e}.png")

        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            "Saved snapshot %s at simulation time %.6e %s to %s",
            step,
            sim_time.value,
            sim_time.unit,
            outfile,
        )

        # Update runtime
        dt = time.time() - silo_instant_start_time
        runtime += dt

        logger.info("Runtime: total %.4e s, snapshot %.4e s", runtime, dt)

    logger.info("X-ray spectrum workflow completed in %.4e s", runtime)


if __name__ == "__main__":
    main()
