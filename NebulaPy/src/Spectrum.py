from .Chianti import chianti
from NebulaPy.src.Utils import getPionSymbol
import NebulaPy.src.Chianti as nebula_chianti
import NebulaPy.src.Constants as const
import os
from NebulaPy.src.EmissionMeasure import emissionMeasure
import multiprocessing as mp
import queue
import traceback
import numpy as np
from NebulaPy.src.LoggingConfig import NebulaError, get_logger
from NebulaPy.src.Progress import Progress, track

logger = get_logger(__name__)

import ChiantiPy.tools.filters as chfilters

##############################################################################
# Worker function
# Keep this outside the class
##############################################################################
def compute_row_spectrum(workerQ, doneQ, spectrum_obj, timeout):
    """Worker function to compute spectra for each grid row."""

    while True:
        level = None
        row = None
        try:
            task = workerQ.get(timeout=timeout)

            if task is None:
                break

            level, row, row_temperature, row_ne = task

            row_species_nonFBCoeff, row_species_FBCoeff = (
                spectrum_obj.compute_species_spectra_coeff(
                    row_temperature=row_temperature,
                    row_ne=row_ne,
                )
            )

            doneQ.put(
                (
                    "RESULT",
                    level,
                    row,
                    row_species_nonFBCoeff,
                    row_species_FBCoeff,
                )
            )

        except queue.Empty:
            break

        except Exception as error:
            doneQ.put(
                (
                    "ERROR",
                    level,
                    row,
                    f"multiprocessing worker failed: {error}",
                    traceback.format_exc(),
                )
            )
            break


class spectrum:

    ######################################################################################
    #
    ######################################################################################
    def __init__(
            self,
            min_wavelength=None,
            max_wavelength=None,
            min_photon_energy=None,
            max_photon_energy=None,
            doBremsstrahlung=False,
            doFreebound=False,
            doLine=False,
            doTwophoton=False,
            elements=None,
            filtername=None,
            filterfactor=None,
            allLines=True,
            userGrid=False,
            MPNcores=4,
            gridSize=1000,
            progress=True,
    ):

        # flags and parameters
        self.N_wvl = None
        self.bremsstrahlung = doBremsstrahlung
        self.freebound = doFreebound
        self.line = doLine
        self.twophoton = doTwophoton
        self.filtername = filtername
        self.filterfactor = filterfactor
        self.allLines = allLines
        self.progress = progress


        logger.info("Initializing spectrum calculation")

        # wavelength and photon energy inputs
        if min_wavelength is not None and max_wavelength is not None:
            self.min_wvl, self.max_wvl = min_wavelength, max_wavelength

        elif min_photon_energy is not None and max_photon_energy is not None:
            self.min_wvl = const.KEV_ANGSTROM / max_photon_energy
            self.max_wvl = const.KEV_ANGSTROM / min_photon_energy

        else:
            raise NebulaError(
                "Provide either wavelength range or photon energy range."
            )

        if self.min_wvl <= 0.0 or self.max_wvl <= 0.0:
            raise NebulaError("Wavelength bounds must be positive.")

        if self.min_wvl >= self.max_wvl:
            raise NebulaError(
                "Minimum wavelength must be smaller than maximum wavelength."
            )

        if not (doBremsstrahlung or doFreebound or doLine or doTwophoton):
            raise NebulaError("No emission processes specified")

        # Verbose output
        enabled_processes = [
            name
            for enabled, name in (
                (doBremsstrahlung, "bremsstrahlung"),
                (doFreebound, "free-bound"),
                (doLine, "lines"),
                (doTwophoton, "two-photon"),
            )
            if enabled
        ]
        logger.info("Radiative processes: %s", ", ".join(enabled_processes))


        # Initialize empty attributes for species and elemental abundances
        self.chianti_species_attributes = {}
        self.build_species_attributes(elements)

        # setup wavelength grid #####################################
        self.WavelengthGrid = []
        self.userGrid = userGrid
        self.gridSize = gridSize

        # Multiprocessing  ##########################################
        self.proc = min(MPNcores, mp.cpu_count())
        logger.info("Multiprocessing: using %s of %s available CPU cores", self.proc, mp.cpu_count())



    ######################################################################################
    # Build Species Attributes
    ######################################################################################
    def build_species_attributes(self, elements):
        """
        Build species attributes.

        Parameters
        ----------
        elements : list
            Elements used in the spectral calculation.
        """
        #todo: only chianti species is implemented. PyNeb species is not implemented yet.

        # Initialize CHIANTI object (dummy plasma state)
        chianti_spec = chianti(
            pion_elements=elements,
            temperature=[1.0e7],  # dummy temperature
            ne=[1.0e9],  # dummy density
        )
        # Return chianti ion attributes for the species
        self.chianti_species_attributes = chianti_spec.species_attributes_container
        # do not terminate rather del
        del chianti_spec

    ######################################################################################
    # SETUP WAVELENGTH GRID
    ######################################################################################
    def setup_wavelength_grid(self, min_wvl, max_wvl, user_grid=False, grid_size=None):

        if not user_grid:
            logger.debug("Setting up the default CHIANTI wavelength grid")
        else:
            logger.debug("Setting up a uniform wavelength grid")

        if min_wvl >= max_wvl:
            raise NebulaError(
                " Minimum wavelength must be smaller than maximum wavelength."
            )

        if not self.chianti_species_attributes:
            raise NebulaError(
                " Species Attributes Container is not initialized or is empty."
            )

        dummy_temperature = [2.0e6]
        dummy_ne = [1.0e9]
        lines = []

        species_list = list(self.chianti_species_attributes.items())

        for species, attributes in track(
                species_list,
                description="Retrieving CHIANTI wavelength grid",
                unit="species",
                enabled=self.progress,
        ):

            if 'line' not in attributes['keys']:
                continue

            chianti_nebula_object = nebula_chianti.chianti(
                chianti_ion=species,
                temperature=dummy_temperature,
                ne=dummy_ne,
            )

            try:
                ion_lines = chianti_nebula_object.get_allLines()

                if ion_lines is not None:
                    lines.extend(
                        np.asarray(ion_lines, dtype=np.float64).tolist()
                    )

            finally:
                del chianti_nebula_object

        # ------------------------------------------------------------------
        # CHIANTI line-based wavelength grid
        # ------------------------------------------------------------------
        if not user_grid:

            lines = np.asarray(lines, dtype=np.float64)

            if lines.size == 0:
                raise NebulaError(
                    " No CHIANTI lines found for the selected species."
                )

            selected_lines = lines[
                (lines >= min_wvl) &
                (lines <= max_wvl)
                ]

            wavelength_grid = np.unique(
                np.concatenate((
                    selected_lines,
                    np.asarray(
                        [min_wvl, max_wvl],
                        dtype=np.float64
                    )
                ))
            )

        # ------------------------------------------------------------------
        # User-defined uniform wavelength grid
        # ------------------------------------------------------------------
        elif grid_size is not None:

            if grid_size < 2:
                raise NebulaError(
                    " grid_size must be at least 2."
                )

            wavelength_grid = np.linspace(
                min_wvl,
                max_wvl,
                int(grid_size),
                dtype=np.float64
            )

        # ------------------------------------------------------------------
        # No valid grid option
        # ------------------------------------------------------------------
        else:
            raise NebulaError(
                " Either use_chianti_grid=True or grid_size must be specified."
            )

        wavelength_grid.sort()

        if wavelength_grid.size == 0:
            raise NebulaError(
                " No wavelength points found in the requested wavelength range."
            )

        self.WavelengthGrid = wavelength_grid
        self.N_wvl = wavelength_grid.size

        logger.info(
            "Wavelength grid: %.4f–%.4f Å, %s spectral points",
            self.WavelengthGrid[0],
            self.WavelengthGrid[-1],
            self.N_wvl,
        )


    ######################################################################################
    # Print Line Cataloger
    ######################################################################################
    def line_cataloger(self, Filebase="", OutDir=""):

        if not Filebase:
            raise NebulaError(
                "line_cataloger: output file base name not specified."
            )

        if not OutDir:
            raise NebulaError(
                "line_cataloger: output directory not specified."
            )

        if not os.path.isdir(OutDir):
            raise NebulaError(
                f"line_cataloger: output directory does not exist: {OutDir}"
            )

        outfile = os.path.join(
            OutDir,
            f"{Filebase}_LineCatalog.txt"
        )

        dummy_temperature = [2.0e6]
        dummy_ne = [1.0e9]

        line_catalog = []

        species_list = list(self.chianti_species_attributes.items())

        for species, attributes in track(
                species_list,
                description="Cataloging CHIANTI wavelength grid",
                unit="species",
                enabled=self.progress,
        ):

            if 'line' not in attributes['keys']:
                continue

            chianti_nebula_object = nebula_chianti.chianti(
                chianti_ion=species,
                temperature=dummy_temperature,
                ne=dummy_ne,
            )

            try:
                ionTransitions = (
                    chianti_nebula_object.get_allLineTransitions()
                )

                spectroscopic_name = (
                    chianti_nebula_object.chianti_ion.Spectroscopic
                )

                wavelengths_all = np.asarray(ionTransitions['wvl'])
                Avalues_all = np.asarray(ionTransitions['Avalue'])
                lower_states_all = np.asarray(
                    ionTransitions['Lower'],
                    dtype=object
                )
                upper_states_all = np.asarray(
                    ionTransitions['Upper'],
                    dtype=object
                )

                mask = (
                        (wavelengths_all >= self.min_wvl)
                        &
                        (wavelengths_all <= self.max_wvl)
                )

                if not np.any(mask):
                    continue

                wavelengths = wavelengths_all[mask]
                Avalues = Avalues_all[mask]
                lower_states = lower_states_all[mask]
                upper_states = upper_states_all[mask]

                for wvl, aval, lower, upper in zip(
                        wavelengths,
                        Avalues,
                        lower_states,
                        upper_states
                ):
                    energy_keV = const.KEV_ANGSTROM / wvl
                    spectral_line = f"{spectroscopic_name} {wvl:.6f}"

                    line_catalog.append(
                        (
                            wvl,
                            spectral_line,
                            energy_keV,
                            aval,
                            str(lower),
                            str(upper)
                        )
                    )

            finally:
                del chianti_nebula_object

        line_catalog.sort(key=lambda entry: entry[0])

        with open(outfile, "w") as f:

            f.write("# CHIANTI Line Catalogue\n")
            f.write(
                f"# Wavelength range: "
                f"{self.min_wvl:.3f} - {self.max_wvl:.3f} Å\n"
            )

            f.write(
                f"{'Spectral line':<35} "
                f"{'Energy[keV]':>12} "
                f"{'A-value':>15} "
                f"{'Lower':>20} "
                f"{'Upper':>20}\n"
            )

            f.write("-" * 110 + "\n")

            for _, spectral_line, energy_keV, aval, lower, upper in line_catalog:
                f.write(
                    f"{spectral_line:<35} "
                    f"{energy_keV:12.6f} "
                    f"{aval:15.6e} "
                    f"{lower:>20} "
                    f"{upper:>20}\n"
                )

        logger.debug("Saved CHIANTI line catalogue to %s", outfile)

    ######################################################################################
    # COMPUTE SPECIES SPECTRA RATE
    ######################################################################################
    def compute_species_spectra_coeff(self, row_temperature, row_ne):

        # Determine the number of temperature values
        N_temp = len(row_temperature)

        '''
        species_spectra_coefficients = {}
        '''

        # Store spectra by emission process
        species_nonfb_coefficients = {}
        species_fb_coefficients = {}

        # info: looping over species to calculate the emission rate from each process
        for species in self.chianti_species_attributes.keys():

            Z = self.chianti_species_attributes[species]['Z']
            ionstage = self.chianti_species_attributes[species]['Ion']
            dielectronic = self.chianti_species_attributes[species]['Dielectronic']

            # reset process arrays for each species
            bremsstrahlung_coefficients = np.zeros((N_temp, self.N_wvl), dtype=np.float64)
            freebound_coefficients = np.zeros((N_temp, self.N_wvl), dtype=np.float64)
            line_coefficients = np.zeros((N_temp, self.N_wvl), dtype=np.float64)
            twophoton_coefficients = np.zeros((N_temp, self.N_wvl), dtype=np.float64)

            #if species not in ['fe_25', 'fe_26', 'si_14', 'si_13', 's_16']:
            #if species not in ['h_2']:
            #    #logger.warning(f"{count} Skipping {species} ...")
            #    continue

            processes = self.chianti_species_attributes[species]['keys']
            CHIANTI = chianti(
                chianti_ion=species,
                temperature=row_temperature,
                ne=row_ne,
            )

            # Bremsstrahlung (free-free emission)
            if self.bremsstrahlung and 'ff' in processes:
                bremsstrahlung_coefficients = CHIANTI.get_bremsstrahlung_coefficients(
                    wavelength=self.WavelengthGrid
                )

            # Free-bound emission
            if self.freebound and 'fb' in processes:
                freebound_coefficients = CHIANTI.get_freebound_coefficients(
                    wavelength=self.WavelengthGrid
                )

            # Line emission
            if self.line and 'line' in processes:
                line_coefficients = CHIANTI.get_line_coefficients(
                    wavelength=self.WavelengthGrid
                )
                # dividing by ne to obtain the same value returned by CHIANTI
                line_coefficients = line_coefficients / row_ne[:, None]

            # Two-photon emission
            if self.twophoton and 'line' in processes:
                if (Z - ionstage) in [0, 1] and not dielectronic:
                    twophoton_coefficients = CHIANTI.get_twophoton_coefficients(
                        wavelength=self.WavelengthGrid
                    )

            CHIANTI.terminate()

            '''
            # sum over all processes for this species
            species_emission_coeff = (
                    bremsstrahlung_coefficients
                    + freebound_coefficients
                    + line_coefficients
                    + twophoton_coefficients
            )

            species_spectra_coefficients[species] = species_emission_coeff * row_grid_mask[:, None]
            '''

            # Free-free + line + two-photon emission
            nonfb_coefficients = (
                    bremsstrahlung_coefficients
                    + line_coefficients
                    + twophoton_coefficients
            )

            species_nonfb_coefficients[species] = nonfb_coefficients

            # Free-bound emission only
            species_fb_coefficients[species] = freebound_coefficients

        '''
        return species_spectra_coefficients
        '''
        return species_nonfb_coefficients, species_fb_coefficients


    ######################################################################################
    # generate spectrum for 2D data
    ######################################################################################
    def generate_spectrum(
            self,
            temperature,
            ne,
            species_densities,
            grid_volume,
            grid_mask
    ):

        ##########################################################################
        # Check whether the CHIANTI species attribute is initialized
        if not self.chianti_species_attributes:
            raise NebulaError(
                "Species Attributes Container is not initialized or is empty."
            )

        ##########################################################################
        # Check input arrays
        temperature = np.asarray(temperature, dtype=np.float64)
        ne = np.asarray(ne, dtype=np.float64)
        grid_volume = np.asarray(grid_volume, dtype=np.float64)
        grid_mask = np.asarray(grid_mask, dtype=np.float64)

        if not (
                temperature.shape == ne.shape ==
                grid_volume.shape == grid_mask.shape
        ):
            raise NebulaError(
                "Input arrays have inconsistent shapes."
            )

        species_densities = {
            species: np.asarray(density, dtype=np.float64)
            for species, density in species_densities.items()
        }

        for species, density in species_densities.items():

            if density.shape != temperature.shape:
                raise NebulaError(
                    f"Species density input arrays have inconsistent "
                    f"shapes for species {species}."
                )


        ##########################################################################
        # Setting Up wavelength grid
        self.setup_wavelength_grid(self.min_wvl, self.max_wvl,
                                   user_grid=self.userGrid,
                                   grid_size=self.gridSize)

        ##########################################################################
        # DEM calculation
        # initialize emission measure class
        EM = emissionMeasure(
            Tmin=100,
            Tmax=1.0e9,
            Nbins=300,
            progress=self.progress,
        )
        # generate DEM
        EM.DEM2D(temperature=temperature,
                 ne=ne, speciesDensities=species_densities,
                 volume=grid_volume, gridMask=grid_mask)

        ##########################################################################
        # Allocate only binned spectra
        coefficient_shape = (EM.Nbins, self.N_wvl)

        species_nonfb_spectra = {
            species: np.zeros(coefficient_shape, dtype=np.float64)
            for species in species_densities
        } if (self.bremsstrahlung or self.line or self.twophoton) else {}

        species_fb_spectra = {
            species: np.zeros(coefficient_shape, dtype=np.float64)
            for species in species_densities
        } if self.freebound else {}

        ##########################################################################
        # Map emitting ion q to recombining ion q+1
        recombining_species = {}

        for chianti_species, attributes in self.chianti_species_attributes.items():
            higher_ion = attributes.get("higher", 0)

            if higher_ion == 0:
                continue

            emitting_ion = getPionSymbol(chianti_species)
            recombining_ion = getPionSymbol(higher_ion)
            recombining_species[emitting_ion] = recombining_ion

        ##########################################################################
        # Multiprocessing row spectra calculation
        N_grid_level = len(temperature)

        # uniform grid or 1 level grid -------------------------------------------
        if N_grid_level == 1:
            logger.warning("Uniform grid not implemented.")

        # multilevel grid ---------------------------------------------------------
        else:
            timeout = 0.1

            # Gathering all task and configuring the No of multiprocessing cores ==
            AllTasks = []
            for level in range(N_grid_level):
                for row in range(len(temperature[level])):
                    AllTasks.append(
                        (level, row,
                         temperature[level, row],
                         ne[level, row])
                    )
            Ntasks = len(AllTasks)

            proc = min(self.proc, Ntasks)

            logger.info(
                "Multiprocessing: %s tasks, %s levels, %s grid slices per level",
                Ntasks,
                N_grid_level,
                len(temperature[0]),
            )

            # Creating worker and done queues for multiprocessing =================
            workerQ = mp.Queue()
            doneQ = mp.Queue()

            for task in AllTasks:
                workerQ.put(task)

            for _ in range(proc):
                workerQ.put(None)

            # Initializing Worker Processes =======================================
            processes = []

            for _ in range(proc):
                p = mp.Process(
                    target=compute_row_spectrum,
                    args=(
                        workerQ,
                        doneQ,
                        self,
                        timeout
                    )
                )

                p.start()
                processes.append(p)

            completed = 0

            logger.debug("Computing spectrum")

            with Progress(
                    "Computing species spectra",
                    Ntasks,
                    unit="tasks",
                    enabled=self.progress,
            ) as progress:

                while completed < Ntasks:

                    (
                        status,
                        level,
                        row,
                        nonFBCoeff_or_error,
                        FBCoeff_or_traceback,
                    ) = doneQ.get()

                    if status == "ERROR":
                        for p in processes:
                            if p.is_alive():
                                p.terminate()
                        for p in processes:
                            p.join()

                        logger.error(
                            "Spectrum worker failed at grid level %s, row %s: %s",
                            level,
                            row,
                            nonFBCoeff_or_error,
                        )
                        logger.debug(
                            "Spectrum worker traceback:\n%s",
                            FBCoeff_or_traceback,
                        )
                        raise NebulaError(nonFBCoeff_or_error)

                    if status != "RESULT":
                        for p in processes:
                            if p.is_alive():
                                p.terminate()
                        for p in processes:
                            p.join()
                        raise NebulaError(
                            f"Unknown multiprocessing result status: {status!r}"
                        )

                    row_species_nonfb_coefficients = nonFBCoeff_or_error
                    row_species_fb_coefficients = FBCoeff_or_traceback

                    row_mask = grid_mask[level, row][:, None]

                    row_species_nonfb_coefficients = {
                        species: coefficients * row_mask
                        for species, coefficients
                        in row_species_nonfb_coefficients.items()
                    }

                    row_species_fb_coefficients = {
                        species: coefficients * row_mask
                        for species, coefficients
                        in row_species_fb_coefficients.items()
                    }

                    row_bins = EM.DEM_indices[level, row]

                    occupied_bins = np.unique(
                        row_bins[(row_bins >= 0) & (row_bins < EM.Nbins)]
                    )

                    for bin_idx in occupied_bins:
                        bin_mask = row_bins == bin_idx

                    # Only CHIANTI ions are currently supported by the spectrum calculation.
                    # PyNeb species are not yet implemented.
                        for chianti_species in self.chianti_species_attributes:
                            pion_species = getPionSymbol(chianti_species)

                            if (
                                    pion_species not in species_nonfb_spectra
                                    and pion_species not in species_fb_spectra
                            ):
                                continue

                            if pion_species in species_nonfb_spectra:
                                nonfb_coefficients = row_species_nonfb_coefficients.get(
                                    chianti_species
                                )

                                if nonfb_coefficients is not None:
                                    species_nonfb_spectra[pion_species][bin_idx, :] += np.sum(
                                        nonfb_coefficients[bin_mask, :],
                                        axis=0,
                                    )

                            if pion_species in species_fb_spectra:
                                fb_coefficients = row_species_fb_coefficients.get(
                                    chianti_species
                                )

                                if fb_coefficients is not None:
                                    species_fb_spectra[pion_species][bin_idx, :] += np.sum(
                                        fb_coefficients[bin_mask, :],
                                        axis=0,
                                    )


                    completed += 1
                    progress.advance()

            for p in processes:
                p.join()

        ##########################################################################
        # Multiply by DEM and sum over temperature bins
        Spectrum = {}

        spectrum_species = (
                set(species_nonfb_spectra)
                | set(species_fb_spectra)
        )

        for species in spectrum_species:
            species_spectrum = np.zeros(self.N_wvl, dtype=np.float64)

            # Non-free-bound emission from ion q uses DEM[q].
            nonfb_spectra = species_nonfb_spectra.get(species)

            if nonfb_spectra is not None and species in EM.DEM:
                species_spectrum += np.sum(
                    nonfb_spectra * EM.DEM[species][:, None],
                    axis=0,
                )

            # Free-bound emission from ion q uses DEM[q+1].
            fb_spectra = species_fb_spectra.get(species)
            recombining_ion = recombining_species.get(species)

            if (
                    fb_spectra is not None
                    and recombining_ion in EM.DEM
            ):
                species_spectrum += np.sum(
                    fb_spectra * EM.DEM[recombining_ion][:, None],
                    axis=0,
                )

            Spectrum[species] = species_spectrum

        if Spectrum:
            integrated_spectrum = np.sum(
                np.stack(list(Spectrum.values())),
                axis=0,
            )
        else:
            integrated_spectrum = np.zeros(
                self.N_wvl,
                dtype=np.float64,
            )

        self.Spectrum = integrated_spectrum
