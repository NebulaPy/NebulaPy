from .Chianti import chianti
from NebulaPy.src import Utils as utils
import NebulaPy.src.Chianti as nebula_chianti
import NebulaPy.src.Constants as const
import os
from NebulaPy.src.EmissionMeasure import emissionMeasure
import multiprocessing as mp
import queue
import numpy as np
from tqdm import tqdm

#from .CIE import cieMode
import ChiantiPy.tools.filters as chfilters

##############################################################################
# Worker function
# Keep this outside the class
##############################################################################
def compute_row_spectrum(workerQ, doneQ, spectrum_obj, timeout):
    """Worker function to compute spectra for each grid row."""

    while True:
        try:
            task = workerQ.get(timeout=timeout)

            if task is None:
                break

            level, row, row_temperature, row_ne, row_grid_mask = task

            '''
            row_species_spectra_coefficients = spectrum_obj.computeSpeciesSpectraCoeff(
                row_temperature=row_temperature,
                row_ne=row_ne,
                row_grid_mask=row_grid_mask,
            )
            '''

            row_species_nonFBCoeff, row_species_FBCoeff = (
                spectrum_obj.computeSpeciesSpectraCoeff(
                    row_temperature=row_temperature,
                    row_ne=row_ne,
                    row_grid_mask=row_grid_mask,
                )
            )

            '''
            doneQ.put((level, row, row_species_spectra_coefficients))
            '''

            doneQ.put((level, row, row_species_nonFBCoeff, row_species_FBCoeff))

        except queue.Empty:
            break

        except Exception as e:
            doneQ.put(("ERROR", None, f"multiprocessing worker failed: {e}"))
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
            CIE=False,
            filtername=None,
            filterfactor=None,
            allLines=True,
            userGrid=False,
            MPNcores=4,
            gridSize=1000,
            verbose=True
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
        self.verbose = verbose


        print(" [ SPECTRUM MODULE ] : Initializing spectrum calculation")

        # wavelength and photon energy inputs
        if min_wavelength is not None and max_wavelength is not None:
            self.min_wvl, self.max_wvl = min_wavelength, max_wavelength

        elif min_photon_energy is not None and max_photon_energy is not None:
            self.min_wvl = const.kev2Ang / max_photon_energy
            self.max_wvl = const.kev2Ang / min_photon_energy

        else:
            utils.nebula_exit_with_error(
                "Provide either wavelength range or photon energy range."
            )

        if self.min_wvl <= 0.0 or self.max_wvl <= 0.0:
            utils.nebula_exit_with_error("Wavelength bounds must be positive.")

        if self.min_wvl >= self.max_wvl:
            utils.nebula_exit_with_error(
                "Minimum wavelength must be smaller than maximum wavelength."
            )

        if not (doBremsstrahlung or doFreebound or doLine or doTwophoton):
            utils.nebula_exit_with_error("No emission processes specified")

        # Verbose output
        print(" Radiative process configuration:")
        print(f"   Bremsstrahlung continuum : {'Enabled' if doBremsstrahlung else 'Disabled'}")
        print(f"   Free-bound continuum     : {'Enabled' if doFreebound else 'Disabled'}")
        print(f"   Spectral line emission   : {'Enabled' if doLine else 'Disabled'}")
        print(f"   Two-photon emission      : {'Enabled' if doTwophoton else 'Disabled'}")


        # Initialize empty attributes for species and elemental abundances
        self.chianti_species_attributes = {}
        self.build_species_attributes(elements)

        # setup wavelength grid #####################################
        self.WavelengthGrid = []
        self.userGrid = userGrid
        self.gridSize = gridSize

        # Multiprocessing  ##########################################

        self.proc = min(MPNcores, mp.cpu_count())
        print(
            f" [ MULTIPROCESSING ]: Using {self.proc}/{mp.cpu_count()} available CPU cores"
        )

        #todo: CIE can be input to the generateSpectrum Method rather than Spectrum class itself.
        '''
        self.CIE = CIE
        self.NEQ = not CIE
        if CIE:
            cie = cieMode(verbose=True)
            cie.load_cie()
        '''

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
        # Initialize CHIANTI object (dummy plasma state)
        chianti_spec = chianti(
            pion_elements=elements,
            temperature=[1.0e7],  # dummy temperature
            ne=[1.0e9],  # dummy density
            verbose=self.verbose
        )
        # Return chianti ion attributes for the species
        self.chianti_species_attributes = chianti_spec.species_attributes_container
        # do not terminate rather del
        del chianti_spec

    ######################################################################################
    # SETUP WAVELENGTH GRID
    ######################################################################################
    def setup_wavelength_grid(self, min_wvl, max_wvl, user_grid=False, grid_size=None):

        if self.verbose:
            if not user_grid:
                print(" [ WAVELENGTH GRID ] : Setting up default CHIANTI wavelength grid")
            else:
                print(" [ WAVELENGTH GRID ] : Setting up uniform wavelength grid")

        if min_wvl >= max_wvl:
            utils.nebula_exit_with_error(
                " Minimum wavelength must be smaller than maximum wavelength."
            )

        if not self.chianti_species_attributes:
            utils.nebula_exit_with_error(
                " Species Attributes Container is not initialized or is empty."
            )

        dummy_temperature = [2.0e6]
        dummy_ne = [1.0e9]
        lines = []

        species_list = list(self.chianti_species_attributes.items())

        for species, attributes in tqdm(
                species_list,
                desc=" Retrieving CHIANTI wavelength grid",
                unit=" species",
                ncols=100,
                leave=True,
                disable=self.verbose
        ):

            if 'line' not in attributes['keys']:
                continue

            chianti_nebula_object = nebula_chianti.chianti(
                chianti_ion=species,
                temperature=dummy_temperature,
                ne=dummy_ne,
                verbose=self.verbose
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
                utils.nebula_exit_with_error(
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
                utils.nebula_exit_with_error(
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
            utils.nebula_exit_with_error(
                " Either use_chianti_grid=True or grid_size must be specified."
            )

        wavelength_grid.sort()

        if wavelength_grid.size == 0:
            utils.nebula_exit_with_error(
                " No wavelength points found in the requested wavelength range."
            )

        self.WavelengthGrid = wavelength_grid
        self.N_wvl = wavelength_grid.size

        print(
            f" [ WAVELENGTH GRID ]: "
            f"{self.WavelengthGrid[0]:.4f}–{self.WavelengthGrid[-1]:.4f} Å | "
            f"{self.N_wvl} spectral points"
        )




    ######################################################################################
    # Print Line Cataloger
    ######################################################################################
    def LineCataloger(self, Filebase="", OutDir=""):

        if not Filebase:
            utils.nebula_exit_with_error(
                "LineCataloger: output file base name not specified."
            )

        if not OutDir:
            utils.nebula_exit_with_error(
                "LineCataloger: output directory not specified."
            )

        if not os.path.isdir(OutDir):
            utils.nebula_exit_with_error(
                f"LineCataloger: output directory does not exist: {OutDir}"
            )

        outfile = os.path.join(
            OutDir,
            f"{Filebase}_LineCatalog.txt"
        )

        dummy_temperature = [2.0e6]
        dummy_ne = [1.0e9]

        line_catalog = []

        species_list = list(self.chianti_species_attributes.items())

        for species, attributes in tqdm(
                species_list,
                desc=" Cataloging CHIANTI wavelength grid",
                unit=" species",
                ncols=100,
                leave=True,
                disable=not self.verbose
        ):

            if 'line' not in attributes['keys']:
                continue

            chianti_nebula_object = nebula_chianti.chianti(
                chianti_ion=species,
                temperature=dummy_temperature,
                ne=dummy_ne,
                verbose=False
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
                    energy_keV = 12.398419843320026 / wvl
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

        utils.nebula_done_comment(
            f"Saved CHIANTI line catalogue to {outfile}"
        )

    ######################################################################################
    # COMPUTE SPECIES SPECTRA RATE
    ######################################################################################
    def computeSpeciesSpectraCoeff(self, row_temperature, row_ne, row_grid_mask):

        # Determine the number of temperature values
        N_temp = len(row_temperature)

        '''
        species_spectra_coefficients = {}
        '''

        # Store spectra by emission process
        species_nonFBCoeff = {}
        species_FBCoeff = {}

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
            #    #utils.nebula_warning(f"{count} Skipping {species} ...")
            #    continue

            species_processes = self.chianti_species_attributes[species]['keys']
            CHIANTI = chianti(
                chianti_ion=species,
                temperature=row_temperature,
                ne=row_ne,
                verbose=self.verbose
            )

            # Bremsstrahlung/free-free emission
            if self.bremsstrahlung and 'ff' in species_processes:
                bremsstrahlung_coefficients = CHIANTI.get_bremsstrahlung_coefficients(
                    wavelength=self.WavelengthGrid
                )

            # Free-bound emission
            if self.freebound and 'fb' in species_processes:
                freebound_coefficients = CHIANTI.get_freebound_coefficients(
                    wavelength=self.WavelengthGrid
                )

            # Line emission
            if self.line and 'line' in species_processes:
                line_coefficients = CHIANTI.get_line_coefficients(
                    wavelength=self.WavelengthGrid
                )
                # dividing by ne to obtain the same value returned by CHIANTI
                line_coefficients = line_coefficients / row_ne[:, None]

            # Two-photon emission
            if self.twophoton and 'line' in species_processes:

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
            nonFBCoeff = (
                    bremsstrahlung_coefficients
                    + line_coefficients
                    + twophoton_coefficients
            )

            species_nonFBCoeff[species] = nonFBCoeff * row_grid_mask[:, None]

            # Free-bound emission only
            species_FBCoeff[species] = freebound_coefficients * row_grid_mask[:, None]

        '''
        return species_spectra_coefficients
        '''
        return species_nonFBCoeff, species_FBCoeff


    ######################################################################################
    # generate spectrum for 2D data
    ######################################################################################
    def generateSpectrum(
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
            utils.nebula_exit_with_error(
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
            utils.nebula_exit_with_error(
                "Input arrays have inconsistent shapes."
            )

        species_densities = {
            species: np.asarray(density, dtype=np.float64)
            for species, density in species_densities.items()
        }

        for species, density in species_densities.items():

            if density.shape != temperature.shape:
                utils.nebula_exit_with_error(
                    f"Species density input arrays have inconsistent "
                    f"shapes for species {species}."
                )

        #todo: CIE can be input to the generateSpectrum Method rather than Spectrum class itself.
        '''
        if self.NEQ:
            utils.nebula_warning(
                "NEI mode is not implemented yet; using CIE instead."
            )

        elif self.CIE:
            utils.nebula_warning(
                "Using collisional ionization equilibrium (CIE)."
            )
        '''

        ##########################################################################
        # Setting Up wavelength grid
        self.setup_wavelength_grid(self.min_wvl, self.max_wvl,
                                   user_grid=self.userGrid,
                                   grid_size=self.gridSize)

        ##########################################################################
        # DEM calculation
        # initialize emission measure class
        EM = emissionMeasure(Tmin=100, Tmax=1.0e9, Nbins=300, verbose=self.verbose)
        # generate DEM
        EM.DEM2D(temperature=temperature,
                 ne=ne, speciesDensities=species_densities,
                 volume=grid_volume, gridMask=grid_mask)

        ##########################################################################
        # Allocate only binned spectra
        SpeciesSpectrum = {
            species: np.zeros(
                (EM.Nbins, self.N_wvl),
                dtype=np.float64
            )
            for species in species_densities
        }

        ##########################################################################
        # Multiprocessing row spectra calculation
        N_grid_level = len(temperature)

        # uniform grid or 1 level grid -------------------------------------------
        if N_grid_level == 1:
            utils.nebula_warning("Uniform grid not implemented.")

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
                         ne[level, row], grid_mask[level, row])
                    )
            Ntasks = len(AllTasks)

            proc = min(self.proc, Ntasks)

            print(
                f" [ MULTIPROCESSING ]: Total tasks {Ntasks} | "
                f"{N_grid_level} levels | "
                f"{len(temperature[0])} grid slices/level "
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

            if self.verbose:
                print(" Computing spectrum")

            with tqdm(
                    total=Ntasks,
                    desc=f" Computing species spectra",
                    unit="task",
                    ncols=90,
                    disable=self.verbose
            ) as pbar:

                while completed < Ntasks:

                    '''
                    level, row, row_species_spectra_coefficents = doneQ.get()
                    '''
                    level, row, row_species_nonFBCoeff, row_species_FBCoeff = doneQ.get()


                    if level == "ERROR":
                        for p in processes:
                            p.terminate()

                        '''
                        utils.nebula_exit_with_error(row_species_spectra_coefficents)
                        '''
                        utils.nebula_exit_with_error(row_species_nonFBCoeff)

                    row_bins = EM.DEM_indices[level, row]

                    for chianti_species, spectra in row_species_nonFBCoeff.items():
                        pion_species = utils.getPionSymbol(chianti_species)
                        if pion_species not in SpeciesSpectrum:
                            continue

                        for bin_idx in range(EM.Nbins):
                            mask = row_bins == bin_idx
                            if not np.any(mask):
                                continue

                            SpeciesSpectrum[pion_species][bin_idx, :] += np.sum(
                                spectra[mask, :],
                                axis=0
                            )

                    completed += 1
                    pbar.update(1)

            for p in processes:
                p.join()

        ##########################################################################
        # Multiply by DEM and sum over temperature bins

        Spectrum = {}

        for species, binned_spectra in SpeciesSpectrum.items():
            weighted_binned_spectra = (
                    binned_spectra * EM.DEM[species][:, None]
            )

            Spectrum[species] = np.sum(
                weighted_binned_spectra,
                axis=0
            )

        integrated_spectrum = np.sum(
            list(Spectrum.values()),
            axis=0
        )

        self.Spectrum = integrated_spectrum


