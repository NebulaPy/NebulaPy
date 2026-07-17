import os
os.environ['XUVTOP']
import io as _io
import warnings
from contextlib import redirect_stdout

# ChiantiPy prints version and GUI-selection messages unconditionally during
# import and calls os.path.isfile(False) when no chiantirc exists. Keep those
# known import-time side effects quiet, particularly in spawned workers.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="bool is used as a file descriptor",
        category=RuntimeWarning,
    )
    with redirect_stdout(_io.StringIO()):
        import ChiantiPy
        import ChiantiPy.core as ch
        import ChiantiPy.tools.filters as chfilters
        import ChiantiPy.tools.io as chio
        import ChiantiPy.tools.data as chdata
        from ChiantiPy.base import specTrails
import numpy as np
import ChiantiPy.tools.util as chianti_util
from NebulaPy.src import Constants as const
from ChiantiPy.core.Continuum import continuum
import ChiantiPy.tools.io as io
from scipy.interpolate import splev, splrep
from NebulaPy.src.LoggingConfig import NebulaError, get_logger

logger = get_logger(__name__)

# Limit the number of threads used by OpenMP and Intel MKL to 1.
# This prevents oversubscription of CPU cores when running multiprocessing tasks
# and ensures efficient resource utilization in parallel workflows.
os.environ["OMP_NUM_THREADS"] = "1"  # Restrict OpenMP to 1 thread
os.environ["MKL_NUM_THREADS"] = "1"  # Restrict Intel MKL to 1 thread


class chianti:

    ######################################################################################
    # class initialization
    ######################################################################################
    def __init__(self, temperature, ne,
                 chianti_ion=None, pion_ion=None, pion_elements=None,
                 continuum=False,
                 verbose=False):

        self.temperature = temperature
        self.ne = ne
        self.verbose = verbose

        # Count the number of arguments that are not None
        non_none_count = sum(arg is not None for arg in [chianti_ion, pion_ion, pion_elements])

        # If more than one argument is not None, raise a ValueError
        if non_none_count > 1:
            raise NebulaError("invalid arguments: set only one of 'chianti_ion', 'pion_ion', or 'pion_elements")

        if pion_ion is not None:
            self.chianti_ion_name = self.get_chianti_symbol(pion_ion, make=False)
            self.chianti_ion = ch.ion(self.chianti_ion_name, temperature=self.temperature, eDensity=self.ne,
                                 pDensity='default', radTemperature=None, rStar=None, abundance=None,
                                 setup=True, em=None, verbose=self.verbose)
            self.get_ion_attributes()

        if chianti_ion is not None:
            self.chianti_ion = ch.ion(chianti_ion, temperature=self.temperature, eDensity=self.ne,
                                 pDensity='default', radTemperature=None, rStar=None, abundance=None,
                                 setup=True, em=None, verbose=self.verbose)
            self.chianti_ion_name = chianti_ion

        if pion_elements is not None:
            chianti_element_list = []
            for element in pion_elements:
                element_symbol = self.get_chianti_symbol(element, make=True)
                chianti_element_list.append(element_symbol)
            self.chianti_element_list = chianti_element_list
            self.get_elements_attributes()

    ######################################################################################
    # terminate any class objects
    ######################################################################################
    def terminate(self):
        """
        Close the Chianti object and release any resources.
        """
        if hasattr(self, 'chianti_ion'):
            del self.chianti_ion
        if hasattr(self, 'species_attributes_container'):
            del self.species_attributes_container

    ######################################################################################
    # generate species chianti symbol
    ######################################################################################
    def get_chianti_symbol(self, species, make=False):
        '''
        Converts a PION species symbol to a corresponding CHIANTI species symbol.

        This function takes a species symbol used in PION (a computational tool) and
        converts it into the format required by CHIANTI, a database for atomic data.
        The function can generate either the elemental symbol or the ionized symbol
        depending on the 'make' parameter.

        :param species: str, the PION species symbol, which can represent both neutral
                        and ionized states.
        :param make: bool, if True, returns the elemental symbol (e.g., 'h' for hydrogen).
                     if False, returns the CHIANTI ion symbol (e.g., 'h_2' for H+).
        :return: str, the corresponding CHIANTI symbol for the species.
        '''

        # Convert the input species symbol to lowercase and remove any '+' characters
        # (denoting ionization)
        species = species.lower().replace('+', '')

        # Extract alphabetic characters to identify the element symbol (e.g., 'h' from 'h1' or 'h+')
        element = ''.join(filter(str.isalpha, species))

        if make:
            # If 'make' is True, return only the element symbol (e.g., 'h')
            return element
        else:
            # Extract numeric characters to determine the ionization level (e.g., '1' from 'h1')
            ion_level = ''.join(filter(str.isdigit, species))

            # If no numeric characters are found, set the CHIANTI ionization level to 1
            chianti_level = int(ion_level) + 1 if ion_level else 1

            # Return the element symbol followed by the ionization level, separated
            # by an underscore (e.g., 'h_2')
            return f"{element}_{chianti_level}"

    ######################################################################################
    # get attributes of all elements in chianti element list
    ######################################################################################
    def get_elements_attributes(self):
        '''
        Generates and appends species attributes for each element in the
        `chianti_element_list` to a dictionary called `species_attributes`.

        This function first retrieves abundance data from the `chdata.Abundance`
        dictionary using the specified abundancoe name ('unity'). It then initializes
        an instance of the `specTrails` class to handle species-related data, setting
        its abundance and temperature properties.

        The function calls the `ionGate` method on the `species` object to process
        the elements in `chianti_element_list`, generating species data based on
        the specified parameters.

        The species data is then looped over, and for each element key in the sorted
        `species.Todo` dictionary:
        - The key is converted using CHIANTI's ``convertName`` helper.
        - The relevant data is stored in the `species_attributes` dictionary under
          each element's key.
        - Unnecessary entries like 'filename' and 'experimental' are removed
          from the dictionary before final storage.

        :return: None
        '''
        AbundanceName = 'unity'
        abundAll = chdata.Abundance[AbundanceName]['abundance']

        species = specTrails()  # Create an instance of specTrails to manage species data
        species.AbundAll = abundAll
        species.Temperature = self.temperature  # Set the temperature for the species

        species.ionGate(
            elementList=self.chianti_element_list,
            minAbund=None, doLines=True,
            doContinuum=True, doWvlTest=0,
            doIoneqTest=0, verbose=False
        )

        self.species_attributes_container = {}

        for akey in sorted(species.Todo.keys()):

            self.species_attributes_container[akey] = chianti_util.convertName(akey)

            # Store relevant data
            self.species_attributes_container[akey]['keys'] = species.Todo[akey]

            # Remove unnecessary entries safely
            self.species_attributes_container[akey].pop('filename', None)
            self.species_attributes_container[akey].pop('experimental', None)

        if self.verbose:
            names = [
                attributes["spectroscopic"]
                for attributes in self.species_attributes_container.values()
            ]
            logger.info("Retrieved CHIANTI attributes for %s species", len(names))
            logger.debug("CHIANTI species: %s", ", ".join(names))

        # Finalize the species attributes dictionary
        # At this point, `self.species_attributes` contains all the relevant
        # attributes for the species in `chianti_element_list`

    ######################################################################################
    # get ion attributes
    ######################################################################################
    def get_ion_attributes(self):

        AbundanceName = 'unity'
        abundAll = chdata.Abundance[AbundanceName]['abundance']

        species = specTrails()  # Create an instance of specTrails to manage species data
        species.AbundAll = abundAll
        species.Temperature = self.temperature  # Set the temperature for the species

        ion_list = [self.chianti_ion_name]
        species.ionGate(ionList=ion_list,
            minAbund=None, doLines=True,
            doContinuum=True, doWvlTest=0,
            doIoneqTest=0, verbose=False)

        self.species_attributes_container = {}

        # Loop through the sorted keys in the dictionary of species
        if self.verbose:
            logger.info("Retrieving CHIANTI ion attributes for %s", self.chianti_ion_name)

        for akey in sorted(species.Todo.keys()):
            self.species_attributes_container[akey] = chianti_util.convertName(akey)  # Convert the key and store it
            self.species_attributes_container[akey]['keys'] = species.Todo[akey]  # Store relevant data
            # Remove unnecessary data from the dictionary
            del self.species_attributes_container[akey]['filename']
            del self.species_attributes_container[akey]['experimental']

    ######################################################################################
    # get all lines of the ion
    ######################################################################################
    def get_allLines(self):
        """
                   Retrieve all spectral lines associated with a specified ion
                   :return: wave-length array
                   """
        if self.verbose:
            logger.info("Retrieving all spectral lines for %s", self.chianti_ion.Spectroscopic)
        wvl = np.asarray(self.chianti_ion.Wgfa['wvl'], np.float64)
        wvl = np.abs(wvl)
        return wvl

    ######################################################################################
    # get all lines and transitions of the ion
    ######################################################################################
    def get_allLineTransitions(self):
        """
        Retrieve all spectral lines associated with a specified ion
        :return: wave-length array
        """
        if self.verbose:
            logger.info("Retrieving spectral lines and transitions for %s", self.chianti_ion.Spectroscopic)

        Ref = self.chianti_ion.Elvlc['ref']
        A_value = np.asarray(self.chianti_ion.Wgfa['avalue'], np.float64)
        Pretty1 = self.chianti_ion.Wgfa['pretty1']
        Pretty2 = self.chianti_ion.Wgfa['pretty2']
        wvl = np.asarray(self.chianti_ion.Wgfa['wvl'], np.float64)

        wvl = np.abs(wvl)
        A_value = np.abs(A_value)
        return {'Reference': Ref, 'wvl': wvl, 'Avalue': A_value, 'Lower': Pretty1, 'Upper': Pretty2}

    ######################################################################################
    # Get line emissivity, this is an internal method
    ######################################################################################
    def get_line_emissivity(self, allLines=True):
        """
               Retrieve the emissivity values for all spectral lines associated
               with a specified ion.

               :return: Dict Emiss. Emiss has several quantities, namely, ion,
               # wvl(angstrom), emissivity (ergs s^-1 str^-1), pretty1, pretty2.
               """
        if 'line' not in self.species_attributes_container[self.chianti_ion_name]['keys']:
            logger.warning(f'no line emission associate with {self.chianti_ion.Spectroscopic}')
            return None

        else:
            if self.verbose:
                logger.info("Retrieving all line emissivities for %s", self.chianti_ion.Spectroscopic)
            self.chianti_ion.emiss(allLines=allLines)
            emissivity = self.chianti_ion.Emiss
            return emissivity



    ######################################################################################
    # get line emissivity for a list of lines, this is an internal method
    ######################################################################################
    def get_line_emissivity_for_list(self, line_list):
        """
        Retrieves the emissivity values for a given list of spectral lines.

        Parameters:
        ----------
        line_list : list
            A list of spectral line identifiers for which emissivity values need to be retrieved.

        Returns:
        -------
        line_emissivity : dict
            A dictionary where keys are formatted line identifiers (e.g., "Fe XIV 530.3")
            and values are the corresponding emissivity values.
        """

        # Retrieve the full list of available spectral lines from the current object.
        all_lines = np.array(self.get_allLines())  # Get all available lines as a NumPy array.
        all_lines = all_lines[all_lines != 0]  # Remove any zero entries (which may indicate missing or invalid lines).

        # Print a message if verbose mode is enabled.
        if self.verbose:
            logger.info("Retrieving indices for requested spectral lines")

        # Check if every requested line exists in the available list of lines.
        missing_lines = [line for line in line_list if line not in all_lines]

        # If there are missing lines, provide suggestions
        if missing_lines:
            suggestion_lines = [" Suggestions for missing lines:"]
            for line in missing_lines:
                try:
                    closest = min(all_lines, key=lambda x: abs(float(x) - float(line)))
                    suggestion_lines.append(
                        f" {float(line):8.3f} Å — do you mean {float(closest):9.3f} Å?"
                    )
                except Exception:
                    suggestion_lines.append(f" {float(line):8.3f} Å — no close match found")

            suggestion_str = "\n".join(suggestion_lines)

            raise NebulaError(
                f"Following {self.chianti_ion.Spectroscopic} line(s) were not found in Chianti: {missing_lines}\n"
                f" Note: Chianti spectral line data are based on theoretical models\n"
                f"{suggestion_str}"
            )


        # If any requested lines are missing, terminate execution with an error message.
        if missing_lines:
            raise NebulaError(f"following {self.chianti_ion.Spectroscopic} line(s) are not found in chianti: {missing_lines}")

        # Retrieve the indices of the requested lines within the all_lines array.
        # np.where(all_lines == line) returns an array of indices where the condition is met.
        # We take the first occurrence with [0][0] since it's assumed that each line appears only once.
        line_indices = [np.where(all_lines == line)[0][0] for line in line_list if line in all_lines]

        # Retrieve the full emissivity array from the method get_line_emissivity.
        # The `allLines=False` argument ensures that emissivity is returned only for relevant lines.
        emissivity = self.get_line_emissivity(allLines=False)['emiss']

        # Initialize a dictionary to store emissivity values for the requested lines.
        line_emissivity = {}

        # Iterate through the requested lines and their corresponding indices.
        for i, index in enumerate(line_indices):
            # The spectroscopic notation (e.g., "Fe XIV") is combined with the wavelength (or another identifier).
            line_str = self.chianti_ion.Spectroscopic + " " + str(line_list[i])

            # Retrieve the emissivity value corresponding to the current line.
            specific_line_emissivity = emissivity[index]

            # Store the retrieved emissivity value in the dictionary.
            line_emissivity[line_str] = specific_line_emissivity

        # Return the dictionary containing emissivity values for the requested lines.
        return line_emissivity



    ######################################################################################
    # get bremsstrahlung emission rate info: verified OK
    ######################################################################################
    def get_bremsstrahlung_coefficients(self, wavelength):
        """
        Calculates the free-free emission (bremsstrahlung) rate for a single ion.

        Returns
        -------
        numpy.ndarray
            2D array with shape (n_temperature, n_wavelength), where each row
            contains the bremsstrahlung emission spectrum for one temperature.
        """

        # -----------------------------
        # Prepare temperature and wavelength
        # -----------------------------

        temperature = np.asarray(self.temperature, dtype=np.float64).reshape(-1)
        wavelength = np.asarray(wavelength, dtype=np.float64).reshape(-1)

        nT = temperature.size
        nW = wavelength.size

        # -----------------------------
        # Ion charge
        # -----------------------------
        Zion = self.chianti_ion.Ion - 1

        # -----------------------------
        # Continuum object
        # -----------------------------
        continuum_gaunt = continuum(
            self.chianti_ion_name,
            temperature=temperature,
            abundance=None,
            em=None,
            verbose=False
        )

        if self.verbose:
            logger.debug(
                "Computing bremsstrahlung emission rate for %s",
                self.chianti_ion.Spectroscopic,
            )


        prefactor_const = ((const.light * 1e8) / 3. / const.emass
                     * (const.fine * const.planck / np.pi) ** 3
                     * np.sqrt(2. * const.pi / 3. / const.emass / const.kB))

        # shape -> (nT, 1)
        prefactor = (prefactor_const * Zion ** 2 / np.sqrt(temperature))[:, np.newaxis]

        # -----------------------------
        # Exponential factor
        # shape -> (nT, nW)
        # -----------------------------
        exponent = -const.planck * (1.0e8 * const.light) / (
                const.boltzmann * np.outer(temperature, wavelength)
        )
        exp_factor = np.exp(exponent) / (wavelength[np.newaxis, :] ** 2)

        # -----------------------------
        # Gaunt factors
        # -----------------------------
        gf_itoh = np.asarray(continuum_gaunt.itoh_gaunt_factor(wavelength))
        gf_sutherland = np.asarray(continuum_gaunt.sutherland_gaunt_factor(wavelength))

        # Replace NaNs in Itoh with Sutherland
        gf = np.where(np.isnan(gf_itoh), gf_sutherland, gf_itoh)

        # -----------------------------
        # Force Gaunt factor to shape (nT, nW)
        # -----------------------------

        # -----------------------------
        # Final emission
        # shape -> (nT, nW)
        # -----------------------------
        bremsstrahlung_coefficients = prefactor * exp_factor * gf

        # Final safety check
        bremsstrahlung_coefficients = np.asarray(bremsstrahlung_coefficients, dtype=np.float64).squeeze()

        if self.verbose:
            logger.debug(
                "Bremsstrahlung emission computed for %s",
                self.chianti_ion.Spectroscopic,
            )

        return bremsstrahlung_coefficients

    ######################################################################################
    # get free-bound emission coefficients
    ######################################################################################
    def get_freebound_coefficients(self, wavelength, verner=True):
        """
        Calculates the free-bound (radiative recombination) continuum emissivity of an ion.
        """

        if self.verbose:
            logger.debug(
                "Computing free-bound emission rate for %s",
                self.chianti_ion.Spectroscopic,
            )

        # Create a continuum object
        continuum_fb = continuum(
            self.chianti_ion_name,
            temperature=self.temperature,
            abundance=None,
            em=None,
            verbose=self.verbose
        )
        N_temp = len(self.temperature)
        N_wvl = len(wavelength)
        # Note: em=None is used to avoid scaling the emissivity by emission measure.
        continuum_fb.freeBound(wavelength, includeAbund=False, includeIoneq=False, verner=verner)
        # Note: By setting includeAbund=False and includeIoneq=False, we ensure that
        # the returned emissivity is purely for the ion of interest, without scaling
        # by abundance or ionization fraction.
        if 'intensity' in continuum_fb.FreeBound:
            freebound_coefficients = continuum_fb.FreeBound['intensity']
        else:
            freebound_coefficients = np.zeros((N_temp, N_wvl), dtype=np.float64)
        # Clean up the continuum object to free memory
        del continuum_fb
        if self.verbose:
            logger.debug(
                "Free-bound emission computed for %s",
                self.chianti_ion.Spectroscopic,
            )

        return freebound_coefficients


    ######################################################################################
    # get line emission rate for all lines within a wavelength range, with optional filter
    ######################################################################################
    def get_line_coefficients(self, wavelength, allLines=True, filtername=None, filterfactor=None):

        if self.verbose:
            logger.debug(
                "Retrieving all spectral-line emissivities for %s",
                self.chianti_ion.Spectroscopic,
            )

        wavelength = np.asarray(wavelength, dtype=np.float64)

        self.chianti_ion.emiss(allLines=allLines)

        emissivity = np.asarray(self.chianti_ion.Emiss['emiss'], dtype=np.float64)
        lines = np.asarray(self.chianti_ion.Emiss['wvl'], dtype=np.float64)

        min_wvl = wavelength.min()
        max_wvl = wavelength.max()

        useFilter = chfilters.gaussianR
        useFactor = 1000.0

        selected_idx = np.where(
            (lines >= min_wvl) & (lines <= max_wvl)
        )[0]

        N_temp = len(self.temperature)
        N_wvl = len(wavelength)

        line_emission_coefficients = np.zeros((N_temp, N_wvl), dtype=np.float64)

        if len(selected_idx) == 0:
            if self.verbose:
                logger.info(
                    "No %s lines found between %.2e and %.2e Å; skipping",
                    self.chianti_ion.Spectroscopic,
                    min_wvl,
                    max_wvl,
                )
            return line_emission_coefficients

        selected_lines = lines[selected_idx]
        selected_emissivity = emissivity[selected_idx, :]

        for temp_idx in range(N_temp):
            for line_idx, line_wvl in enumerate(selected_lines):
                line_profile = useFilter(
                    wavelength,
                    line_wvl,
                    factor=useFactor
                )

                line_emission_coefficients[temp_idx, :] += (
                        line_profile * selected_emissivity[line_idx, temp_idx]
                )

        if self.verbose:
            logger.debug(
                "Spectral-line calculation completed for %s",
                self.chianti_ion.Spectroscopic,
            )

        return line_emission_coefficients

    ######################################################################################
    # get two photon emission coefficients
    ######################################################################################
    def get_twophoton_coefficients(self, wavelength):

        '''
        Returns the emissivity of two-photon emission for the specified wavelength(s).
        The return value is divided by the electron number denisty.
        :param wavelength:
        :return:
        '''

        #  upper level of two photon transitions for helium-like ions
        heseqLvl2 = [-1, 3, -1, -1, -1, 3, 6, 6, -1, 6, 6, 6, 6, 6, 3,
                     5, 3, 5, 3, 5, -1, 5, -1, 5, -1, 5, -1, 5, -1, 5]

        N_temp = len(self.temperature)
        N_wvl = len(wavelength)

        twophoton_coefficients = np.zeros((N_temp, N_wvl), dtype=np.float64)

        # for hydrogen sequence
        if self.chianti_ion.Z == self.chianti_ion.Ion:
            l1 = 1 - 1
            l2 = 2 - 1
            wvl0 = 1.e+8 / (self.chianti_ion.Elvlc['ecm'][l2] - self.chianti_ion.Elvlc['ecm'][l1])
            has_valid_wavelength = np.any(wavelength > wvl0)
        # for Helium sequence
        else:
            l1 = 1 - 1
            l2 = heseqLvl2[self.chianti_ion.Z - 1] - 1
            wvl0 = 1.e+8 / (self.chianti_ion.Elvlc['ecm'][l2] - self.chianti_ion.Elvlc['ecm'][l1])
            has_valid_wavelength = np.any(wavelength > wvl0)

        if has_valid_wavelength:
            if self.verbose:
                logger.debug(
                    "Computing two-photon emission rate for %s",
                    self.chianti_ion.Spectroscopic,
                )
            self.chianti_ion.twoPhotonEmiss(wavelength)
            twophoton_coefficients = self.chianti_ion.TwoPhotonEmiss['emiss']

            if self.verbose:
                logger.debug(
                    "Two-photon emission computed for %s",
                    self.chianti_ion.Spectroscopic,
                )

        if not has_valid_wavelength:
            if self.verbose:
                logger.info(
                    f"Skipping two-photon emission for {self.chianti_ion.Spectroscopic}: "
                    f"no wavelengths > {wvl0:.3f} Å.")

        return twophoton_coefficients
