
import pyneb as pn
from pyneb import RecAtom
from pyneb import Atom
from pyneb import atomicData as pyneb_atomic_data
from NebulaPy.src import Constants as const
from NebulaPy.src import chianti as nebula_chianti
import ChiantiPy.core as ch
#import pkg_resources
import warnings
from .Chianti import chianti
import numpy as np
from NebulaPy.src.LoggingConfig import NebulaError, get_logger

logger = get_logger(__name__)

pn.log_.level = 0

'''
This class is soley written for obtaining recombination line emissivities using PyNeb package
'''

class pyneb:

    ######################################################################################
    # class initialization
    ######################################################################################
    def __init__(self, temperature, ne, pion_ion=None):

        self.pion_ion = pion_ion
        self.temperature = temperature
        self.ne = ne

        # Count the number of arguments that are not None
        non_none_count = sum(arg is not None for arg in [pion_ion])

        # If more than one argument is not None, raise a ValueError
        if non_none_count > 1:
            raise NebulaError("invalid arguments: set 'pion_ion' for PyNeb class")

        if pion_ion is not None:
            if pion_ion in const.FULLY_IONIZED_IONS:
                logger.warning("top-level ions are not PION species, processing as PyNeb species")
            self.pyneb_ion_element, self.pyneb_ion_spectral_level = self.get_pyneb_symbol(pion_ion)
            self.pyneb_ion_name = self.pyneb_ion_element + str(self.pyneb_ion_spectral_level)
            self.Spectroscopic = self.spectroscopic_form(pion_ion)

            all_recombination_ions = pyneb_atomic_data.getAllAtoms(coll=False, rec=True)
            if self.pyneb_ion_name in all_recombination_ions:
                logger.debug("PyNeb recombination data found for %s", self.Spectroscopic)

                # begin todo: when implemting for other species, modify below
                if self.pyneb_ion_element != 'H':
                    raise NebulaError(f"PyNeb is implemented only for H I recombination lines, not {self.Spectroscopic}")
                else:
                    logger.warning("PyNeb is currently implemented only for H I recombination lines")
                    # important: only implemented for recombination lines of hydrogen
                    pn.atomicData.setDataFile('h_i_rec_SH95.hdf5')
                    self.pyneb_recomb_ion = RecAtom(self.pyneb_ion_element, self.pyneb_ion_spectral_level)
                    self.pyneb_ion = Atom(self.pyneb_ion_element, self.pyneb_ion_spectral_level)
                # end todo

            else:
                raise NebulaError(f"no recombination data for '{self.Spectroscopic}' in PyNeb")


    ######################################################################################
    # generate species pyneb symbol
    ######################################################################################
    def get_pyneb_symbol(self, species):
        """
        Parse a species string (e.g., 'H1', 'He2+', 'o3') into PyNeb convention.

        Args:
            species (str): Input atomic/ionic species.

        Returns:
            tuple: (Element, Spectral Level, Ionization Level)
                - Element: str, e.g. 'O'
                - Spectral Level: int, PyNeb stage (e.g. 3 for 'O3')
                - Ionization Level: int, electrons removed (Spectral Level - 1)
        """

        # Normalize input
        species = species.lower().replace('+', '')

        # Extract element (letters only)
        element = ''.join(filter(str.isalpha, species))

        # Extract spectral level (digits only), default 1 if missing
        digits = ''.join(filter(str.isdigit, species))
        ionisation_level = int(digits) if digits else 0
        spectral_level = ionisation_level + 1  # PyNeb stage

        return element.capitalize(), spectral_level

    ######################################################################################
    # Get spectroscopic form of species
    ######################################################################################
    def spectroscopic_form(self, species: str) -> str:
        """
        Convert species like 'H1+', 'He1+', 'C2+' into spectroscopic notation.
        'H1+' -> 'H II', 'C2+' -> 'C III', 'H' -> 'H I'
        """
        import re

        match = re.match(r"([A-Za-z]+)(\d*)\+?", species)
        if not match:
            raise ValueError(f"Invalid species format: {species}")

        element, ion_stage = match.groups()
        element = element.capitalize()
        ion_stage = int(ion_stage) if ion_stage else 0  # default to 0 if missing

        roman_stage = ion_stage + 1
        if roman_stage < 1 or roman_stage > len(const.ROMAN_ION_STAGES):
            raise ValueError(f"Ionization stage out of range: {roman_stage}")

        return f"{element} {const.ROMAN_ION_STAGES[roman_stage - 1]}"

    ######################################################################################
    # Get all recombination lines available in PyNeb for given ion
    ######################################################################################
    def get_allLines(self):
        """
        Retrieve all available H I recombination line wavelengths from PyNeb.

        This method calculates the emissivities for a dummy electron temperature and
        density to access all transitions available in PyNeb's recombination data.
        It then extracts the wavelengths for transitions up to a maximum principal
        quantum number (n_max = 40) and filters out invalid entries (wavelength = 0).

        Steps:
        1. Use a dummy temperature (1e4 K) and electron density (1e3 cm^-3)
           to generate emissivities for all H I transitions.
        2. Loop over all transitions, splitting the labels into upper and lower levels.
        3. Skip transitions above the maximum level (n_max = 40).
        4. Retrieve the wavelength for each valid transition using PyNeb.
        5. Filter out any zero wavelengths.
        6. Return a sorted list of all valid wavelengths.

        Returns:
            list of float: Sorted wavelengths of all available H I recombination lines.
        """
        logger.debug("Retrieving all PyNeb recombination lines for %s", self.Spectroscopic)

        all_lines = []
        dummy_temperature = 1e4  # K
        dummy_ne = 1e3  # cm^-3

        # Get labelled emissivities for the dummy conditions
        labelled_emissivity = self.pyneb_recomb_ion.getEmissivity(
            tem=dummy_temperature,
            den=dummy_ne,
            product=False
        )

        n_max = 40  # Maximum principal quantum number to consider

        # Loop over each transition and get the corresponding wavelength
        for transition in labelled_emissivity.keys():
            upper, lower = map(int, transition.split('_'))
            if upper > n_max or lower > n_max:
                continue

            wavelength = self.pyneb_recomb_ion.getWave(lev_i=upper, lev_j=lower)
            if wavelength != 0:  # Skip invalid entries
                all_lines.append(wavelength)

        all_lines = sorted(all_lines)

        return np.array(all_lines)

    ######################################################################################
    # Get recombination line emissivity for list of lines
    ######################################################################################
    def get_recomb_line_emissivity_for_list(self, line_list):
        """
        Retrieve recombination line emissivities for a given list of spectral lines.

        Parameters
        ----------
        line_list : list of float
            Wavelengths of the spectral lines for which emissivities are requested.

        Returns
        -------
        dict
            Dictionary where keys are "SpectroscopicName Wavelength" strings and values
            are the corresponding emissivity values (erg s^-1 sr^-1).
        """

        logger.debug(
            "Retrieving PyNeb recombination-line emissivities for %s",
            self.Spectroscopic,
        )

        # Retrieve all available recombination lines for the current ion
        all_lines = self.get_allLines()

        # Set a wavelength tolerance for matching lines (approximately 3 decimal places)
        tolerance = 1e-3

        # Identify which requested lines are missing from the available data
        missing_lines = [
            line for line in line_list
            if not any(abs(line - l) < tolerance for l in all_lines)
        ]

        # If there are missing lines, provide suggestions for possible close matches
        if missing_lines:
            suggestion_lines = [" suggestions for missing lines:"]
            for line in missing_lines:
                try:
                    # Find the closest available line to the requested one
                    closest = min(all_lines, key=lambda x: abs(float(x) - float(line)))
                    suggestion_lines.append(
                        f" {float(line):.3f} Å — do you mean {float(closest):.3f} Å?"
                    )
                except Exception:
                    suggestion_lines.append(f" {float(line):.3f} Å — no close match found")

            # Combine suggestions into a single string for error message
            suggestion_str = "\n".join(suggestion_lines)

            # Raise an error with detailed information about missing lines
            raise NebulaError(
                f"following {self.Spectroscopic} line(s) were not found in PyNeb: {missing_lines}\n"
                f" note: PyNeb spectral line data are sourced from the NIST database\n"
                f"{suggestion_str}"
            )

        # Initialize a dictionary to store emissivity values for each requested line
        line_emissivity = {}

        # Loop through each requested line to calculate its emissivity
        for line in line_list:
            line_str = f"{self.Spectroscopic} {line}"

            # Retrieve emissivity for the specific recombination line
            specific_line_emissivity = self.pyneb_recomb_ion.getEmissivity(
                tem=self.temperature, den=self.ne, wave=line, product=False
            )

            # excluding emissivity value for temperature out of range for recombination caseB coeffcicient for hydrogen
            minimum_temperature, maximum_temperature = (
                const.PYNEB_H_RECOMBINATION_TEMPERATURE_RANGE
            )
            invalid_temperature = (
                (self.temperature < minimum_temperature)
                | (self.temperature > maximum_temperature)
            )
            # Now invalid_parameter_range is True where either temperature or ne is out of range
            specific_line_emissivity[invalid_temperature] = 1.0e-50

            # Detect NaN values and display an error message
            if np.isnan(specific_line_emissivity).any():
                raise NebulaError(
                    f"pyneb: found NaN emissivity for given T and ne in line {line_str}"
                )

            # Convert to emissivity per steradian
            line_emissivity[line_str] = specific_line_emissivity * self.ne / (4.0 * const.PI)

        # Return the dictionary containing emissivities for all requested lines
        return line_emissivity
