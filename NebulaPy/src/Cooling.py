import numpy as np
import os
from NebulaPy.src import Constants as const
from scipy import interpolate
from NebulaPy.src.LoggingConfig import NebulaError, get_logger

logger = get_logger(__name__)

class cooling():

    ######################################################################################
    # initializing the class cooling
    ######################################################################################
    def __init__(self, pion_ion, verbose=True):
        """
        Initializes the cooling class for a specific pion ion

        Args:
            database (str): Path to the database directory containing the required data files.
            pion_ion (str): Ion symbol used to fetch the corresponding cooling data.
            verbose (bool): If True, prints detailed logging information. Defaults to True.

        Raises:
            FileNotFoundError: If the required database or cooling table is missing.
        """
        self.verbose = verbose
        self.ion = pion_ion

        # get database
        database = os.environ.get("NEBULAPYDB")
        # Check if the database exists, exit if missing
        if database is None:
            raise NebulaError("required database missing, install database to proceed")

        # Get the corresponding CHIANTI ion symbol for the given pion_ion
        chinati_ion = self.get_chianti_symbol(pion_ion)

        if self.verbose:
            logger.info("Initializing cooling calculation for %s", pion_ion)
        # Construct the filename for the ion cooling table based on the ion symbol
        ion_cooling_filename = chinati_ion + '.txt'
        cooling_database = os.path.join(database, "Cooling", "Chianti")

        # Full path to the cooling table
        self.ion_cooling_file = os.path.join(cooling_database, ion_cooling_filename)

        if self.verbose:
            logger.info("Retrieving %s cooling-rate data", pion_ion)

        # Check if the cooling file exists, exit if not found
        if not os.path.exists(self.ion_cooling_file):
            raise NebulaError(f"database does not contain a cooling table for {pion_ion}")
        else:
            # Set up the cooling table if the file exists
            self.setup_cooling_table()

    ######################################################################################
    # setup cooling table
    ######################################################################################
    def setup_cooling_table(self):
        """
        Sets up the cooling table for the ion by loading the cooling data and preparing the
        interpolation function for cooling rates based on electron density and temperature.

        The cooling rates are interpolated from a pre-existing table of temperature and
        electron density values, and stored in the self.cooling_rate function for use in
        further calculations.

        This method processes the cooling data file by removing the temperature data and
        performing an interpolation to get a smooth cooling rate function.
        """

        if self.verbose:
            logger.info("Computing cooling-rate interpolation for %s", self.ion)

        # Prepare a list of temperatures (in Kelvin) from log(10) scale, ranging from 10^1 to 10^8
        nemo_temperature = []
        for i in range(81):
            log_temp = 1.0 + i * 0.1
            nemo_temperature.append(pow(10, log_temp))

        # Prepare a list of electron densities (in cm^-3) from log(10) scale, ranging from 10^0 to 10^6
        nemo_ne = []
        for i in range(13):
            log_ne = i * 0.5
            nemo_ne.append(pow(10, log_ne))

        # Load the cooling data from the file
        ion_cooling_log_data = np.loadtxt(self.ion_cooling_file)

        # Remove the temperature column from each row (keeping only cooling data)
        ion_cooling_log_data = [inner[1:] for inner in ion_cooling_log_data]

        # Convert the cooling data to linear scale (from log(10) scale)
        ion_cooling_data = np.power(10, ion_cooling_log_data)

        # Create an interpolation function for cooling rate as a function of electron density and temperature
        self.cooling_rate = interpolate.interp2d(nemo_ne, nemo_temperature, ion_cooling_data, kind='linear')

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
    # generate cooling map
    ######################################################################################
    def generate_cooling_rate_map(self, temperature, ne):
        """
        Generates a cooling rate map for a given temperature and electron density array.

        This method computes the cooling rate at each point in the input arrays for temperature
        and electron density by using the pre-defined interpolation function `self.cooling_rate`.
        It ensures that the input arrays for temperature and electron density are of the same shape
        and then generates a corresponding map of cooling rates for each combination of temperature
        and electron density.

        Args:
            temperature (array-like): A 2D or nD array of temperature values.
            ne (array-like): A 2D or nD array of electron density values corresponding to the temperature array.

        Returns:
            np.ndarray: A 2D or nD array of cooling rates corresponding to each temperature and electron density value.

        Raises:
            ValueError: If the shapes of the temperature and electron density arrays do not match.
        """

        # Ensure input arrays are NumPy arrays
        temperature = np.array(temperature)
        ne = np.array(ne)

        # Check if the shapes of the temperature and ne arrays match
        if temperature.shape != ne.shape:
            raise NebulaError(f"incompatible shapes between the temperature array and the ne array.")

        # Initialize a map to store cooling rates
        cooling_rate_map = np.zeros(temperature.shape)

        # Iterate over each element in the temperature array (using the shape of the array)
        for indices in np.ndindex(temperature.shape):
            # Retrieve the temperature and electron density values at the current index
            ne_value = ne[indices]
            temp_value = temperature[indices]

            # Interpolate the cooling rate for the current temperature and electron density
            cooling_rate_map[indices] = self.cooling_rate(ne_value, temp_value)

        # Return the map of cooling rates
        cooling_rate_map[cooling_rate_map < 1e-50] = 1e-50
        return cooling_rate_map
