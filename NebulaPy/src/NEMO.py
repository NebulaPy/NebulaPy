"""NEMO chemistry, chemical tracers, and plasma number densities for PION data."""

import re
import numpy as np
from pypion.SiloHeader_data import OpenData
from NebulaPy.src.Utils import get_element_symbol
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import NebulaError, get_logger
from NebulaPy.src.Progress import track

logger = get_logger(__name__)


class NEMO:
    """Read NEMO chemistry using an existing PION data reader.

    Geometry and grid selection belong to the supplied ``pion`` object.
    Chemistry reads use its current selection without copying geometry state.

    Example
    -------
    >>> simulation = nebula.pion(batched_silos)
    >>> simulation.load_geometry(scale='au')
    >>> simulation.restrict_grid_levels(min_level=2)
    >>> nemo = NEMO(simulation)
    >>> nemo.load_chemistry()
    >>> ne = nemo.get_ne(batched_silos[0])
    """

    def __init__(self, pion):
        self.pion = pion
        self.chemistry_container = {}

    # ==================================================================================#
    # ******************************* LOAD CHEMISTRY ***********************************#
    # ==================================================================================#
    def load_chemistry(self):
        '''
        This method extracts information related to the chemistry and chemical tracers,
        transforming the chemical tracer names to a format that PyPion can directly
        read from the Silo file. This method can be included in the next version of
        PyPion.

        Parameters
        ----------
        instant_silo_set : The instance for which chemical data is to be extracted

        Returns
        -------
        Generates and stores the following in self.chemistry_container:
        - 'dynamics': Dynamics data retrieved from the Silo file
        - 'chemistry': Chemistry flag indicating if chemistry data is available
        - 'E_update': Energy update information (if chemistry flag is true)
        - 'chemistry_code': The code indicating the type of chemistry (if chemistry flag is true)
        - 'microphysics': List of microphysics processes (if chemistry flag is true)
        - 'Ntracers': Number of chemical tracers
        - 'mpv10_elements': List of elements identified for MPv10 chemistry code
        - 'mpv10_tracers': List of tracers corresponding to each element for MPv10 chemistry code
        '''

        # Open the data for the first silo instant silo
        header_data = OpenData(self.pion.silo_set[0])
        # Set the directory to '/header'
        header_data.db.SetDir('/header')
        #print(header_data.header_info())
        # Retrieve the value of "EP_chemistry" from the header data
        chemistry_flag = header_data.db.GetVar("EP_chemistry")
        self.chemistry_container['chemistry'] = chemistry_flag

        # Define the list of process variable names
        processes = ['EP_coll_ionisation', 'EP_rad_recombination',
                     'EP_cooling', 'EP_raytracing', 'EP_phot_ionisation',
                     'EP_charge_exchange']

        # Define the list of process names corresponding to the process variable names
        processes_name = ['coll_ionisation', 'rad_recombination', 'cooling',
                          'raytracing', 'phot_ionisation', 'charge_exchange']

        # Check if chemistry_flag is true
        if chemistry_flag:
            # Retrieve the value of "EP_update_erg"
            energy_update = header_data.db.GetVar("EP_update_erg")
            # save the energy_update value in the chemistry_container dictionary
            self.chemistry_container['E_update'] = energy_update
            # Retrieve the value of "chem_code"
            chemistry_code = header_data.db.GetVar("chem_code")[0]
            # save the chemistry_code value in the chemistry_container dictionary
            self.chemistry_container['chemistry_code'] = chemistry_code

            # Initialize an empty list to store microphysics processes
            microphysics = []
            # Check if the chemistry_code is not 'MPv10'
            if not chemistry_code == 'MPv10':
                # Exit with an error if the chemistry_code is not 'MPv10'
                raise NebulaError(" PION is not running NEMO v1.0; NelubaPy functionality is limited.")
            else:
                logger.info("Loading chemistry: NEMO microphysics features")

                # Loop through each process
                for index, process in enumerate(processes):
                    # Check if the process variable exists in the header data
                    if header_data.db.GetVar(process):
                        # Append the corresponding process name to the microphysics list
                        microphysics.append(processes_name[index])

                # save the microphysics list in the chemistry_container dictionary
                self.chemistry_container['microphysics'] = microphysics
                # Retrieve the number of tracers
                Ntracers = header_data.db.GetVar('num_tracer')
                # elements in the tracer list
                tracer_elements = []
                # pion_chemical_tracers
                pion_chemical_tracers = []
                # mass_fraction
                mass_fractions = {}
                # list of element wise tracer list
                elementWiseTracers = [[] for _ in const.SUPPORTED_ELEMENTS]
                # Element symbols supported by NebulaPy chemistry.
                element_list = list(const.SUPPORTED_ELEMENTS)
                # save the number of tracers in the chemistry_container dictionary
                self.chemistry_container['Ntracers'] = Ntracers


                # Loop through each tracer index
                for i in range(Ntracers):
                    # create a tracer index string with leading zeros
                    tracer_index = f'Tracer{i:03}'
                    # retrieve the tracer value
                    chem_tracer = header_data.db.GetVar(tracer_index)[0]

                    # check if the tracer is an element ('X' denoting elemental mass fraction)
                    if 'X' in chem_tracer and chem_tracer.replace("_", "").replace("X", "") in const.SUPPORTED_ELEMENTS:
                        # extract the element name
                        element = chem_tracer.replace("_", "").replace("X", "")
                        tracer_elements.append(element)
                        mass_fractions[element] = f'Tr{i:03}_' + chem_tracer

                        # get the index of the element in the element_list
                        element_index = element_list.index(element)
                        # append the tracer with the corresponding element to the mpv10tracers list
                        if 0 <= element_index < len(elementWiseTracers):
                            elementWiseTracers[element_index].append(f'Tr{i:03}_' + chem_tracer)

                    # check if the tracer is a corresponding ion
                    if re.sub(r'\d{1,2}\+', '', chem_tracer) in const.SUPPORTED_ELEMENTS:
                        self.chemistry_container[chem_tracer] = f'Tr{i:03}_' + chem_tracer.replace('+', 'p')
                        # extract the element name
                        element = re.sub(r'\d{1,2}\+', '', chem_tracer)
                        # get the index of the element in the element_list
                        element_index = element_list.index(element)
                        # ppend the tracer with the corresponding ion to the mpv10tracers list
                        elementWiseTracers[element_index].append(f'Tr{i:03}_' + chem_tracer.replace('+', 'p'))
                        # append pion chemical tracers
                        pion_chemical_tracers.append(chem_tracer)

                logger.info(
                    "Chemistry: %s elements, %s tracers",
                    len(tracer_elements),
                    Ntracers,
                )
                logger.info("Elements included: %s", ", ".join(tracer_elements))

                # save mass fraction to chemistry_container dictionary
                #self.chemistry_container['mass_fractions'] = mass_fractions
                self.element_list = tracer_elements

                self.chemistry_container['mass_fractions'] = mass_fractions
                self.chemistry_container['tracer_elements'] = tracer_elements
                self.chemistry_container['pion_tracers'] = pion_chemical_tracers
                self.element_wise_tracer_list = elementWiseTracers
        header_data.close()


        nebulapy_all_species = pion_chemical_tracers
        # Upgrade pion_tracers to include top ion
        for element in tracer_elements:
            Z = const.ATOMIC_NUMBER[element]
            top_ion = f"{element}{Z}+"
            if top_ion not in pion_chemical_tracers:
                nebulapy_all_species.append(top_ion)
        # Sort ions by element and ionization stage
        def ion_sort_key(ion):
            import re
            match = re.match(r"([A-Za-z]+)(\d*)\+?$", ion)
            element = match.group(1)
            charge = match.group(2)
            # Neutral species
            if charge == "":
                charge = 0
            else:
                charge = int(charge)
            return (
                tracer_elements.index(element),
                charge
            )
        nebulapy_all_species = sorted(nebulapy_all_species, key=ion_sort_key)
        self.chemistry_container['nebulapy_all_species'] = nebulapy_all_species

    ######################################################################################
    # show all tracer string
    ######################################################################################
    def show_all_tracers(self):

        header_data = OpenData(self.pion.silo_set[0])
        header_data.db.SetDir('/header')

        Ntracers = header_data.db.GetVar('num_tracer')
        tracers = {}

        for i in range(Ntracers):
            tracer_index = f'Tracer{i:03}'
            tracer = header_data.db.GetVar(tracer_index)[0]

            tracers[tracer] = f'Tr{i:03}_{tracer}'

        header_data.close()

        tracer_items = [
            f"{k}: {v}"
            for k, v in tracers.items()
        ]

        ncols = 3
        n = len(tracer_items)
        nrows = (n + ncols - 1) // ncols

        col_width = max(len(item) for item in tracer_items) + 2
        box_width = ncols * (col_width + 1) + 1

        logger.info("PION tracer information: %s tracers", Ntracers)
        logger.debug("PION tracers: %s", ", ".join(tracer_items))

    ######################################################################################
    # get elements
    ######################################################################################
    def get_elements(self):
        return np.array(self.chemistry_container['tracer_elements'])

    ######################################################################################
    # get chemical tracers
    ######################################################################################
    def get_chemical_tracer_list(self):
        """
        Retrieve the list of chemical tracer strings for each tracer in the chemistry
        container dictionary, processed element by element. Each sublist starts with the
        mass fraction of the element followed by the tracers.

        Returns:
            list of lists: Each sublist contains the mass fraction followed by the values of
            the tracers for a specific element.
        """
        elements = self.get_elements()
        tracers = []

        for element in elements:
            # Retrieve tracers for the element
            element_tracers = [self.chemistry_container[f"{element}{q}+" if q > 0 else element]
                               for q in range(const.ATOMIC_NUMBER[element])]

            tracers.append(element_tracers)

        return tracers

    ######################################################################################
    # get elemental mass fraction
    ######################################################################################
    def get_elemental_mass_frac(self, silo_instant):

        elements = self.get_elements()
        elemental_mass_fraction = []
        for element in elements:
            # Retrieve mass fraction
            element_tracer = self.chemistry_container['mass_fractions'][element]
            elemental_mass_fraction.append(self.pion.get_parameter(element_tracer, silo_instant))

        return np.array(elemental_mass_fraction)

    ######################################################################################
    # get elemental mass fraction
    ######################################################################################
    def get_ion_tracer(self, ion):
        """
        Retrieves the tracer for a specific ion from the chemistry container.

        Parameters:
        ion (str): The key corresponding to the specific ion (e.g., 'H+', 'He++').

        Returns:
        tracer: The value associated with the provided ion key in the chemistry container.

        Raises:
        KeyError: If the specified ion key does not exist in the chemistry container.
        """
        if ion not in self.chemistry_container:
            raise NebulaError(f" ion {ion} not found in the chemistry container")
        return self.chemistry_container[ion]


    ######################################################################################
    # check if the ion is top pion
    ######################################################################################
    def top_ion_check(self, ion):
        """
        Check if a given ion qualifies as a top-level ion based on predefined criteria.

        Args:
            ion (str): The ion to check (e.g., 'H+', 'C++').

        Returns:
            bool: True if the ion is a top-level ion, False otherwise.
        """

        # Extract the element symbol from the ion string using the utility function.
        # For example, 'C++' would return 'C'.
        element = get_element_symbol(ion)

        # Check if the ion meets the criteria for being a top-level ion:
        # 1. It is listed in the predefined set of top-level ions (const.FULLY_IONIZED_IONS).
        # 2. Its associated element is a recognized tracer element in the chemistry model.
        if ion in const.FULLY_IONIZED_IONS and element in self.chemistry_container['tracer_elements']:
            logger.debug(
                "Ion '%s' is a top-level ion and not an explicit NEMO species",
                ion,
            )
            return True  # The ion qualifies as a top-level ion.
        else:
            # If the ion does not meet the criteria, return False.
            return False

    ######################################################################################
    # check if the ion exist in pion simulation file
    ######################################################################################
    def ion_batch_check(self, ion=None, ion_list=None, top_ion_check=False, terminate=False):
        """
        This method checks if the given ion(s) are valid according to the chemistry model and optional top-level ion conditions.
        It allows for checking a single ion or a list of ions, and can either raise an exception or print warnings when ions are invalid.

        Parameters:
        ion (str): The ion to check (e.g., 'O+2', 'H+1'). If provided, only this single ion will be checked.
        ion_list (list): A list of ions to check (e.g., ['O+2', 'H+1']). If provided, each ion in the list will be checked.
        top_ion_check (bool): If True, checks if the ion is a top-level ion using the top_ion_check method. Defaults to False.
        terminate (bool): If True and the ion is not found, an exception will be raised. If False, a warning is logged instead. Defaults to False.

        Returns:
        list: A list of valid ions that passed the check.
        """
        # Printing separator for clarity

        # Checking if both ion and ion_list are provided, which is an error
        if ion is not None and ion_list is not None:
            raise NebulaError("Ion batch check - provide either 'ion' or 'ionlist', but not both")

        # If neither ion nor ion_list is provided, exit with an error
        if ion is None and ion_list is None:
            raise NebulaError("Provide either 'ion' or 'ionlist' for ion batch check")

        filtered_ion_list = []  # List to hold ions that pass the check

        # Case 1: Single ion check
        if ion is not None:
            found_ion = False
            # If top_ion_check is enabled, check if the ion is a top-level ion
            if top_ion_check:
                # Check if the ion is a top-level ion
                if self.top_ion_check(ion):
                    filtered_ion_list.append(ion)
                    found_ion = True
                # If the ion is found in the chemistry container, add it to the filtered list
                elif ion in self.chemistry_container:
                    filtered_ion_list.append(ion)
                    found_ion = True
                # If ion is not in the container and terminate is False, log a warning
                elif ion not in self.chemistry_container and terminate is False:
                    logger.warning(f"Ion '{ion}' not recognized")
                # If ion is not in the container and terminate is True, exit with an error
                elif ion not in self.chemistry_container and terminate:
                    raise NebulaError(f"Ion '{ion}' not recognized")

            # If top_ion_check is False, just check if the ion exists in the chemistry container
            elif ion in self.chemistry_container:
                filtered_ion_list.append(ion)
                found_ion = True
            # If the ion is not found, log a warning or exit based on terminate flag
            else:
                if terminate:
                    raise NebulaError(f"Ion '{ion}' not recognized")
                elif not terminate:
                    logger.warning(f"Ion '{ion}' not recognized")

            if found_ion:
                logger.debug("Ion check: %s found in chemistry container", ion)

            return filtered_ion_list

        # Case 2: List of ions check
        elif ion_list is not None:
            for ion in ion_list:
                found_ion = False
                # If top_ion_check is enabled, check if the ion is a top-level ion
                if top_ion_check:
                    # Check if the ion is a top-level ion
                    if self.top_ion_check(ion):
                        filtered_ion_list.append(ion)
                        found_ion = True
                    # If the ion is found in the chemistry container, add it to the filtered list
                    elif ion in self.chemistry_container:
                        filtered_ion_list.append(ion)
                        found_ion = True
                    # If ion is not in the container and terminate is False, log a warning
                    elif ion not in self.chemistry_container and terminate is False:
                        logger.warning(f"ion '{ion}' not recognized")
                    # If ion is not in the container and terminate is True, exit with an error
                    elif ion not in self.chemistry_container and terminate:
                        raise NebulaError(f"ion '{ion}' not recognized")

                # If top_ion_check is False, just check if the ion exists in the chemistry container
                elif ion in self.chemistry_container:
                    filtered_ion_list.append(ion)
                    found_ion = True
                # If the ion is not found, log a warning or exit based on terminate flag
                else:
                    if terminate:
                        raise NebulaError(f"Ion '{ion}' not recognized")
                    elif not terminate:
                        logger.warning(f"Ion '{ion}' not recognized")

                if found_ion:
                    logger.debug("Ion check: %s found in chemistry container", ion)

            return filtered_ion_list

    ######################################################################################
    # get tracer values
    ######################################################################################
    def get_chemical_tracers(self, silo_instant):
        """
        Retrieves the chemical tracer values for the given time instant from the
        simulation silo data.
        Parameters:
        ----------
        silo_instant : silo file(s)

        Returns:
        -------
        tracer_values : list of lists
            A 2D list containing the tracer values for each ion in the tracers array.
        """

        logger.debug("Chemistry container: %s", self.chemistry_container)

        # Retrieve the 2D array of chemical tracers.
        tracers = self.get_chemical_tracer_list()

        # Initialize tracer_values using list comprehension for better efficiency.
        tracer_values = np.array([
            [self.pion.get_parameter(ion, silo_instant) for ion in element_row]
            for element_row in tracers
        ], dtype=object)

        return tracer_values





    ######################################################################################
    # get ion mass fraction values
    ######################################################################################
    def get_ion_values(self, ion, silo_instant):
        '''
        This methods will return the ion mass fraction value set

        Parameters
        ----------
        ion name
        silo_instant

        Returns
        -------
        ion mass fraction
        '''

        element = get_element_symbol(ion)
        ion_tracer = None
        if ion not in self.chemistry_container:
            if ion in const.FULLY_IONIZED_IONS and element in self.chemistry_container['mass_fractions']:
                logger.debug(
                    "Ion '%s' is a top-level ion, not an explicit NEMO species",
                    ion,
                )
                return None
            else:
                raise NebulaError(f"ion {ion} is not in silo file")
        else:
            ion_tracer = self.chemistry_container[ion]

        return self.pion.get_parameter(ion_tracer, silo_instant)

    ######################################################################################
    # get electron number density
    ######################################################################################
    def get_ne(self, silo_instant, progress=None):
        """
        Return electron number density for a specific silo file.
        """

        # 1D spherical grid
        if self.pion.geometry_container['coordinate_sys'] == 'spherical':

            density = self.pion.get_parameter("Density", silo_instant)
            ne = np.zeros(len(density))

            for e, element in enumerate(self.element_wise_tracer_list):

                if not element:
                    continue

                massfrac_sum = np.zeros(len(density))
                element_name = self.element_list[e]
                atomic_number = len(element) - 1
                top_ion = self.pion.get_parameter(element[0], silo_instant)

                for i, ion in enumerate(element[1:], start=1):
                    charge = i - 1
                    ion_density = self.pion.get_parameter(ion, silo_instant)

                    top_ion -= ion_density
                    massfrac_sum += charge * ion_density

                massfrac_sum += atomic_number * np.maximum(top_ion, 0.0)
                ne += massfrac_sum / const.ATOMIC_MASS[element_name]

            ne *= density
            below_floor = ne < const.ELECTRON_DENSITY_FLOOR
            corrected_cells = np.count_nonzero(below_floor)

            if corrected_cells:
                logger.warning(
                    "Electron density is below %.3e cm^-3 in %s cells; "
                    "applying the numerical floor",
                    const.ELECTRON_DENSITY_FLOOR,
                    corrected_cells,
                )
                ne[below_floor] = const.ELECTRON_DENSITY_FLOOR

            return ne

        # 2D cylindrical grid
        if self.pion.geometry_container['coordinate_sys'] == 'cylindrical':

            Nlevel = self.pion.geometry_container['Nlevel']

            if Nlevel == 1:
                grid = "uniform grid"
            else:
                grid = "grid level"

            density = self.pion.get_parameter("Density", silo_instant)

            shape_list = [arr.shape for arr in density]

            ne = [np.zeros(shape) for shape in shape_list]

            show_progress = self.pion.progress if progress is None else progress
            level_iterator = track(
                range(Nlevel),
                total=Nlevel,
                description="Calculating electron density",
                unit="grid levels",
                enabled=show_progress,
            )

            for level in level_iterator:

                for e, element in enumerate(self.element_wise_tracer_list):

                    if not element:
                        continue

                    element_name = self.element_list[e]
                    atomic_number = len(element) - 1

                    massfrac_sum = np.zeros(shape_list[level])

                    top_ion = self.pion.get_parameter(element[0], silo_instant)

                    for i, ion in enumerate(element[1:], start=1):
                        charge = i - 1
                        ion_density = self.pion.get_parameter(ion, silo_instant)

                        top_ion[level] -= ion_density[level]
                        massfrac_sum += charge * ion_density[level]

                    massfrac_sum += atomic_number * np.maximum(top_ion[level], 0.0)

                    ne[level] += massfrac_sum / const.ATOMIC_MASS[element_name]

            ne = [density[level] * ne[level] for level in range(Nlevel)]
            below_floor = [
                level_ne < const.ELECTRON_DENSITY_FLOOR
                for level_ne in ne
            ]
            corrected_cells = sum(
                np.count_nonzero(level_mask)
                for level_mask in below_floor
            )

            if corrected_cells:
                logger.warning(
                    "Electron density is below %.3e cm^-3 in %s cells; "
                    "applying the numerical floor",
                    const.ELECTRON_DENSITY_FLOOR,
                    corrected_cells,
                )

                for level, level_mask in enumerate(below_floor):
                    ne[level][level_mask] = const.ELECTRON_DENSITY_FLOOR

            return ne

    ######################################################################################
    # get top ion mass fraction
    ######################################################################################
    def get_top_ion_massfrac(self, ion, silo_instant):

        # Extract the element string from ion string
        element = get_element_symbol(ion)
        atomic_number = const.ATOMIC_NUMBER[element]
        element_tracer = self.chemistry_container['mass_fractions'][element]
        # set elemental mass fraction to top level ion mass fraction
        top_ion_mass_frac = self.pion.get_parameter(element_tracer, silo_instant)

        if self.pion.geometry_container['coordinate_sys'] == 'cylindrical':
            # Get the number of nested grid levels in the geometry container.
            Nlevel = self.pion.geometry_container['Nlevel']

            for charge in range(atomic_number):
                if charge == 0:
                    ion = f"{element}"
                    ion_value = self.get_ion_values(ion, silo_instant)
                    top_ion_mass_frac = [top_ion_mass_frac[level] - ion_value[level] for level in range(Nlevel)]

                else:
                    ion = f"{element}{charge}+"  # Adding + for positive ions
                    ion_value = self.get_ion_values(ion, silo_instant)
                    top_ion_mass_frac = [top_ion_mass_frac[level] - ion_value[level] for level in range(Nlevel)]

        top_ion_mass_frac = [np.maximum(top_ion_mass_frac[level], 0.0) for level in range(Nlevel)]

        return top_ion_mass_frac


    ######################################################################################
    # get get ion number density
    ######################################################################################
    def get_ion_number_density(self, ion, silo_instant):
        """
        Calculates the number density of a given ion across different nested grid levels.

        This method is currently implemented for a cylindrical 2D coordinate system.
        A 1D spherically symmetric case will require separate handling (TODO).

        Parameters:
        ion (str): The identifier for the ion (e.g., 'H+', 'O++').
        silo_instant: The current simulation time or instant for which the calculation is performed.

        Returns:
        list: A list of arrays containing the ion number density for each nested grid level.
        """

        # Retrieve the density parameter at the given simulation instant
        density = self.pion.get_parameter('Density', silo_instant)

        # Identify the shape of each density array for consistency
        shape_list = [arr.shape for arr in density]

        # Initialize arrays to store the ion number density
        ion_num_density = [np.zeros(shape) for shape in shape_list]

        # Extract the element symbol from the ion identifier
        element = get_element_symbol(ion)

        # Get the mass of the element from constants
        element_mass = const.ATOMIC_MASS[element]

        if self.pion.geometry_container['coordinate_sys'] == 'spherical':
            # TODO: Implement the calculation for 1D spherically symmetric coordinate system
            raise NotImplementedError(
                "Ion number density calculation for spherically symmetric 1D geometry is not yet implemented.")

        # Currently implemented for cylindrical 2D coordinate system
        elif self.pion.geometry_container['coordinate_sys'] == 'cylindrical':
            # Get the number of nested grid levels in the geometry container
            Nlevel = self.pion.geometry_container['Nlevel']

            # Check if the ion is a top-level ion (no sub-ion values available)
            if self.get_ion_values(ion, silo_instant) is None:
                logger.debug("Computing number density for top-level ion %s", ion)
                # Retrieve the mass fraction for the top-level ion
                ion_mass_frac = self.get_top_ion_massfrac(ion, silo_instant)

                # Calculate the ion number density for each grid level
                for level in range(Nlevel):
                    ion_num_density[level] = density[level] * ion_mass_frac[level] / element_mass
            else:
                # Retrieve the mass fraction for sub-level ions
                ion_mass_frac = self.get_ion_values(ion, silo_instant)

                # Calculate the ion number density for each grid level
                for level in range(Nlevel):
                    ion_num_density[level] = density[level] * ion_mass_frac[level] / element_mass

        elif self.pion.geometry_container['coordinate_sys'] == 'cartesian':
            # TODO: Implement the calculation for 3D cartesian coordinate system
            raise NotImplementedError(
                "Ion number density calculation for 3D cartesian geometry is not yet implemented.")

        return ion_num_density

    ######################################################################################
    # GET ALL ION NUMBER DENSITIES
    ######################################################################################
    def get_species_number_densities(self, silo_instant, ion_list=None):
        """Return number-density grids for all or selected ion species.

        Parameters
        ----------
        silo_instant
            Simulation files belonging to one time instant.
        ion_list : sequence of str, optional
            PION ion symbols to process. If omitted, process every species in
            the loaded chemistry container.
        """
        species_number_densities = {}

        species_list = (
            self.chemistry_container['nebulapy_all_species']
            if ion_list is None
            else list(ion_list)
        )

        for ion in track(
                species_list,
                description="Calculating number densities",
                unit="species",
                enabled=self.pion.progress,
        ):
            species_number_densities[ion] = self.get_ion_number_density(
                ion=ion,
                silo_instant=silo_instant,
            )

        return species_number_densities


    ######################################################################################
    # get get total number density ion number density
    ######################################################################################
    def get_ntot(self, silo_instant):

        # 1 dimensional (spherical)
        if self.pion.geometry_container['coordinate_sys'] == 'spherical':
            raise NebulaError('not implemented for 1D coordinate')

        # 2 dimensional (cylindrical)
        if self.pion.geometry_container['coordinate_sys'] == 'cylindrical':

            # Get the number of nested grid levels in the geometry container.
            Nlevel = self.pion.geometry_container['Nlevel']

            logger.info("Calculating total number density for each grid level")

            # Retrieve the density data from the input file at the current simulation instant.
            density = self.pion.get_parameter("Density", silo_instant)

            # Identify the shape of each density array to ensure compatibility with other parameters.
            shape_list = [arr.shape for arr in density]

            # Initialize arrays for electron number density (ne) and mass fraction sum,
            # with zeroes matching the shape of the density data.
            ne = [np.zeros(shape) for shape in shape_list]
            massfrac_sum = [np.zeros(shape) for shape in shape_list]
            neutral_massfrac = [np.zeros(shape) for shape in shape_list]
            n_total = [np.zeros(shape) for shape in shape_list]

            # Loop through each element in the tracer list to calculate contributions from individual ions.
            for e, element in enumerate(self.element_wise_tracer_list):

                # If the current element has no associated tracers, skip to the next element.
                if not element:
                    continue

                # Get the element name and compute its atomic number (total ions minus one).
                element_name = self.element_list[e]
                atomic_number = len(element) - 1

                # Get neutral elemental mass fraction
                neutral_massfrac = self.pion.get_parameter(element[0], silo_instant)
                # Add neutral elemental mass fraction into top ion mass fraction first
                top_ion_massfrac = self.pion.get_parameter(element[0], silo_instant)
                # Add neutral mass fraction into total number density
                for level in range(Nlevel):
                    n_total[level] += neutral_massfrac[level] / const.ATOMIC_MASS[element_name]

                # For each subsequent ion
                for i, ion in enumerate(element[1:], start=1):
                    charge = i - 1  # Charge is one less than ionization state index.
                    ion_massfrac = self.pion.get_parameter(ion, silo_instant)

                    # Update the top ion and mass fraction sum for each grid level.
                    for level in range(Nlevel):
                        top_ion_massfrac[level] -= ion_massfrac[level]
                        massfrac_sum[level] += charge * ion_massfrac[level]  # Update mass fraction sum by ion charge.
                        n_total[level] += ion_massfrac[level] / const.ATOMIC_MASS[element_name]

                # Finalize mass fraction sum and update electron density for each grid level.
                for level in range(Nlevel):
                    # Add contribution of the top ion with atomic number, ensuring non-negative values.
                    massfrac_sum[level] += atomic_number * np.maximum(top_ion_massfrac[level], 0.0)
                    # Calculate electron number density using mass fraction sum and atomic mass.
                    ne[level] += massfrac_sum[level] / const.ATOMIC_MASS[element_name]
                    n_total[level] += top_ion_massfrac[level] / const.ATOMIC_MASS[element_name]

            # Scale the electron number density by the density for each grid level.
            ne = [density[level] * ne[level] for level in range(Nlevel)]

            n_total = [density[level] * n_total[level] for level in range(Nlevel)]
            for level in range(Nlevel):
                n_total[level] += ne[level]

            logger.info("Total number-density calculation completed")
            return n_total
