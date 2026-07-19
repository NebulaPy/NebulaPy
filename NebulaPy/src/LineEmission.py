from .Chianti import chianti
import numpy as np
import copy
from .PyNeb import pyneb
import multiprocessing
from multiprocessing import Pool, Manager
from typing import Optional
from NebulaPy.src.Utils import get_spectroscopic_symbol
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import NebulaError, get_logger
from NebulaPy.src.Progress import update_progress

logger = get_logger(__name__)

class line_emission():

    ######################################################################################
    # initializing the class emissionline
    ######################################################################################
    def __init__(self, ion):
        """
        only single ion is considered here
        """
        self.ion = ion
        self.line_emission_container = {}
        self.line_emission_container['ion'] = self.ion

    ######################################################################################
    # check the line list exist in all lines of the species
    ######################################################################################
    def chianti_line_batch_check(self, lines):

        # Retrieve the list of possible emission lines for the species
        dummy_temperature_array = [1000]
        dummy_ne_array = [1.0]
        ion = chianti(pion_ion=self.ion, temperature=dummy_temperature_array, ne=dummy_ne_array)
        spectroscopic_name = ion.chianti_ion.Spectroscopic
        all_lines = ion.get_line_emissivity(allLines=True)['wvl']
        del ion

        missing_line = [line for line in lines if line not in all_lines]
        if missing_line:
            missing_line_str = ", ".join(map(str, missing_line))
            raise NebulaError(f"Requested line(s) {spectroscopic_name} {missing_line_str} not "
                                        f"found in the CHIANTI database")
        else:
            logger.info("Requested %s lines found in the CHIANTI database", spectroscopic_name)


    ######################################################################################
    # check the line list exist in pyneb database
    ######################################################################################
    def pyneb_line_batch_check(self, lines):
        logger.warning("Line check not implemented for PyNeb")

    ######################################################################################
    # line luminosity in 1D spherical setting
    ######################################################################################
    def line_luminosity_1D(self, lines, temperature, ne, species_density, shell_volume):
        '''

        Parameters
        ----------
        lines
        temperature
        ne
        ns
        dV

        Returns
        -------

        '''

        ion = chianti(pion_ion=self.ion, temperature=temperature, ne=ne)
        self.line_emission_container['temperature'] = temperature
        self.line_emission_container['ne'] = ne
        self.line_emission_container['spectroscopic'] = ion.chianti_ion.Spectroscopic

        all_emissivity_data = ion.get_line_emissivity()
        allLines = all_emissivity_data['wvl']
        self.line_emission_container['allLines'] = allLines
        self.line_emission_container['lines'] = lines

        indices = []
        for line in lines:
            logger.debug(
                "Identifying %.6g Å in %s line data",
                line,
                ion.chianti_ion.Spectroscopic,
            )
            index = (np.abs(allLines - line)).argmin()
            tolerance = 10 ** -4
            if np.abs(allLines[index] - line) <= tolerance:
                logger.debug("Line %.6g Å found at index %s", line, index)
                indices.append(index)
            else:
                raise NebulaError('line not found in allLines')

        self.line_emission_container['line_indices'] = indices

        logger.debug(
            "Retrieving %s emissivities for lines %s",
            ion.chianti_ion.Spectroscopic,
            lines,
        )

        emissivity = np.asarray(all_emissivity_data['emiss'][indices])
        self.line_emission_container['emiss'] = emissivity

        # Calculating line Luminosity
        logger.info(
            "Calculating %s luminosities for lines %s",
            ion.chianti_ion.Spectroscopic,
            lines,
        )
        luminosity = [4.0 * const.PI * np.sum(e * species_density * shell_volume) for e in emissivity]
        self.line_emission_container['luminosity'] = luminosity

    ######################################################################################
    # line emissivity map for a given list of lines in cylindrical coordinate system
    ######################################################################################
    def line_emissivity_2D_map(self, lines, temperature, ne, progress_bar=True):
        # Computes the emissivity maps of spectral lines for a specified ion in a
        # cylindrical coordinate grid using CHIANTI atomic data.
        #
        # Parameters:
        # - lines: list of strings representing spectral line identifiers (e.g., '5007', '6563').
        # - temperature: 2D list of NumPy arrays for electron temperature, organized by grid level.
        # - ne: 2D list of NumPy arrays for electron number density, organized by grid level.
        # - progress_bar: boolean flag to show progress updates.
        #
        # Returns:
        # - Dictionary where keys are line identifiers and values are 2D emissivity maps
        #   corresponding to each line.

        # Define a minimum value for electron density to prevent division by zero
        electron_tolerance = const.ELECTRON_DENSITY_FLOOR

        # Get the number of grid levels from the temperature data
        NGlevel = len(temperature)

        # Determine the number of rows per level (assumes uniform shape across levels)
        rows = len(temperature[0])

        # Prepare an array to hold the emissivity maps for each requested line
        lines_emissivity_map = {}

        # Create a "zero" emissivity map template with the same shape as temperature levels
        zero_emissivity_map = [np.zeros(shape) for shape in [arr.shape for arr in temperature]]
        zero_emissivity_map = np.array(zero_emissivity_map)

        # make keys for lines emissivity map
        for line in lines:
            map_key = f"{get_spectroscopic_symbol(self.ion)} {line}"
            lines_emissivity_map[map_key] = copy.deepcopy(zero_emissivity_map)

        # Iterate through each grid level
        for level in range(NGlevel):
            # Ensure electron density is a NumPy array
            ne[level] = np.array(ne[level])

            # Replace zero values in electron density to avoid numerical issues
            ne[level][ne[level] == 0] = electron_tolerance

            # Process each row in the current level
            for row in range(rows):
                # Extract 1D temperature and electron density arrays for this row
                temperature_row = temperature[level][row]
                ne_row = ne[level][row]
                # Create a CHIANTI ion object to calculate emissivities
                ion = chianti(pion_ion=self.ion, temperature=temperature_row, ne=ne_row)
                # Get emissivities for the specified lines as a dictionary {line: values}
                lines_emissivity_row = ion.get_line_emissivity_for_list(line_list=lines)
                del ion  # Free the ion object memory
                # saving to the main lines emissivity map
                for map_key, row_key in zip(lines_emissivity_map.keys(), lines_emissivity_row.keys()):
                    lines_emissivity_map[map_key][level][row] = lines_emissivity_row[row_key]
                del lines_emissivity_row

                # Display progress bar if enabled
                if progress_bar:
                    prefix_msg = f'chianti: computing emissivity of {self.ion} lines at grid-level {level}'
                    update_progress(
                        key=prefix_msg,
                        description=prefix_msg,
                        completed=row + 1,
                        total=rows,
                        unit="rows",
                    )

        # Return a dictionary of emissivity maps, keyed by line identifiers
        return lines_emissivity_map

    ######################################################################################
    # get dominant list of lines for a simulation snapshot in cylindrical coordinate system
    ######################################################################################
    def get_species_dominant_lines(
            self,
            temperature,
            ne,
            species_density,
            cell_volume,
            grid_mask,
            Nlines: Optional[int] = None
    ):
        """
        Computes the most luminous emission lines for a given species in a 1D or 2D dataset.

        This method calculates the line luminosity for a given ionized species across multiple grid levels.
        It retrieves the brightest emission lines by computing the line emissivity for each cell and summing
        the contributions across the grid.

        Parameters
        ----------
        temperature : list (1D or 2D array-like)
            Temperature values of the grid cells.
        ne : list (1D or 2D array-like)
            Electron density values corresponding to each grid cell.
        species_density : list (1D or 2D array-like)
            Density of the given ionized species in each grid cell.
        cell_volume : list (1D or 2D array-like)
            Volume of each grid cell.
        grid_mask : list (1D or 2D array-like)
            Mask specifying active grid cells.
        Nlines : int
            The number of most luminous emission lines to retrieve.

        Returns
        -------
        dict
            A dictionary containing:
            - 'spectroscopic' : str, the spectroscopic name of the species.
            - 'lines' : list, the N most luminous emission line wavelengths.
            - 'luminosity' : list, the luminosity values of the brightest lines.

        Notes
        -----
        - A small shared electron-density floor is used to avoid division by zero.
        - The method loops over each grid level and computes line luminosities.
        - The most luminous lines are selected using `np.argsort()`.
        """

        # Define a tolerance for electron density (to avoid division by zero)
        electron_tolerance = const.ELECTRON_DENSITY_FLOOR

        # Get the number of grid levels in the temperature dataset
        NGlevel = len(temperature)

        # Handle cylindrical coordinates (assumes 2D data)
        rows = len(temperature[0])  # Number of rows in the temperature array

        # Retrieve the list of possible emission lines for the species
        dummy_temperature_array = [1000]
        dummy_ne_array = [1.0]
        species = chianti(pion_ion=self.ion, temperature=dummy_temperature_array, ne=dummy_ne_array)
        spectroscopic_name = species.chianti_ion.Spectroscopic
        keys = species.species_attributes_container[species.chianti_ion_name]['keys']

        # Check if the species has emission lines
        if 'line' not in keys:
            logger.warning(f"{spectroscopic_name} has no line emission associated")
            return {'spectroscopic': spectroscopic_name}
        else:
            all_lines = species.get_allLines()
            # get A value, transition info
            allLineTransitions = species.get_allLineTransitions()
            all_lines = all_lines[all_lines != 0]
        del species

        # Initialize an array to store total line luminosities
        species_all_line_luminosity = np.zeros_like(all_lines)

        # Loop over each grid level
        for level in range(NGlevel):
            #print(f" computing luminosity for all {spectroscopic_name} lines at grid level {level}", end='\r')

            # Ensure electron density values are nonzero
            ne[level] = np.array(ne[level])
            ne[level][ne[level] == 0] = electron_tolerance

            # Initialize luminosity storage for this level
            species_all_lines_luminosity_level = np.zeros_like(all_lines)


            # Loop over each row (assumes a 2D dataset)
            for row in range(rows):
                temperature_row = temperature[level][row]
                ne_row = ne[level][row]
                species_density_row = species_density[level][row]
                cell_volume_row = cell_volume[level][row]
                grid_mask_row = grid_mask[level][row]

                prefix_msg = f'Chianti: computing the luminosity of {spectroscopic_name} lines at grid-level {level}'

                # Compute emissivity for the species at the given conditions
                species = chianti(pion_ion=self.ion, temperature=temperature_row, ne=ne_row)
                all_lines_emissivity_info_row = species.get_line_emissivity(allLines=False)
                del species

                if not np.array_equal(all_lines, np.abs(all_lines_emissivity_info_row['wvl'])):
                    raise NebulaError("internal consistency check failed: line arrays from "
                                                "get_allLines() and get_line_emissivity() differ")

                all_lines_emissivity_row = all_lines_emissivity_info_row['emiss']

                # Compute total luminosity for each emission line
                for index in range(len(all_lines)):
                    species_all_lines_luminosity_level[index] += (
                            4.0 * const.PI * np.sum(all_lines_emissivity_row[index]
                                                    * species_density_row
                                                    * cell_volume_row
                                                    * grid_mask_row)
                    )
                del all_lines_emissivity_info_row

                update_progress(
                    key=prefix_msg,
                    description=prefix_msg,
                    completed=row + 1,
                    total=rows,
                    unit="rows",
                )

            # Accumulate luminosity across all grid levels
            species_all_line_luminosity += species_all_lines_luminosity_level

        # Retrieve the N most luminous lines
        # Handle Nlines option
        if Nlines is None:
            logger.info("Retrieving all luminous %s lines", spectroscopic_name)
            # Sort all lines (descending)
            indices = np.argsort(species_all_line_luminosity)[::-1]
        else:
            if not isinstance(Nlines, int) or Nlines <= 0:
                raise ValueError("Nlines must be a positive integer or None")
            logger.info("Retrieving the %s most luminous %s lines", Nlines, spectroscopic_name)

            # Top N lines (sorted)
            indices = np.argsort(species_all_line_luminosity)[-Nlines:][::-1]

        brightest_lines_luminosity = species_all_line_luminosity[indices]
        brightest_lines = all_lines[indices]

        # Sort A value and transition into
        Avalue = np.zeros(len(brightest_lines))
        transition_lower = np.full(len(brightest_lines), "—", dtype=object)
        transition_upper = np.full(len(brightest_lines), "—", dtype=object)
        if 'line' in keys:
            # Build lookup (fast + preserves mapping)
            lookup = {w: i for i, w in enumerate(allLineTransitions['wvl'])}
            for i, wvl in enumerate(brightest_lines):
                idx = lookup.get(wvl, None)
                if idx is not None:
                    Avalue[i] = allLineTransitions['Avalue'][idx]
                    transition_lower[i] = allLineTransitions['Lower'][idx]
                    transition_upper[i] = allLineTransitions['Upper'][idx]

        return {
            'spectroscopic': spectroscopic_name,
            'lines': brightest_lines,
            'luminosity': brightest_lines_luminosity,
            'Avalue': Avalue,
            'Lower': transition_lower,
            'Upper': transition_upper
        }

    ######################################################################################
    # line luminosity for a given list of lines in 2D coordinate system
    ######################################################################################
    def line_luminosity_2D(self, lines, temperature, ne, species_density, cell_volume, grid_mask, progress_bar=True):
        """
        Compute the total line luminosity for a given ion in a cylindrical coordinate system.

        Parameters:
        - lines: List of emission lines for which luminosity is calculated.
        - temperature: 3D array (NGlevel x rows x columns) containing temperature values at different grid levels.
        - ne: 3D array (NGlevel x rows x columns) containing electron density values at different grid levels.
        - species_density: 3D array containing the density of the species (ion) at different grid points.
        - cell_volume: 3D array containing the volume of each cell in the grid.
        - grid_mask: 3D array (boolean or numeric) acting as a mask to include/exclude specific grid cells.

        Returns:
        - Dictionary mapping emission lines to their computed luminosity values.

        The function follows these steps:
        1. Ensures that electron densities are nonzero to avoid division errors.
        2. Iterates over grid levels and rows to compute line emissivity using Chianti.
        3. Computes and accumulates total line luminosity across the cylindrical grid.
        """

        # Define a small tolerance value to prevent division errors when electron density is zero
        electron_tolerance = const.ELECTRON_DENSITY_FLOOR

        # Get the number of grid levels (assumed to be the first dimension of the temperature array)
        NGlevel = len(temperature)

        # Determine the number of rows in each temperature level (assumes 2D grid structure)
        rows = len(temperature[0])

        # Initialize an array to store the total luminosity of all requested lines
        lines_luminosity = np.zeros_like(lines)

        # Loop through each level in the grid
        for level in range(NGlevel):
            # Convert electron density to an array (if not already) and ensure no zero values
            ne[level] = np.array(ne[level])
            ne[level][ne[level] == 0] = electron_tolerance

            # Initialize an array to store line luminosity for this level
            lines_luminosity_level = np.zeros_like(lines)

            # Iterate over each row in the cylindrical grid
            for row in range(rows):
                # Extract data for the current row at the given level
                temperature_row = temperature[level][row]  # Temperature values for the row
                ne_row = ne[level][row]  # Electron density for the row
                species_density_row = species_density[level][row]  # Species density for the row
                cell_volume_row = cell_volume[level][row]  # Cell volume for the row
                grid_mask_row = grid_mask[level][row]  # Grid mask for the row

                if progress_bar:
                    prefix_msg = f'computing the luminosity of {self.ion} lines at grid-level {level}'

                # Compute the line emissivity for the species using Chianti
                species = chianti(pion_ion=self.ion, temperature=temperature_row, ne=ne_row)
                lines_emissivity_row = species.get_line_emissivity_for_list(
                    lines)  # Retrieve emissivities for specified lines
                line_keys = lines_emissivity_row.keys()  # Get the emission line identifiers
                del species  # Free memory

                # Compute the total luminosity for each emission line in this row
                for index, line in enumerate(line_keys):
                    lines_luminosity_level[index] += (
                            4.0 * const.PI * np.sum(
                        lines_emissivity_row[line] * species_density_row * cell_volume_row * grid_mask_row
                    )
                    )
                del lines_emissivity_row  # Free memory after usage

                if progress_bar:
                    update_progress(
                        key=prefix_msg,
                        description=prefix_msg,
                        completed=row + 1,
                        total=rows,
                        unit="rows",
                    )


            # Sum up the computed luminosities across all levels
            lines_luminosity += lines_luminosity_level

        # Return a dictionary mapping line identifiers to their computed luminosities
        return dict(zip(line_keys, lines_luminosity))


    ######################################################################################
    # recombination line emissivity map for a given list of lines in 2D coordinate system
    ######################################################################################
    def recombination_line_emissivity_2D_map(self, lines, temperature, ne, progress_bar=True):

        # Define a minimum value for electron density to prevent division by zero
        electron_tolerance = const.ELECTRON_DENSITY_FLOOR

        # Get the number of grid levels from the temperature data
        NGlevel = len(temperature)

        # Determine the number of rows per level (assumes uniform shape across levels)
        rows = len(temperature[0])

        # Prepare an array to hold the emissivity maps for each requested line
        lines_emissivity_map = {}

        # Create a "zero" emissivity map template with the same shape as temperature levels
        zero_emissivity_map = [np.zeros(shape) for shape in [arr.shape for arr in temperature]]
        zero_emissivity_map = np.array(zero_emissivity_map)

        # make keys for lines emissivity map
        for line in lines:
            map_key = f"{get_spectroscopic_symbol(self.ion)} {line}"
            lines_emissivity_map[map_key] = copy.deepcopy(zero_emissivity_map)

        # Iterate through each grid level
        for level in range(NGlevel):
            # Ensure electron density is a NumPy array
            ne[level] = np.array(ne[level])

            # Replace zero values in electron density to avoid numerical issues
            ne[level][ne[level] == 0] = electron_tolerance

            # Process each row in the current level
            for row in range(rows):
                # Extract 1D temperature and electron density arrays for this row
                temperature_row = temperature[level][row]
                ne_row = ne[level][row]

                # Create a PyNeb atom object to calculate recombination emissivities
                ion = pyneb(pion_ion=self.ion, temperature=temperature_row, ne=ne_row)
                # Get emissivities for the specified lines as a dictionary {line: values}
                lines_emissivity_row = ion.get_recomb_line_emissivity_for_list(line_list=lines)
                del ion  # Free the ion object memory
                # saving to the main lines emissivity map
                for map_key, row_key in zip(lines_emissivity_map.keys(), lines_emissivity_row.keys()):
                    lines_emissivity_map[map_key][level][row] = lines_emissivity_row[row_key]
                del lines_emissivity_row

                # Display progress bar if enabled
                if progress_bar:
                    prefix_msg = f'pyneb: computing emissivity of {self.ion} lines at grid-level {level}'
                    update_progress(
                        key=prefix_msg,
                        description=prefix_msg,
                        completed=row + 1,
                        total=rows,
                        unit="rows",
                    )

        # Return a dictionary of emissivity maps, keyed by line identifiers
        return lines_emissivity_map


######################################################################################
    # line luminosity for a given list of lines in 2D coordinate system
    ######################################################################################
    def recombination_line_luminosity_2D(self, lines, temperature, ne, species_density, cell_volume,
                                         grid_mask, progress_bar=True):
        """
        Compute the total line luminosity for a given ion in a cylindrical coordinate system.

        Parameters:
        - lines: List of emission lines for which luminosity is calculated.
        - temperature: 3D array (NGlevel x rows x columns) containing temperature values at different grid levels.
        - ne: 3D array (NGlevel x rows x columns) containing electron density values at different grid levels.
        - species_density: 3D array containing the density of the species (ion) at different grid points.
        - cell_volume: 3D array containing the volume of each cell in the grid.
        - grid_mask: 3D array (boolean or numeric) acting as a mask to include/exclude specific grid cells.

        Returns:
        - Dictionary mapping emission lines to their computed luminosity values.

        The function follows these steps:
        1. Ensures that electron densities are nonzero to avoid division errors.
        2. Iterates over grid levels and rows to compute line emissivity using Chianti.
        3. Computes and accumulates total line luminosity across the cylindrical grid.
        """

        # Define a small tolerance value to prevent division errors when electron density is zero
        electron_tolerance = const.ELECTRON_DENSITY_FLOOR

        # Get the number of grid levels (assumed to be the first dimension of the temperature array)
        NGlevel = len(temperature)

        # Determine the number of rows in each temperature level (assumes 2D grid structure)
        rows = len(temperature[0])

        # Initialize an array to store the total luminosity of all requested lines
        lines_luminosity = np.zeros_like(lines)

        # Loop through each level in the grid
        for level in range(NGlevel):
            # Convert electron density to an array (if not already) and ensure no zero values
            ne[level] = np.array(ne[level])
            ne[level][ne[level] == 0] = electron_tolerance

            # Initialize an array to store line luminosity for this level
            lines_luminosity_level = np.zeros_like(lines)

            # Iterate over each row in the cylindrical grid
            for row in range(rows):
                # Extract data for the current row at the given level
                temperature_row = temperature[level][row]  # Temperature values for the row
                ne_row = ne[level][row]  # Electron density for the row
                species_density_row = species_density[level][row]  # Species density for the row
                cell_volume_row = cell_volume[level][row]  # Cell volume for the row
                grid_mask_row = grid_mask[level][row]  # Grid mask for the row

                if progress_bar:
                    prefix_msg = f'computing the luminosity of {self.ion} lines at grid-level {level}'

                # Compute the line emissivity for the species using Pyneb
                species = pyneb(pion_ion=self.ion, temperature=temperature_row, ne=ne_row)
                lines_emissivity_row = species.get_recomb_line_emissivity_for_list(lines)  # Retrieve emissivities for specified lines
                line_keys = lines_emissivity_row.keys()  # Get the emission line identifiers
                del species  # Free memory

                # Compute the total luminosity for each emission line in this row
                for index, line in enumerate(line_keys):
                    lines_luminosity_level[index] += (
                            4.0 * const.PI * np.sum(
                        lines_emissivity_row[line] * species_density_row * cell_volume_row * grid_mask_row
                    )
                    )
                del lines_emissivity_row  # Free memory after usage

                if progress_bar:
                    update_progress(
                        key=prefix_msg,
                        description=prefix_msg,
                        completed=row + 1,
                        total=rows,
                        unit="rows",
                    )


            # Sum up the computed luminosities across all levels
            lines_luminosity += lines_luminosity_level

        # Return a dictionary mapping line identifiers to their computed luminosities
        return dict(zip(line_keys, lines_luminosity))
