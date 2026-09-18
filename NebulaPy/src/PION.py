"""Read PION geometry, simulation times, grid masks, volumes, and fields.

NEMO chemistry and derived number densities are provided by ``NEMO``.
"""

from copy import deepcopy

import numpy as np
from pypion.ReadData import ReadData
from pypion.SiloHeader_data import OpenData
from NebulaPy.src import Constants as const
import astropy.units as unit
from NebulaPy.src.LoggingConfig import NebulaError, get_logger

logger = get_logger(__name__)

class pion():
    '''
    This class is not an alternative to Pypion; rather, it is a
    bundle of methods useful for creating synthetic emission
    maps from the Silo file.
    '''

    def __init__(self, silo_set, progress=True):
        self.silo_set = silo_set
        self.progress = progress
        self._min_grid_level = 0
        self._full_geometry = None
        self._level_radii = None
        self.geometry_container = {}

    ######################################################################################
    # get simulation time
    ######################################################################################
    def get_simulation_time(self, silo_instant, time_unit='sec'):
        """Return the simulation time as an Astropy quantity."""
        from NebulaPy.src.Silo import Silo

        header_data = OpenData(silo_instant)
        try:
            header_data.db.SetDir('/header')
            coord_sys = int(header_data.db.GetVar("coord_sys"))
        finally:
            header_data.close()

        dataio = ReadData(silo_instant)
        try:
            readers = {
                1: dataio.get_3Darray,
                2: dataio.get_2Darray,
                3: dataio.get_1Darray,
            }
            reader = readers.get(coord_sys)
            if reader is None:
                raise NebulaError(
                    f"Unsupported coordinate-system identifier: {coord_sys}"
                )
            simulation_time = reader('Density')['sim_time'] * unit.s
        finally:
            dataio.close()

        selected_unit, _ = Silo._time_unit(time_unit)
        return simulation_time.to(selected_unit)


    ######################################################################################
    # show all parameters in silo file
    ######################################################################################
    def show_all_parameters(self):
        # Open the data for the first silo instant
        header_data = OpenData(self.silo_set[0])
        header_data.db.SetDir('/header')
        header_info = header_data.header_info()


        col_width = 25
        ncols = 5

        logger.info("PION silo header contains %s variables", header_info.nvar)

        # Define categories by prefix or known names
        categories = {
            " Boundary Conditions": lambda v: v.startswith("BC_"),
            " Physics Modules": lambda v: v.startswith("EP_"),
            " Radiative Transfer": lambda v: v.startswith("RT_"),
            " Tracers": lambda v: v.startswith("Tracer"),
            " Stellar Winds": lambda v: v.startswith("WIND_"),
            " Units": lambda v: v.startswith("unit"),
            " Time Control": lambda v: v.startswith("t_") or v in ("last_dt", "min_timestep"),
            " Grid / Solver": lambda v: v in ("CFL", "Gamma", "JetSim", "solver"),
            " Output": lambda v: v in ("outfile", "typeofop", "typeofbc_str", "tracer_str"),
            " Values": lambda v: v.endswith("val"),
        }

        # Group variables
        grouped = {cat: [] for cat in categories}
        for v in header_info.var_names:
            for cat, rule in categories.items():
                if rule(v):
                    grouped[cat].append(v)
                    break
            else:
                grouped.setdefault(" Unsorted", []).append(v)

        # Print in columns
        for cat, items in grouped.items():
            if items:
                logger.info("%s (%s): %s", cat.strip(), len(items), ", ".join(items))
        header_data.close()

    # ==================================================================================
    # LOAD GEOMETRY
    # ==================================================================================
    def load_geometry(self, scale='cm'):
        '''
        This method will load geometry of the simulation from
        the given silo file, preserving any grid-level restriction.

        Parameters
        ----------
        scale : {'cm', 'pc', 'au'}, optional
            Unit for geometry edges: centimetres (default), parsecs, or
            astronomical units. Spherical radii and shell volumes remain in CGS.

        Returns
        -------

        '''
        # Open the data for the first silo instant silo
        header_data = OpenData(self.silo_set[0])
        # Set the directory to '/header'
        header_data.db.SetDir('/header')
        # Retrieve what coordinate system is used
        coord_sys = header_data.db.GetVar("coord_sys")
        header_data.close()
        # Dimension scale
        self.dim_scale = scale

        if coord_sys == 3:
            logger.info("Loading geometry: %s coordinates", const.COORDINATE_SYSTEMS[coord_sys])
            self.spherical_grid(self.silo_set[0])
        elif coord_sys == 2:
            logger.info("Loading geometry: %s coordinates", const.COORDINATE_SYSTEMS[coord_sys])
            self.cylindrical_grid(self.silo_set[0])
        elif coord_sys == 1:
            logger.info("Loading geometry: %s coordinates", const.COORDINATE_SYSTEMS[coord_sys])
            raise NebulaError(f"{const.COORDINATE_SYSTEMS[coord_sys]} coordinates not defined, todo list")


    ######################################################################################
    # Apply grid level restriction
    ######################################################################################
    def restrict_grid_levels(self, min_level=0):
        """Retain an original grid level and every finer level.

        Call after ``load_geometry``. Levels are zero-based, with 0 the
        coarsest. For five levels, ``min_level=2`` retains levels 2, 3, 4.
        Selection persists across geometry reloads and subsequent field/filter
        reads. ``geometry_container['level_indices']`` records original level
        IDs; ``total_levels`` records the unrestricted count, and ``Nlevel``
        records the selected count. Selection uses cached full geometry without
        rereading files. Call with 0 to restore all levels.
        Previously returned arrays are unaffected and must be read again
        after changing the selection.
        """
        if self._full_geometry is None:
            raise NebulaError("Load geometry before restricting grid levels.")
        total = self._full_geometry['total_levels']
        if (isinstance(min_level, (bool, np.bool_))
                or not isinstance(min_level, (int, np.integer))
                or not 0 <= min_level < total):
            raise NebulaError(f"min_level must be an integer between 0 and {total - 1}.")

        min_level = int(min_level)
        selected = deepcopy(self._full_geometry)
        selected['level_indices'] = list(range(min_level, total))
        selected['Nlevel'] = total - min_level
        for key in ('edges_min', 'edges_max', 'mask'):
            selected[key] = selected[key][min_level:]

        if selected['coordinate_sys'] == 'spherical' and min_level:
            mask = selected['mask']
            radii = self._level_radii[min_level:]
            if len(radii) == 1:
                radius = radii[0] * mask[0]
            else:
                radius = np.concatenate([
                    radii[-1],
                    *[radii[i][np.asarray(mask[i]) != 0]
                      for i in range(len(radii) - 2, -1, -1)],
                ])
            selected['radius'] = radius
            selected['shell_volumes'] = np.concatenate([
                [4.0 * const.PI * radius[0] ** 3 / 3.0],
                4.0 * const.PI * np.diff(radius ** 3) / 3.0,
            ])

        self.geometry_container.clear()
        self.geometry_container.update(selected)
        self._min_grid_level = min_level

        logger.info(
            "Restricting to finer grid levels: using %s of %s levels.",
            self.geometry_container['Nlevel'],
            self.geometry_container['total_levels'],
        )

    ######################################################################################
    # Apply grid mask filters
    ######################################################################################
    def apply_grid_filters(
            self,
            silo_instant,
            *,
            temperature=None,
            min_temperature=None,
            max_temperature=None,
    ):
        """Return a fresh snapshot grid mask with optional filters applied.

        Temperature limits are inclusive and may be supplied independently. If
        neither limit is supplied, the original ``NG_Mask`` for the requested
        snapshot is returned for the selected grid levels. The stored geometry
        mask is not modified.

        Parameters
        ----------
        silo_instant : str
            Path to the PION Silo snapshot whose grid mask is required.
        temperature : array-like, optional
            Snapshot temperature grid. Required when either temperature limit is
            supplied and must have the same shape as the selected ``NG_Mask``
            levels (as returned by this method without temperature limits).
        min_temperature, max_temperature : float, optional
            Inclusive lower and upper temperature limits in K.

        Returns
        -------
        numpy.ndarray
            A new grid mask containing the original mask values for selected
            cells and zero for cells rejected by a filter.
        """

        temperature_filter = (
            min_temperature is not None
            or max_temperature is not None
        )

        if temperature_filter and temperature is None:
            raise NebulaError(
                "temperature is required when a temperature limit is supplied."
            )

        if (
                min_temperature is not None
                and max_temperature is not None
                and min_temperature > max_temperature
        ):
            raise NebulaError(
                "Minimum temperature cannot exceed maximum temperature."
            )

        header_data = OpenData(silo_instant)
        try:
            header_data.db.SetDir('/header')
            coord_sys = int(header_data.db.GetVar("coord_sys"))
        finally:
            header_data.close()

        readers = {
            1: 'get_3Darray',
            2: 'get_2Darray',
            3: 'get_1Darray',
        }
        reader_name = readers.get(coord_sys)
        if reader_name is None:
            raise NebulaError(
                f"Unsupported coordinate-system identifier: {coord_sys}"
            )

        dataio = ReadData(silo_instant)
        try:
            reader = getattr(dataio, reader_name)
            grid_mask = np.array(
                reader('NG_Mask')['data'][self._min_grid_level:],
                copy=True,
            )
        finally:
            dataio.close()

        if not temperature_filter:
            return grid_mask

        temperature = np.asarray(temperature, dtype=np.float64)
        if temperature.shape != grid_mask.shape:
            raise NebulaError(
                "temperature and grid mask must have identical shapes: "
                f"{temperature.shape} != {grid_mask.shape}."
            )

        selected = np.ones(temperature.shape, dtype=bool)

        if min_temperature is not None:
            selected &= temperature >= min_temperature

        if max_temperature is not None:
            selected &= temperature <= max_temperature

        grid_mask[~selected] = 0
        return grid_mask


    ######################################################################################
    # spherical grid # todo: redo this section, move volume calculation and mind dim scaling
    ######################################################################################
    def spherical_grid(self, silo_instant):

        # Open the data for the first silo instant silo
        header_data = OpenData(silo_instant)
        # Set the directory to '/header'
        header_data.db.SetDir('/header')
        # Retrieve what coordinate system is used
        coord_sys = header_data.db.GetVar("coord_sys")
        if not coord_sys == 3:
            raise NebulaError(f"Geometry mismatch {const.COORDINATE_SYSTEMS[coord_sys]}")
        # Retrieve no of nested grid levels
        Nlevels = header_data.db.GetVar("grid_nlevels")
        Ngrid = header_data.db.GetVar("NGrid")
        # close the object
        header_data.close()

        # Store coordinate-system and grid metadata.
        self.geometry_container['coordinate_sys'] = const.COORDINATE_SYSTEMS[coord_sys]
        if not 0 <= self._min_grid_level < Nlevels:
            raise NebulaError("Selected minimum grid level is absent from this snapshot.")
        self.geometry_container['total_levels'] = Nlevels
        self.geometry_container['level_indices'] = list(range(Nlevels))
        self.geometry_container['Nlevel'] = Nlevels
        self.geometry_container['Ngrid'] = Ngrid
        self.geometry_container['dim_scale'] = self.dim_scale
        logger.info("Dimensional scale: %s", self.dim_scale)

        # read silo file
        data = ReadData(silo_instant)
        basic = data.get_1Darray('Density')
        mask = data.get_1Darray("NG_Mask")['data']
        mask = np.array(mask)

        if self.dim_scale == 'cm':
            dims_max = (basic['max_extents'] * unit.cm)
            dims_min = (basic['min_extents'] * unit.cm)
            self.geometry_container['edges_min'] = dims_min
            self.geometry_container['edges_max'] = dims_max
        elif self.dim_scale == 'pc':
            dims_max = (basic['max_extents'] * unit.cm).to(unit.pc)
            dims_min = (basic['min_extents'] * unit.cm).to(unit.pc)
            self.geometry_container['edges_min'] = dims_min
            self.geometry_container['edges_max'] = dims_max
        elif self.dim_scale == 'au':
            dims_max = (basic['max_extents'] * unit.cm).to(unit.au)
            dims_min = (basic['min_extents'] * unit.cm).to(unit.au)
            self.geometry_container['edges_min'] = dims_min
            self.geometry_container['edges_max'] = dims_max

        # radial axis
        rmax = (basic['max_extents'] * unit.cm)
        rmin = (basic['min_extents'] * unit.cm)
        Ngrid = data.ngrid()
        # close the object
        data.close()

        # calculating radial points
        logger.debug("Calculating radial points")
        radius = []
        for level in range(Nlevels):
            level_min = rmin[level].value
            level_max = rmax[level].value
            level_dr = (level_max[0] - level_min[0]) / Ngrid[0]
            r0 = level_min[0] + 0.5 * level_dr
            rn = level_max[0] - 0.5 * level_dr
            r = np.linspace(r0, rn, Ngrid[0])
            radius.append(r)  # append radius of each level

        self._level_radii = [r.copy() for r in radius]

        if Nlevels > 1:
            # last element of the tracer array tracer[Nlevels - 1]
            fine_level = radius[Nlevels - 1]
            # Loop through the tracer array starting from the second-to-last element down to the first element,
            # goes from Nlevels - 2 (second-to-last element) to 0 (first element)
            for i in range(Nlevels - 2, -1, -1):
                # Use the mask array to selectively delete certain elements from tracer[i]. np.where(mask[i] == 0)
                # finds the indices in mask[i] where the value is 0. np.delete(tracer[i], np.where(mask[i] == 0))
                # removes the elements from tracer[i] at those indices.
                coarse_level = np.delete(radius[i], np.where(mask[i] == 0))
                # append the filtered array coarse_level to the result array to fine_level.
                fine_level = np.append(fine_level, coarse_level)
            radius = np.array(fine_level)

        # if the data is single level (uniform grid)
        if Nlevels == 1:
            radius = radius[0] * mask[0]

        # Todo: This has to be a separate method
        # calculating shell volumes
        logger.debug("Calculating shell volumes")
        # Calculating the core volume
        core = 4.0 * const.PI * radius[0] ** 3.0 / 3.0
        # Calculating the shell volumes
        shell_volumes = 4.0 * const.PI * (radius[1:] ** 3 - radius[:-1] ** 3) / 3.0
        # Insert the core volume at the beginning of the shell_volumes array
        shell_volumes = np.insert(shell_volumes, 0, core)

        self.geometry_container['radius'] = radius
        self.geometry_container['shell_volumes'] = shell_volumes

        self.geometry_container['mask'] = mask
        del mask
        self._full_geometry = deepcopy(self.geometry_container)
        self.restrict_grid_levels(self._min_grid_level)

    ######################################################################################
    # cylindrical grid
    ######################################################################################
    def cylindrical_grid(self, silo_instant):

        # Open the data for the first silo instant silo
        header_data = OpenData(silo_instant)
        # Set the directory to '/header'
        header_data.db.SetDir('/header')
        #print(header_data.db.GetToc())
        # Retrieve what coordinate system is used
        coord_sys = header_data.db.GetVar("coord_sys")
        if not coord_sys == 2:
            raise NebulaError(f"Geometry mismatch {const.COORDINATE_SYSTEMS[coord_sys]}")
        # Retrieve no of nested grid levels
        Nlevel = header_data.db.GetVar("grid_nlevels")
        Ngrid = header_data.db.GetVar("NGrid")
        # close the object
        header_data.close()

        # Store coordinate-system and grid metadata.
        self.geometry_container['coordinate_sys'] = const.COORDINATE_SYSTEMS[coord_sys]
        if not 0 <= self._min_grid_level < Nlevel:
            raise NebulaError("Selected minimum grid level is absent from this snapshot.")
        self.geometry_container['total_levels'] = Nlevel
        self.geometry_container['level_indices'] = list(range(Nlevel))
        self.geometry_container['Nlevel'] = Nlevel
        self.geometry_container['Ngrid'] = Ngrid
        self.geometry_container['dim_scale'] = self.dim_scale
        logger.info("Geometry: %s grid levels, dimensional scale %s", Nlevel, self.dim_scale)

        # Read the data from the current silo file
        dataio = ReadData(silo_instant)
        basic = dataio.get_2Darray('Density')  # Retrieve basic simulation data, such as density
        mask = dataio.get_2Darray('NG_Mask')['data']
        dataio.close()  # Close the data file

        logger.info("Retrieving simulation domain information")

        self.geometry_container['mask'] = mask
        del mask
        if self.dim_scale == 'cm':
            dims_max = (basic['max_extents'] * unit.cm)
            dims_min = (basic['min_extents'] * unit.cm)
            self.geometry_container['edges_min'] = dims_min
            self.geometry_container['edges_max'] = dims_max
        elif self.dim_scale == 'pc':
            dims_max = (basic['max_extents'] * unit.cm).to(unit.pc)
            dims_min = (basic['min_extents'] * unit.cm).to(unit.pc)
            self.geometry_container['edges_min'] = dims_min
            self.geometry_container['edges_max'] = dims_max
        elif self.dim_scale == 'au':
            dims_max = (basic['max_extents'] * unit.cm).to(unit.au)
            dims_min = (basic['min_extents'] * unit.cm).to(unit.au)
            self.geometry_container['edges_min'] = dims_min
            self.geometry_container['edges_max'] = dims_max
        del basic
        self._full_geometry = deepcopy(self.geometry_container)
        self.restrict_grid_levels(self._min_grid_level)


    ######################################################################################
    # cylindrical grid 2D volume
    ######################################################################################
    def get_grid_volumes_2D(self):
        """
        Computes the volume of grid cells in a cylindrical coordinate system.

        This function assumes a static grid with multiple refinement levels.
        The volume of each cylindrical shell segment is calculated using the difference
        in squared radii multiplied by the cell height.

        Returns:
            list[np.ndarray]: A list of 2D arrays containing cell volumes in the selected length unit cubed ('cm', 'pc', or 'au').
        """

        # Extract necessary grid parameters
        Ngrid, Nlevel = self.geometry_container['Ngrid'], self.geometry_container['Nlevel']
        edges_min, edges_max = self.geometry_container['edges_min'], self.geometry_container['edges_max']
        mask_shapes = [mask.shape for mask in self.geometry_container['mask']]

        # Initialize cell volumes
        cell_volume = [np.zeros(shape) for shape in mask_shapes]

        # compute cell volume for each refinement level
        for level in range(Nlevel):
            delta_z = (edges_max[level][0].value - edges_min[level][0].value) / Ngrid[0]
            delta_r = (edges_max[level][1].value - edges_min[level][1].value) / Ngrid[1]

            r_cells = np.arange(Ngrid[1]) * delta_r
            r_squares = (r_cells + delta_r) ** 2 - r_cells ** 2

            # Compute cell volumes
            cell_volume[level][:, :] = delta_z * const.PI * r_squares[:, None]

        return cell_volume


    ######################################################################################
    # get parameter //todo: this is not clear
    ######################################################################################
    def get_parameter(self, parameter, silo_instant):
        '''
        Method will return the parameter value for a spherical nested grid

        Parameters
        ----------
        parameter physical
        silo_instant

        Returns
        -------
        physical parameter value for a spherical nested grid
        '''

        # 1 dimensional (spherical) ######################################################
        if self.geometry_container['coordinate_sys'] == 'spherical':
            # get nested grid level
            Nlevel = self.geometry_container['Nlevel']

            # pypion ReadDate object
            data = ReadData(silo_instant)
            # get parameter values
            parameter = data.get_1Darray(parameter)['data'][self._min_grid_level:]
            # get mask
            mask = data.get_1Darray("NG_Mask")['data'][self._min_grid_level:]
            data.close()
            # if the data is single level (uniform grid)
            if Nlevel == 1:
                return parameter[0] * mask[0]

            # last element of the parameter array parameter[Nlevels - 1]
            fine_level = parameter[Nlevel - 1]
            # Loop through the parameter array starting from the second-to-last element down to the first element,
            # goes from Nlevels - 2 (second-to-last element) to 0 (first element)
            for i in range(Nlevel - 2, -1, -1):
                # Use the mask array to selectively delete certain elements from parameter[i]. np.where(mask[i] == 0)
                # finds the indices in mask[i] where the value is 0. np.delete(parameter[i], np.where(mask[i] == 0))
                # removes the elements from parameter[i] at those indices.
                coarse_level = np.delete(parameter[i], np.where(mask[i] == 0))
                # append the filtered array coarse_level to the result array to fine_level.
                fine_level = np.append(fine_level, coarse_level)
            return np.array(fine_level)
        # end of 1 dimensional ***********************************************************

        # 2 dimensional (cylindrical) ####################################################
        if self.geometry_container['coordinate_sys'] == 'cylindrical':

            # get nested grid level
            Nlevel = self.geometry_container['Nlevel']
            # pypion ReadDate object
            data = ReadData(silo_instant)
            # get parameter values
            parameter = data.get_2Darray(parameter)['data'][self._min_grid_level:]
            # get mask
            mask = data.get_2Darray("NG_Mask")['data']
            data.close()
            return parameter
        # end of 2 dimensional ***********************************************************
        # 3 dimensional (cartesian) ######################################################
        # end of 3 dimensional ***********************************************************
