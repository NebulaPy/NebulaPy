from .Chianti import chianti
from NebulaPy.src.Utils import getPionSymbol, get_element_symbol
import NebulaPy.src.Chianti as nebula_chianti
import NebulaPy.src.Constants as const
import os
import multiprocessing as mp
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
def compute_row_spectrum(workerQ, doneQ, spectrum_obj):
    """Compute and reduce one complete grid row at a time."""

    while True:
        level = None
        row = None
        try:
            task = workerQ.get()

            if task is None:
                break

            (
                level,
                row,
                row_temperature,
                row_ne,
                row_species_densities,
                row_volume,
                row_grid_mask,
            ) = task

            row_species_spectra = (
                spectrum_obj.compute_row_species_spectra(
                    row_temperature=row_temperature,
                    row_ne=row_ne,
                    row_species_densities=row_species_densities,
                    row_volume=row_volume,
                    row_grid_mask=row_grid_mask,
                )
            )

            doneQ.put(
                (
                    "RESULT",
                    level,
                    row,
                    row_species_spectra,
                    None,
                )
            )

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
    """Construct spectra from CHIANTI emissivities and simulation plasma data.

    The class first selects the wavelength interval and radiative processes.
    :meth:`generate_spectrum` then builds a wavelength grid, converts the input
    hydrodynamic rows into worker tasks, evaluates CHIANTI coefficients, and
    immediately integrates them with electron density, ion density, cell volume,
    and the supplied grid mask.

    Notes
    -----
    The lowercase class name is retained for compatibility with the public API.
    Wavelengths are measured in Angstrom and photon energies in keV.  The final
    ``Spectrum`` attribute contains luminosity per wavelength after integration
    over the full ``4*pi`` solid angle.
    """

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
            ion_list=None,
            filtername=None,
            filterfactor=None,
            allLines=True,
            userGrid=False,
            MPNcores=4,
            gridSize=1000,
            progress=True,
    ):
        """Store configuration and discover the available CHIANTI ions.

        Parameters
        ----------
        min_wavelength, max_wavelength : float, optional
            Lower and upper wavelength bounds in Angstrom.  Both values must be
            supplied together.  They take precedence if energy bounds are also
            supplied.
        min_photon_energy, max_photon_energy : float, optional
            Photon-energy bounds in keV.  They are converted to wavelength via
            ``wavelength = KEV_ANGSTROM / energy``; consequently the maximum
            energy gives the minimum wavelength.
        doBremsstrahlung, doFreebound, doLine, doTwophoton : bool
            Enable free-free, free-bound, discrete-line, and two-photon
            emission, respectively.  At least one process must be enabled.
        elements : sequence
            Elements passed to :class:`chianti` when building the ion metadata
            container.
        ion_list : sequence of str, optional
            Ions included in the calculation, using PION notation such as
            ``["O6+", "O7+", "Fe24+"]``. If omitted, all available ions
            belonging to the selected elements are included.
        filtername, filterfactor
            Stored line-profile configuration for API compatibility.  The
            values are not consumed directly by this class at present.
        allLines : bool
            Stored line-selection flag for API compatibility.
        userGrid : bool
            If true, use a uniformly spaced wavelength grid.  Otherwise use
            the unique CHIANTI line wavelengths plus the requested endpoints.
        MPNcores : int
            Maximum number of worker processes.  The actual count is capped by
            both the machine CPU count and, later, the number of tasks.
        gridSize : int
            Number of samples in a user-defined uniform wavelength grid.
        progress : bool
            Enable progress displays during expensive loops.
        """

        # ``N_wvl`` is unknown until ``setup_wavelength_grid`` creates the grid.
        self.N_wvl = None
        # Store one Boolean for each optional radiative process.
        self.bremsstrahlung = doBremsstrahlung
        self.freebound = doFreebound
        self.line = doLine
        self.twophoton = doTwophoton
        # Preserve optional filtering and line-selection settings for callers.
        self.filtername = filtername
        self.filterfactor = filterfactor
        self.allLines = allLines
        # All progress helpers consult this single flag.
        self.progress = progress


        logger.info("Initializing spectrum calculation")

        # Prefer an explicitly supplied wavelength interval.
        if min_wavelength is not None and max_wavelength is not None:
            self.min_wvl, self.max_wvl = min_wavelength, max_wavelength

        # Otherwise convert a complete energy interval from keV to Angstrom.
        elif min_photon_energy is not None and max_photon_energy is not None:
            # Energy and wavelength are inverse, so the bounds exchange order.
            self.min_wvl = const.KEV_ANGSTROM / max_photon_energy
            self.max_wvl = const.KEV_ANGSTROM / min_photon_energy

        # A partial interval cannot define the requested spectral domain.
        else:
            raise NebulaError(
                "Provide either wavelength range or photon energy range."
            )

        # Non-positive wavelengths are physically invalid and risk bad divisions.
        if self.min_wvl <= 0.0 or self.max_wvl <= 0.0:
            raise NebulaError("Wavelength bounds must be positive.")

        # The algorithms below assume an increasing closed wavelength interval.
        if self.min_wvl >= self.max_wvl:
            raise NebulaError(
                "Minimum wavelength must be smaller than maximum wavelength."
            )

        # Avoid doing an expensive DEM calculation that cannot emit anything.
        if not (doBremsstrahlung or doFreebound or doLine or doTwophoton):
            raise NebulaError("No emission processes specified")

        # Build human-readable process names solely for the initialization log.
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

        #####################################################################
        # check for elements
        if elements is None:
            raise NebulaError(
                "elements must be supplied for the spectrum calculation."
            )

        if isinstance(elements, str):
            elements = [elements]
        else:
            elements = list(elements)

        if not elements:
            raise NebulaError(
                "elements must contain at least one element."
            )

        unsupported_elements = set(elements) - set(const.SUPPORTED_ELEMENTS)

        if unsupported_elements:
            raise NebulaError(
                "Unsupported elements: "
                + ", ".join(sorted(unsupported_elements))
                + ". Supported elements are: "
                + ", ".join(const.SUPPORTED_ELEMENTS)
            )

        if set(elements) == set(const.SUPPORTED_ELEMENTS):
            logger.info(
                "Spectrum elements: using all supported elements"
            )
        else:
            logger.info(
                "Spectrum elements restricted to: %s",
                ", ".join(elements),
            )

        if ion_list is not None:
            if isinstance(ion_list, str):
                ion_list = [ion_list]
            else:
                ion_list = list(ion_list)

            if not ion_list:
                raise NebulaError(
                    "ion_list must contain at least one ion when supplied."
                )

            missing_elements = {
                get_element_symbol(ion)
                for ion in ion_list
            } - set(elements)

            if missing_elements:
                raise NebulaError(
                    "Elements required by ion_list are missing from elements: "
                    + ", ".join(sorted(missing_elements))
                )

            logger.info(
                "Spectrum ions restricted to: %s",
                ", ".join(ion_list),
            )
        else:
            logger.info(
                "Spectrum ions: using all available ions for selected elements"
            )

        #####################################################################
        # The mapping is populated as ``CHIANTI ion name -> ion metadata``.
        self.chianti_species_attributes = {}
        # Discover every ion and supported process for the requested elements.
        self.build_species_attributes(elements)
        # Keep all species when ion_list is None.
        self.ion_list = ion_list
        self.required_density_ions = None
        # Otherwise, retain only the requested ions.
        if ion_list is not None:
            requested_ions = set(ion_list)
            selected_species = {
                chianti_ion: attributes
                for chianti_ion, attributes
                in self.chianti_species_attributes.items()
                if getPionSymbol(chianti_ion) in requested_ions
            }
            available_ions = {
                getPionSymbol(chianti_ion)
                for chianti_ion in selected_species
            }
            missing_ions = requested_ions - available_ions
            if missing_ions:
                raise NebulaError(
                    "Requested ions are unavailable: "
                    + ", ".join(sorted(missing_ions))
                )
            self.chianti_species_attributes = selected_species

            required_density_ions = list(dict.fromkeys(ion_list))
            if self.freebound:
                freebound_dependencies = set()
                for attributes in selected_species.values():
                    lower_chianti_ion = attributes.get("lower", 0)
                    if "fb" in attributes["keys"] and lower_chianti_ion:
                        freebound_dependencies.add(
                            getPionSymbol(lower_chianti_ion)
                        )

                required_density_ions.extend(
                    sorted(
                        freebound_dependencies - set(required_density_ions)
                    )
                )

            self.required_density_ions = required_density_ions

            if set(required_density_ions) != set(ion_list):
                logger.info(
                    "Additional ion densities required for free-bound emission: %s",
                    ", ".join(
                        ion
                        for ion in required_density_ions
                        if ion not in set(ion_list)
                    ),
                )

        #####################################################################
        # The numerical wavelength grid is deferred until spectrum generation.
        self.WavelengthGrid = []
        # ``userGrid`` chooses uniform sampling instead of CHIANTI line sampling.
        self.userGrid = userGrid
        # ``gridSize`` is used only for the uniform-grid branch.
        self.gridSize = gridSize

        # Never request more worker processes than the host can provide.
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
        # Only CHIANTI species are implemented; PyNeb species are future work.

        # A harmless dummy plasma state is sufficient because only metadata is read.
        chianti_spec = chianti(
            pion_elements=elements,
            temperature=[1.0e7],  # K; required by the CHIANTI wrapper constructor.
            ne=[1.0e9],  # cm^-3; required by the CHIANTI wrapper constructor.
        )
        # Retain the wrapper's ion metadata (Z, stage, processes, adjacent ions).
        self.chianti_species_attributes = chianti_spec.species_attributes_container
        # Drop the temporary wrapper; it is not needed for later calculations.
        del chianti_spec

    ######################################################################################
    # SETUP WAVELENGTH GRID
    ######################################################################################
    def setup_wavelength_grid(self, min_wvl, max_wvl, user_grid=False, grid_size=None):
        """Build and store the wavelength samples used by every coefficient.

        ``user_grid=False`` obtains all available CHIANTI line wavelengths in
        the requested interval.  ``user_grid=True`` creates ``grid_size`` evenly
        spaced values instead.  In both modes the result is sorted, stored in
        ``self.WavelengthGrid``, and counted in ``self.N_wvl``.
        """

        if not user_grid:
            logger.debug("Setting up the default CHIANTI wavelength grid")
        else:
            logger.debug("Setting up a uniform wavelength grid")

        # Reject a reversed or zero-width interval before querying every ion.
        if min_wvl >= max_wvl:
            raise NebulaError(
                " Minimum wavelength must be smaller than maximum wavelength."
            )

        # Ion metadata is required to know which species provide line data.
        if not self.chianti_species_attributes:
            raise NebulaError(
                " Species Attributes Container is not initialized or is empty."
            )

        # These representative plasma values are used only to instantiate ions.
        dummy_temperature = [2.0e6]
        dummy_ne = [1.0e9]
        # Accumulate individual line wavelengths from all line-emitting species.
        lines = []

        # Materialize the items so the progress helper knows the total length.
        species_list = list(self.chianti_species_attributes.items())

        for species, attributes in track(
                species_list,
                description="Retrieving CHIANTI wavelength grid",
                unit="species",
                enabled=self.progress,
        ):

            # Species without the ``line`` capability cannot contribute samples.
            if 'line' not in attributes['keys']:
                continue

            # Construct a short-lived wrapper for this particular CHIANTI ion.
            chianti_nebula_object = nebula_chianti.chianti(
                chianti_ion=species,
                temperature=dummy_temperature,
                ne=dummy_ne,
            )

            try:
                ion_lines = chianti_nebula_object.get_allLines()

                # ``None`` denotes an ion with no available line wavelengths.
                if ion_lines is not None:
                    # Normalize possible NumPy output into ordinary float values.
                    lines.extend(
                        np.asarray(ion_lines, dtype=np.float64).tolist()
                    )

            finally:
                # Always release the ion wrapper, including when CHIANTI raises.
                del chianti_nebula_object

        # ------------------------------------------------------------------
        # CHIANTI line-based wavelength grid
        # ------------------------------------------------------------------
        if not user_grid:

            # Use a numerical vector for masking and concatenation below.
            lines = np.asarray(lines, dtype=np.float64)

            if lines.size == 0:
                raise NebulaError(
                    " No CHIANTI lines found for the selected species."
                )

            # Keep only transitions inside the caller's closed interval.
            selected_lines = lines[
                (lines >= min_wvl) &
                (lines <= max_wvl)
                ]

            # Include exact bounds and remove duplicate transition wavelengths.
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

            # Two points are the minimum needed to represent both interval ends.
            if grid_size < 2:
                raise NebulaError(
                    " grid_size must be at least 2."
                )

            # ``linspace`` includes both endpoints and returns float64 samples.
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

        # Sorting makes downstream wavelength-indexed arrays deterministic.
        wavelength_grid.sort()

        if wavelength_grid.size == 0:
            raise NebulaError(
                " No wavelength points found in the requested wavelength range."
            )

        # Save the grid and its width for coefficient-array allocation.
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
        """Write a text catalogue of transitions in the configured interval.

        Parameters
        ----------
        Filebase : str
            Prefix used to create ``<Filebase>_LineCatalog.txt``.
        OutDir : str
            Existing destination directory.  This method deliberately does not
            create directories so that a misspelled path fails visibly.
        """

        # An empty base name would produce an ambiguous output filename.
        if not Filebase:
            raise NebulaError(
                "line_cataloger: output file base name not specified."
            )

        # Require callers to select a destination explicitly.
        if not OutDir:
            raise NebulaError(
                "line_cataloger: output directory not specified."
            )

        # Validate before opening the file and report the original path clearly.
        if not os.path.isdir(OutDir):
            raise NebulaError(
                f"line_cataloger: output directory does not exist: {OutDir}"
            )

        # ``os.path.join`` handles platform-specific path separators.
        outfile = os.path.join(
            OutDir,
            f"{Filebase}_LineCatalog.txt"
        )

        # Representative state required to initialize each CHIANTI ion wrapper.
        dummy_temperature = [2.0e6]
        dummy_ne = [1.0e9]

        # Entries are tuples beginning with wavelength, which enables sorting.
        line_catalog = []

        # Materialize the mapping for progress accounting.
        species_list = list(self.chianti_species_attributes.items())

        for species, attributes in track(
                species_list,
                description="Cataloging CHIANTI wavelength grid",
                unit="species",
                enabled=self.progress,
        ):

            # Skip ions that CHIANTI reports as having no bound-bound transitions.
            if 'line' not in attributes['keys']:
                continue

            # Instantiate the current ion to retrieve its transition metadata.
            chianti_nebula_object = nebula_chianti.chianti(
                chianti_ion=species,
                temperature=dummy_temperature,
                ne=dummy_ne,
            )

            try:
                # The transition mapping supplies wavelength, Einstein A, and states.
                ionTransitions = (
                    chianti_nebula_object.get_allLineTransitions()
                )

                # Use CHIANTI's readable notation, for example ``Fe XXV``.
                spectroscopic_name = (
                    chianti_nebula_object.chianti_ion.Spectroscopic
                )

                # Convert each field to an array so one Boolean mask selects all fields.
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

                # Select transitions in the same closed interval as the spectrum.
                mask = (
                        (wavelengths_all >= self.min_wvl)
                        &
                        (wavelengths_all <= self.max_wvl)
                )

                # Avoid allocating sliced arrays when this ion contributes no lines.
                if not np.any(mask):
                    continue

                # Apply an identical mask to keep transition fields aligned by index.
                wavelengths = wavelengths_all[mask]
                Avalues = Avalues_all[mask]
                lower_states = lower_states_all[mask]
                upper_states = upper_states_all[mask]

                # Package each selected transition into one sortable catalogue row.
                for wvl, aval, lower, upper in zip(
                        wavelengths,
                        Avalues,
                        lower_states,
                        upper_states
                ):
                    # Photon energy follows E[keV] = hc / wavelength[Angstrom].
                    energy_keV = const.KEV_ANGSTROM / wvl
                    # Wavelength is included in the display label at fixed precision.
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
                # Release this species even if transition processing raises an error.
                del chianti_nebula_object

        # Sort numerically by the first tuple element: wavelength.
        line_catalog.sort(key=lambda entry: entry[0])

        # Text mode replaces any existing catalogue with the same requested name.
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

            # Write one fixed-width row for each transition, omitting sort wavelength.
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
    def compute_row_species_spectra(
            self,
            row_temperature,
            row_ne,
            row_species_densities,
            row_volume,
            row_grid_mask,
    ):
        """Return DEM-weighted wavelength vectors for one complete grid row.

        CHIANTI still evaluates all cells in the row vectorially.  Each process
        is weighted and reduced over cells immediately, so only one species-by-
        wavelength result dictionary survives to be transferred to the parent.
        """

        row_temperature = np.asarray(row_temperature, dtype=np.float64)
        row_ne = np.asarray(row_ne, dtype=np.float64)
        row_volume = np.asarray(row_volume, dtype=np.float64)
        row_grid_mask = np.asarray(row_grid_mask, dtype=np.float64)
        row_species_densities = {
            species: np.asarray(density, dtype=np.float64)
            for species, density in row_species_densities.items()
        }

        # Preserve the legacy spectrum temperature domain without allocating
        # the former 300-bin DEM intermediates.
        valid_temperature = (
            (row_temperature >= 100.0)
            & (row_temperature <= 1.0e9)
        )
        cell_weight = (
            row_ne
            * row_volume
            * row_grid_mask
            * valid_temperature
        )
        row_species_spectra = {}

        def accumulate(output_species, density_species, coefficients):
            """Weight one coefficient matrix and sum its cell axis."""
            coefficients = np.asarray(coefficients, dtype=np.float64)
            if coefficients.ndim == 1:
                coefficients = coefficients[np.newaxis, :]

            density = row_species_densities[density_species]
            weighted_cells = cell_weight * density
            spectrum_vector = np.einsum(
                "i,ij->j",
                weighted_cells,
                coefficients,
                optimize=True,
            )
            if output_species not in row_species_spectra:
                row_species_spectra[output_species] = np.zeros(
                    self.N_wvl,
                    dtype=np.float64,
                )
            row_species_spectra[output_species] += spectrum_vector

        for chianti_species, attributes in self.chianti_species_attributes.items():
            pion_species = getPionSymbol(chianti_species)
            lower_chianti_ion = attributes.get("lower", 0)
            lower_pion_species = (
                getPionSymbol(lower_chianti_ion)
                if lower_chianti_ion
                else None
            )
            processes = attributes["keys"]

            has_nonfb_output = (
                pion_species in row_species_densities
                and (
                    (self.bremsstrahlung and "ff" in processes)
                    or (self.line and "line" in processes)
                    or (
                        self.twophoton
                        and "line" in processes
                        and (attributes["Z"] - attributes["Ion"]) in (0, 1)
                        and not attributes["Dielectronic"]
                    )
                )
            )
            has_fb_output = (
                self.freebound
                and "fb" in processes
                and pion_species in row_species_densities
                and lower_pion_species in row_species_densities
            )

            if not (has_nonfb_output or has_fb_output):
                continue

            CHIANTI = chianti(
                chianti_ion=chianti_species,
                temperature=row_temperature,
                ne=row_ne,
            )

            try:
                if self.bremsstrahlung and "ff" in processes:
                    accumulate(
                        pion_species,
                        pion_species,
                        CHIANTI.get_bremsstrahlung_coefficients(
                            wavelength=self.WavelengthGrid,
                        ),
                    )

                if self.line and "line" in processes:
                    line_coefficients = CHIANTI.get_line_coefficients(
                        wavelength=self.WavelengthGrid,
                    )
                    line_coefficients = np.asarray(
                        line_coefficients,
                        dtype=np.float64,
                    )
                    if line_coefficients.ndim == 1:
                        line_coefficients = line_coefficients[np.newaxis, :]
                    line_coefficients = np.divide(
                        line_coefficients,
                        row_ne[:, None],
                        out=np.zeros_like(line_coefficients),
                        where=(
                            row_ne[:, None]
                            >= const.ELECTRON_DENSITY_FLOOR
                        ),
                    )
                    accumulate(
                        pion_species,
                        pion_species,
                        line_coefficients,
                    )

                if (
                        self.twophoton
                        and "line" in processes
                        and (attributes["Z"] - attributes["Ion"]) in (0, 1)
                        and not attributes["Dielectronic"]
                ):
                    accumulate(
                        pion_species,
                        pion_species,
                        CHIANTI.get_twophoton_coefficients(
                            wavelength=self.WavelengthGrid,
                        ),
                    )

                if has_fb_output:
                    accumulate(
                        lower_pion_species,
                        pion_species,
                        CHIANTI.get_freebound_coefficients(
                            wavelength=self.WavelengthGrid,
                        ),
                    )
            finally:
                CHIANTI.terminate()

        return row_species_spectra


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
        """Generate and store the integrated spectrum of a two-dimensional grid.

        Parameters
        ----------
        temperature : array-like
            Cell temperatures in K.  Expected shape is ``(levels, rows, cells)``;
            the first two axes define multiprocessing tasks and the final axis is
            the set of cells processed together.
        ne : array-like
            Electron number density in cm^-3, with the same shape as temperature.
        species_densities : mapping
            PION ion symbol to ion number-density array.  Every value must have
            the same shape as temperature.
        grid_volume : array-like
            Physical volume of each cell, shape-matched to temperature.
        grid_mask : array-like
            Per-cell multiplicative inclusion or filling-factor mask.  Zero
            excludes a cell; fractional values proportionally weight it.

        Side Effects
        ------------
        Builds ``self.WavelengthGrid`` and stores the final one-dimensional
        luminosity-per-wavelength array in ``self.Spectrum``.  Individual ion
        spectra are local intermediates and are summed before this method ends.
        """

        ##########################################################################
        # Coefficient generation cannot proceed without the ion metadata map.
        if not self.chianti_species_attributes:
            raise NebulaError(
                "Species Attributes Container is not initialized or is empty."
            )

        ##########################################################################
        # Normalize physical grids to float64 for stable vectorized arithmetic.
        temperature = np.asarray(temperature, dtype=np.float64)
        ne = np.asarray(ne, dtype=np.float64)
        grid_volume = np.asarray(grid_volume, dtype=np.float64)
        grid_mask = np.asarray(grid_mask, dtype=np.float64)

        # Element-wise products below require all four physical grids to align.
        if not (
                temperature.shape == ne.shape ==
                grid_volume.shape == grid_mask.shape
        ):
            raise NebulaError(
                "Input arrays have inconsistent shapes."
            )

        # Normalize every density independently while preserving dictionary keys.
        species_densities = {
            species: np.asarray(density, dtype=np.float64)
            for species, density in species_densities.items()
        }

        # Validate ion-density shapes now rather than failing inside a worker.
        for species, density in species_densities.items():

            if density.shape != temperature.shape:
                raise NebulaError(
                    f"Species density input arrays have inconsistent "
                    f"shapes for species {species}."
                )


        ##########################################################################
        # Build the common spectral axis used by every ion and radiative process.
        self.setup_wavelength_grid(self.min_wvl, self.max_wvl,
                                   user_grid=self.userGrid,
                                   grid_size=self.gridSize)

        ##########################################################################
        # The parent retains only one wavelength vector per supplied species.
        Spectrum = {
            species: np.zeros(self.N_wvl, dtype=np.float64)
            for species in species_densities
        }

        N_grid_level = len(temperature)
        Ntasks = sum(len(level_rows) for level_rows in temperature)
        if Ntasks == 0:
            raise NebulaError("Spectrum input grid contains no rows.")

        proc = min(self.proc, Ntasks)
        logger.info(
            "Multiprocessing: %s total tasks across %s levels, up to %s workers",
            Ntasks,
            N_grid_level,
            proc,
        )

        # Queue capacity limits serialized row data and completed results to the
        # number of active workers instead of allowing memory growth with Ntasks.
        workerQ = mp.Queue(maxsize=proc)
        doneQ = mp.Queue(maxsize=proc)
        processes = []

        for _ in range(proc):
            process = mp.Process(
                target=compute_row_spectrum,
                args=(workerQ, doneQ, self),
            )
            process.start()
            processes.append(process)

        task_indices = iter(
            (level, row)
            for level in range(N_grid_level)
            for row in range(len(temperature[level]))
        )

        def build_task(level, row):
            return (
                level,
                row,
                temperature[level, row],
                ne[level, row],
                {
                    species: density[level, row]
                    for species, density in species_densities.items()
                },
                grid_volume[level, row],
                grid_mask[level, row],
            )

        submitted = 0
        for _ in range(proc):
            level, row = next(task_indices)
            workerQ.put(build_task(level, row))
            submitted += 1

        completed = 0
        logger.debug("Computing spectrum")

        try:
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
                        result_or_error,
                        worker_traceback,
                    ) = doneQ.get()

                    if status == "ERROR":
                        logger.error(
                            "Spectrum worker failed at grid level %s, row %s: %s",
                            level,
                            row,
                            result_or_error,
                        )
                        logger.debug(
                            "Spectrum worker traceback:\n%s",
                            worker_traceback,
                        )
                        raise NebulaError(result_or_error)

                    if status != "RESULT":
                        raise NebulaError(
                            f"Unknown multiprocessing result status: {status!r}"
                        )

                    for species, row_spectrum in result_or_error.items():
                        Spectrum[species] += row_spectrum

                    completed += 1
                    progress.advance()

                    if submitted < Ntasks:
                        next_level, next_row = next(task_indices)
                        workerQ.put(build_task(next_level, next_row))
                        submitted += 1
        except Exception:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise
        finally:
            if completed == Ntasks:
                for _ in range(proc):
                    workerQ.put(None)

            for process in processes:
                process.join()

            workerQ.close()
            doneQ.close()

        # Stack ion arrays and sum them element-wise along the species axis.
        if Spectrum:
            integrated_spectrum = np.sum(
                np.stack(list(Spectrum.values())),
                axis=0,
            )
        # No matching input ions produces a valid all-zero spectrum of known length.
        else:
            integrated_spectrum = np.zeros(
                self.N_wvl,
                dtype=np.float64,
            )

        # Coefficients are per steradian; 4*pi integrates isotropic emission over
        # the full solid angle and produces total luminosity per wavelength.
        self.Spectrum = 4.0 * const.PI * integrated_spectrum
