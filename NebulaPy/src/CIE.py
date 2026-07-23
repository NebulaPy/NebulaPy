"""Collisional-ionization-equilibrium ion fractions and number densities."""

import os

import numpy as np

from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import NebulaError, get_logger
from NebulaPy.src.Utils import get_element_symbol, getPionSymbol


logger = get_logger(__name__)


class cieMode:
    """Read and interpolate the NebulaPy CIE ion-balance table."""

    def __init__(self):
        database = os.environ.get("NEBULAPYDB")
        if not database:
            message = "required database dir missing, install database to proceed"
            logger.error("Cannot initialize CIE mode: %s", message)
            raise NebulaError(message)

        self.cie_file = os.path.join(database, "IonBalance", "CIE.txt")
        self.data = None
        self.col_index = None
        self.AllSpecies = None
        logger.debug("CIE table path configured: %s", self.cie_file)

    def _ensure_loaded(self):
        """Load the table on first use."""
        if self.data is None:
            logger.debug("CIE table is not loaded; starting lazy load")
            self.load_cie_file()

    def load_cie_file(self):
        """Load and validate the CIE ion-fraction table."""
        if not os.path.isfile(self.cie_file):
            message = f"CIE file not found: {self.cie_file}"
            logger.error(message)
            raise NebulaError(message)

        header = None
        data = []
        logger.info("Loading collisional ionization equilibrium table")

        with open(self.cie_file, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if header is None:
                    header = parts
                    if not header or header[0] != "log_T":
                        message = "CIE table must start with a 'log_T' column"
                        logger.error("%s: %s", message, self.cie_file)
                        raise NebulaError(message)
                    logger.debug(
                        "CIE header found at line %s with %s ion columns",
                        line_number,
                        len(header) - 1,
                    )
                    continue

                if len(parts) != len(header):
                    message = (
                        f"CIE table row {line_number} has {len(parts)} columns; "
                        f"expected {len(header)}"
                    )
                    logger.error(message)
                    raise NebulaError(message)
                try:
                    data.append([float(value) for value in parts])
                except ValueError as exc:
                    message = (
                        f"CIE table row {line_number} contains non-numeric data"
                    )
                    logger.error(message)
                    raise NebulaError(message) from exc

        if header is None:
            message = "CIE file header not found."
            logger.error("%s File: %s", message, self.cie_file)
            raise NebulaError(message)
        if not data:
            message = "CIE file contains no data."
            logger.error("%s File: %s", message, self.cie_file)
            raise NebulaError(message)

        table = np.asarray(data, dtype=np.float64)
        if not np.all(np.isfinite(table)):
            message = "CIE table contains non-finite values"
            logger.error(message)
            raise NebulaError(message)
        if np.any(np.diff(table[:, 0]) <= 0.0):
            message = "CIE temperature grid must be strictly increasing"
            logger.error(message)
            raise NebulaError(message)
        if np.any(table[:, 1:] < 0.0):
            message = "CIE ion fractions must be non-negative"
            logger.error(message)
            raise NebulaError(message)

        try:
            species = [getPionSymbol(name) for name in header[1:]]
        except NebulaError as exc:
            message = f"invalid CIE table header: {exc}"
            logger.error(message)
            raise NebulaError(message) from exc
        if len(species) != len(set(species)):
            message = "CIE table contains duplicate ion columns"
            logger.error(message)
            raise NebulaError(message)

        self.data = table
        self.col_index = {"log_T": 0}
        self.col_index.update(
            {ion: index for index, ion in enumerate(species, start=1)}
        )
        self.AllSpecies = np.asarray(species, dtype=str)

        logger.info(
            "CIE table loaded: %s temperature points, %s ion-fraction columns",
            table.shape[0],
            table.shape[1] - 1,
        )
        logger.debug(
            "CIE log-temperature range: %.6g to %.6g",
            table[0, 0],
            table[-1, 0],
        )

    @staticmethod
    def _validate_temperature(temperature):
        values = np.asarray(temperature, dtype=float)
        if values.size == 0:
            message = "Temperature cannot be empty"
            logger.error(message)
            raise ValueError(message)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            message = "Temperature must contain finite, positive values"
            logger.error(message)
            raise ValueError(message)
        return values

    def _interpolate_cie_fraction(self, ion, temperature):
        """Interpolate a validated temperature array without emitting logs."""
        scalar = temperature.ndim == 0
        log_temperature = np.log10(np.atleast_1d(temperature))
        temperature_grid = self.data[:, self.col_index["log_T"]]
        fraction_grid = self.data[:, self.col_index[ion]]
        clipped_log_temperature = np.clip(
            log_temperature, temperature_grid[0], temperature_grid[-1]
        )
        fraction = np.interp(
            clipped_log_temperature, temperature_grid, fraction_grid
        )
        return fraction[0] if scalar else fraction

    def get_cie_fraction(self, ion, temperature):
        """Interpolate an ion fraction at scalar or array temperatures in K.

        Temperatures outside the tabulated range use the nearest boundary value.
        """
        self._ensure_loaded()
        if ion not in self.col_index:
            message = f"Ion '{ion}' not found."
            logger.error(message)
            raise KeyError(message)

        temperature = self._validate_temperature(temperature)
        temperature_grid = self.data[:, self.col_index["log_T"]]
        log_temperature = np.log10(np.atleast_1d(temperature))
        clipped_count = np.count_nonzero(
            (log_temperature < temperature_grid[0])
            | (log_temperature > temperature_grid[-1])
        )
        logger.debug(
            "Interpolating CIE fraction for %s over shape %s; "
            "%s temperatures clipped to table bounds",
            ion,
            temperature.shape,
            clipped_count,
        )
        return self._interpolate_cie_fraction(ion, temperature)

    def build_cie_number_densities(
        self, element_mass_fractions, temperature, density
    ):
        """Return ion number densities in cm^-3 for all tabulated ions.

        ``element_mass_fractions`` maps element symbols to scalar or array mass
        fractions, ``temperature`` is in K, and ``density`` is the total mass
        density in g cm^-3. All array inputs must be mutually broadcastable.
        Elements omitted from the mapping receive zero number density.
        """
        self._ensure_loaded()
        if not isinstance(element_mass_fractions, dict):
            message = "element_mass_fractions must be a dictionary"
            logger.error(message)
            raise TypeError(message)

        temperature = self._validate_temperature(temperature)
        density = np.asarray(density, dtype=float)
        if density.size == 0:
            message = "Density cannot be empty"
            logger.error(message)
            raise ValueError(message)
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            message = "Density must contain finite, non-negative values"
            logger.error(message)
            raise ValueError(message)

        unknown = set(element_mass_fractions) - set(const.ATOMIC_MASS)
        if unknown:
            names = ", ".join(sorted(unknown))
            message = f"Unsupported element mass fraction(s): {names}"
            logger.error(message)
            raise KeyError(message)

        validated_mass_fractions = {}
        for element, value in element_mass_fractions.items():
            fraction = np.asarray(value, dtype=float)
            if fraction.size == 0:
                message = f"Mass fraction for {element} cannot be empty"
                logger.error(message)
                raise ValueError(message)
            if not np.all(np.isfinite(fraction)) or np.any(fraction < 0.0):
                message = (
                    f"Mass fraction for {element} must be finite and non-negative"
                )
                logger.error(message)
                raise ValueError(message)
            validated_mass_fractions[element] = fraction

        try:
            arrays = np.broadcast_arrays(
                temperature, density, *validated_mass_fractions.values()
            )
        except ValueError as exc:
            message = (
                "Temperature, Density, and mass fractions must be broadcastable"
            )
            logger.error(
                "%s; temperature shape=%s, density shape=%s, "
                "mass-fraction shapes=%s",
                message,
                temperature.shape,
                density.shape,
                {
                    element: fraction.shape
                    for element, fraction in validated_mass_fractions.items()
                },
            )
            raise ValueError(message) from exc

        temperature, density = arrays[:2]
        validated_mass_fractions = dict(
            zip(validated_mass_fractions, arrays[2:])
        )
        scalar = temperature.ndim == 0
        result = {}
        logger.debug(
            "Building CIE number densities for elements %s with output shape %s",
            ", ".join(validated_mass_fractions) or "none",
            temperature.shape,
        )

        for ion in self.AllSpecies:
            element = get_element_symbol(ion)
            element_fraction = validated_mass_fractions.get(element)
            if element_fraction is None:
                number_density = np.zeros(temperature.shape, dtype=float)
            else:
                element_density = density * element_fraction
                number_density = (
                    element_density
                    / const.ATOMIC_MASS[element]
                    * self._interpolate_cie_fraction(ion, temperature)
                )
            result[ion] = float(number_density) if scalar else number_density

        logger.info(
            "Built CIE number densities for %s ions",
            len(result),
        )
        return result
