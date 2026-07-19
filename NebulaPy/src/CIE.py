import numpy as np
from pathlib import Path
import os
from NebulaPy.src.Utils import getPionSymbol
from NebulaPy.src.LoggingConfig import NebulaError, get_logger
from NebulaPy.src.Progress import track

logger = get_logger(__name__)

class cieMode:

    def __init__(self, progress=False):
        self.progress = progress

        # get database
        database = os.environ.get("NEBULAPYDB")

        # Check if the database exists, exit if missing
        if database is None:
            raise NebulaError(
                "required database dir missing, install database to proceed"
            )

        self.cie_file = os.path.join(database, "IonBalance", "CIE.txt")

    ######################################################################################
    # Load CIE ion fraction table from the database directory
    ######################################################################################
    def loadCIEFile(self):

        cie_file = self.cie_file

        if not os.path.exists(cie_file):
            raise NebulaError(f"CIE file not found: {cie_file}")

        header = None
        data = []

        logger.info("Loading collision ionization equilibrium table")

        with open(cie_file, "r") as f:

            lines = f.readlines()

            iterator = track(
                lines,
                description="Importing CIE grid",
                unit="lines",
                enabled=self.progress,
            )

            for line in iterator:

                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if header is None:
                    header = parts
                    continue

                data.append([float(x) for x in parts])

        if header is None:
            raise NebulaError("CIE file header not found.")

        if not data:
            raise NebulaError("CIE file contains no data.")

        self.data = np.array(data, dtype=np.float64)

        # Keep the first column as log_T and convert ion columns to PION symbols
        self.col_index = {
            ("log_T" if i == 0 else getPionSymbol(name)): i
            for i, name in enumerate(header)
        }

        self.AllSpecies = np.array(
            [getPionSymbol(name) for name in header[1:]],
            dtype=str
        )

        logger.info(
            "CIE table loaded: %s temperature points, %s ion-fraction columns",
            self.data.shape[0],
            self.data.shape[1] - 1,
        )

    ######################################################################################
    # Interpolate ion fraction for a given ion and temperature(s)
    ######################################################################################
    def getCIEFraction(self, ion, Temperature):
        """
        Interpolate ion fraction for scalar or array Temperature (Kelvin).
        """
        if ion not in self.col_index:
            raise KeyError(f"Ion '{ion}' not found.")

        Temperature = np.asarray(Temperature, dtype=float)
        scalar = Temperature.ndim == 0
        Temperature = np.atleast_1d(Temperature)

        logT = np.log10(Temperature)

        Tgrid = self.data[:, self.col_index["log_T"]]
        fgrid = self.data[:, self.col_index[ion]]

        logT = np.clip(logT, Tgrid[0], Tgrid[-1])
        frac = np.interp(logT, Tgrid, fgrid)

        return frac[0] if scalar else frac

    ######################################################################################
    # Interpolate ion fraction for a given ion and temperature(s)
    ######################################################################################
    def buildCIENumberDensities(self, ElementMassFraction, Temperature):
        """
        Build a dictionary of ion number densities for all ions in the CIE table.
        ElementMassFraction: dict of element mass fractions (e.g. {'H': 0.7, 'He': 0.28, ...})
        Temperature: scalar or array of temperatures (Kelvin)
        Returns: dict of {ion: number density array}
        """

        pass
