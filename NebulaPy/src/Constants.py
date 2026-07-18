"""Constants and reference data used by NebulaPy.

Physical constants are expressed in centimetre-gram-second (CGS) units unless
the name explicitly identifies a conversion factor. Domain tables contain only
the elements and stellar-atmosphere grids currently supported by NebulaPy.
"""

# Fundamental physical constants (CGS)
PI = 3.1415926535897931
PLANCK_CONSTANT = 6.6260693e-27  # erg s
SPEED_OF_LIGHT = 2.99792458e10  # cm s^-1
BOLTZMANN_CONSTANT = 1.3806504e-16  # erg K^-1
FINE_STRUCTURE_CONSTANT = 7.2973525376e-3
ELECTRON_MASS = 9.10938215e-28  # g
STEFAN_BOLTZMANN_CONSTANT = 5.670373e-5  # erg cm^-2 s^-1 K^-4

# Energy and wavelength conversion factors
EV_TO_ERG = 1.602176487e-12
EV_ANGSTROM = 1.239841875e4  # wavelength [Å] = EV_ANGSTROM / energy [eV]
KEV_ANGSTROM = 12.39841875  # wavelength [Å] = KEV_ANGSTROM / energy [keV]
ANGSTROM_TO_CM = 1.0e-8
CM_TO_ANGSTROM = 1.0e8
PETAHERTZ_TO_HERTZ = 1.0e15
JANSKY_TO_CGS_FLUX_DENSITY = 1.0e-23  # erg s^-1 cm^-2 Hz^-1

# Shared numerical safeguards
ELECTRON_DENSITY_FLOOR = 1.0e-8  # cm^-3

# Astronomical reference values
SOLAR_RADIUS = 6.955e10  # cm
PARSEC = 3.08568025e18  # cm

# Element symbols represented by the NEMO/PION chemistry implementation
SUPPORTED_ELEMENTS = ("H", "He", "C", "N", "O", "Ne", "Si", "S", "Fe")

# Atomic masses in grams
ATOMIC_MASS = {
    "H": 1.6738e-24,
    "He": 6.6464768e-24,
    "C": 1.994374e-23,
    "N": 2.325892e-23,
    "O": 2.6567628e-23,
    "Ne": 3.3509177e-23,
    "Si": 4.6637066e-23,
    "S": 5.3245181e-23,
    "Fe": 9.2732796e-23,
}

ATOMIC_NUMBER = {
    "H": 1,
    "He": 2,
    "C": 6,
    "N": 7,
    "O": 8,
    "Ne": 10,
    "Si": 14,
    "S": 16,
    "Fe": 26,
}

# Highest ionization stage represented for every supported element
FULLY_IONIZED_IONS = frozenset(
    f"{element}{charge}+" for element, charge in ATOMIC_NUMBER.items()
)

# Coordinate-system identifiers stored in PION silo headers
COORDINATE_SYSTEMS = {
    1: "cartesian",
    2: "cylindrical",
    3: "spherical",
}

# Spectroscopic ion stages supported by the element set (neutral through +26)
ROMAN_ION_STAGES = (
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII",
    "XIX", "XX", "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI",
    "XXVII",
)

# CHIANTI upper-level indices for helium-like two-photon transitions, by Z
HE_LIKE_TWO_PHOTON_UPPER_LEVELS = (
    -1, 3, -1, -1, -1, 3, 6, 6, -1, 6, 6, 6, 6, 6, 3,
    5, 3, 5, 3, 5, -1, 5, -1, 5, -1, 5, -1, 5, -1, 5,
)

# PoWR grid clumping factors, indexed by the model-grid identifier
POWR_REFERENCE_VELOCITY_KM_S = 2500.0
POWR_MASS_LOSS_SCALE = 1.0e4
POWR_FLUX_DISTANCE_PC = 10.0
CMFGEN_FLUX_DISTANCE_PC = 1000.0

POWR_CLUMPING_FACTORS = {
    "mw-wne": 4,
    "mw-wnl-h20": 4,
    "mw-wnl-h50": 4,
    "mw-wc": 10,
    "lmc-wne": 10,
    "lmc-wnl-h20": 10,
    "lmc-wnl-h40": 10,
    "lmc-wc": 10,
    "smc-wne": 4,
    "smc-wnl-h20": 4,
    "smc-wnl-h40": 4,
    "smc-wnl-h60": 4,
    "smc-wc": 10,
    "z007-wne": 10,
    "z007-wnl-h20": 10,
    "z007-wnl-h40": 10,
    "z007-wnl-h60": 10,
    "z007-wc": 10,
    "z086-wo": 0.4,
    "mw-ob-i": 10,
    "lmc-ob-i": 10,
    "smc-ob-vd3": 10,
    "smc-ob-i": 10,
    "smc-ob-ii": 10,
    "smc-ob-iii": 10,
}

# Log10 axes of the bundled CHIANTI cooling tables
COOLING_LOG_TEMPERATURE_GRID = tuple(1.0 + 0.1 * index for index in range(81))
COOLING_LOG_ELECTRON_DENSITY_GRID = tuple(0.5 * index for index in range(13))

# Valid temperature interval for the bundled PyNeb H I Case-B data (K)
PYNEB_H_RECOMBINATION_TEMPERATURE_RANGE = (500.0, 30_000.0)

# Effective-temperature grid used to generate blackbody SED models (K)
BLACKBODY_TEMPERATURE_GRID = tuple(
    float(temperature)
    for temperature in (
        *range(3500, 13250, 250),
        *range(14000, 50000, 1000),
    )
)


__all__ = [
    "ANGSTROM_TO_CM",
    "ATOMIC_MASS",
    "ATOMIC_NUMBER",
    "BLACKBODY_TEMPERATURE_GRID",
    "BOLTZMANN_CONSTANT",
    "CM_TO_ANGSTROM",
    "CMFGEN_FLUX_DISTANCE_PC",
    "COOLING_LOG_ELECTRON_DENSITY_GRID",
    "COOLING_LOG_TEMPERATURE_GRID",
    "COORDINATE_SYSTEMS",
    "ELECTRON_DENSITY_FLOOR",
    "ELECTRON_MASS",
    "EV_ANGSTROM",
    "EV_TO_ERG",
    "FINE_STRUCTURE_CONSTANT",
    "FULLY_IONIZED_IONS",
    "HE_LIKE_TWO_PHOTON_UPPER_LEVELS",
    "JANSKY_TO_CGS_FLUX_DENSITY",
    "KEV_ANGSTROM",
    "PARSEC",
    "PI",
    "PLANCK_CONSTANT",
    "PETAHERTZ_TO_HERTZ",
    "POWR_CLUMPING_FACTORS",
    "POWR_FLUX_DISTANCE_PC",
    "POWR_MASS_LOSS_SCALE",
    "POWR_REFERENCE_VELOCITY_KM_S",
    "PYNEB_H_RECOMBINATION_TEMPERATURE_RANGE",
    "SOLAR_RADIUS",
    "SPEED_OF_LIGHT",
    "STEFAN_BOLTZMANN_CONSTANT",
    "SUPPORTED_ELEMENTS",
    "ROMAN_ION_STAGES",
]
