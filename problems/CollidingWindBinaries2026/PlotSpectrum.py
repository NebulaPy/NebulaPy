import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input and output settings
# ============================================================

SpectrumFile = (
    "/home/tony/Desktop/CWBs-2026/Postprocessing/X-raySpectrum/WR140/"
    "wr140_NEMO_d07e13_d2l6n128_Spectrum_NoFB_1.244585e+06.txt"
)

OutDir = (
    "/home/tony/Desktop/CWBs-2026/Postprocessing/X-raySpectrum/WR140/"
    "XraySpectrum/WR140"
)

Filename = "WR140_Xray_Photon_Flux"


# ============================================================
# Physical constants
# ============================================================

h = 6.62607015e-27          # Planck constant [erg s]
c = 2.99792458e10           # Speed of light [cm s^-1]
pc_to_cm = 3.085677581e18   # Parsec to cm
angstrom_to_cm = 1.0e-8     # Angstrom to cm


# ============================================================
# Distance to WR 140
# ============================================================

distance_pc = 1518.0
distance_cm = distance_pc * pc_to_cm


# ============================================================
# Read spectrum
#
# Expected input columns:
# wavelength [Angstrom]
# L_lambda   [erg s^-1 Angstrom^-1]
# ============================================================

wavelength, L_lambda = np.loadtxt(
    SpectrumFile,
    unpack=True
)


# ============================================================
# Remove invalid data
# ============================================================

valid = (
    np.isfinite(wavelength)
    & np.isfinite(L_lambda)
    & (wavelength > 0.0)
)

wavelength = wavelength[valid]
L_lambda = L_lambda[valid]

if wavelength.size < 2:
    raise ValueError(
        "The input spectrum must contain at least two valid wavelength points."
    )


# Sort by increasing wavelength
sort_index = np.argsort(wavelength)

wavelength = wavelength[sort_index]
L_lambda = L_lambda[sort_index]


# ============================================================
# Convert luminosity spectrum to energy flux spectrum
#
# F_lambda = L_lambda / (4 pi d^2)
#
# Unit:
# erg cm^-2 s^-1 Angstrom^-1
# ============================================================

flux_lambda = (
    L_lambda
    / (4.0 * np.pi * distance_cm**2)
)


# ============================================================
# Convert energy flux to photon flux
#
# Photon energy:
#
# E_photon = h c / lambda
#
# Photon flux:
#
# N_lambda = F_lambda / E_photon
#          = F_lambda lambda / (h c)
#
# Since wavelength is supplied in Angstrom:
#
# lambda_cm = wavelength * 1e-8
#
# Unit:
# photons cm^-2 s^-1 Angstrom^-1
# ============================================================

wavelength_cm = wavelength * angstrom_to_cm

photon_flux = (
    flux_lambda
    * wavelength_cm
    / (h * c)
)


# ============================================================
# Integrated quantities over the wavelength range
# ============================================================

integrated_energy_flux = np.trapz(
    flux_lambda,
    wavelength
)

integrated_photon_flux = np.trapz(
    photon_flux,
    wavelength
)

integrated_luminosity = np.trapz(
    L_lambda,
    wavelength
)


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    OutDir,
    exist_ok=True
)


# ============================================================
# Save calculated spectrum
# ============================================================

OutputSpectrumFile = os.path.join(
    OutDir,
    f"{Filename}.txt"
)

output_data = np.column_stack(
    (
        wavelength,
        L_lambda,
        flux_lambda,
        photon_flux
    )
)

header = (
    "Wavelength[A] "
    "L_lambda[erg_s^-1_A^-1] "
    "F_lambda[erg_cm^-2_s^-1_A^-1] "
    "Photon_flux[photons_cm^-2_s^-1_A^-1]\n"
    f"Distance = {distance_pc:.2f} pc\n"
    f"Integrated luminosity = "
    f"{integrated_luminosity:.8e} erg s^-1\n"
    f"Integrated energy flux = "
    f"{integrated_energy_flux:.8e} erg cm^-2 s^-1\n"
    f"Integrated photon flux = "
    f"{integrated_photon_flux:.8e} photons cm^-2 s^-1"
)

np.savetxt(
    OutputSpectrumFile,
    output_data,
    fmt="%.10e",
    header=header
)


# ============================================================
# Plot photon flux spectrum
# ============================================================

positive = (
    np.isfinite(photon_flux)
    & (photon_flux > 0.0)
)

if not np.any(positive):
    raise ValueError(
        "The calculated photon flux contains no positive values, "
        "so it cannot be plotted on a logarithmic y-axis."
    )


fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    wavelength[positive],
    photon_flux[positive],
    color="green",
    linewidth=1.4,
    label="WR 140 photon flux"
)

ax.set_xlabel(
    r"Wavelength [$\AA$]",
    fontsize=12
)

ax.set_ylabel(
    r"$N_\lambda$ "
    r"[photons cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]",
    fontsize=12
)

ax.set_yscale("log")


# ============================================================
# Set y-axis limits
# ============================================================

positive_photon_flux = photon_flux[positive]

ymin = np.min(positive_photon_flux)
ymax = np.max(positive_photon_flux)

ax.set_ylim(
    ymin * 0.8,
    ymax * 2.0
)


# ============================================================
# Tick formatting
# ============================================================

ax.minorticks_on()

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=6,
    width=1.2,
    labelsize=11
)

ax.tick_params(
    axis="both",
    which="minor",
    direction="in",
    top=True,
    right=True,
    length=3,
    width=1.0
)


# ============================================================
# Spine formatting
# ============================================================

for spine in ax.spines.values():
    spine.set_linewidth(1.2)


ax.legend(
    loc="best",
    frameon=False,
    fontsize=10
)

fig.tight_layout()


# ============================================================
# Save plot
# ============================================================

PlotFile = os.path.join(
    OutDir,
    f"{Filename}.png"
)

fig.savefig(
    PlotFile,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Print summary
# ============================================================

print(f"Read luminosity spectrum from: {SpectrumFile}")
print(f"Distance to WR 140: {distance_pc:.2f} pc")
print(f"Distance in cm: {distance_cm:.8e} cm")
print(f"Number of wavelength points: {wavelength.size}")

print(
    f"Wavelength range: "
    f"{wavelength.min():.6e} - "
    f"{wavelength.max():.6e} Angstrom"
)

print(
    f"Integrated luminosity: "
    f"{integrated_luminosity:.8e} erg s^-1"
)

print(
    f"Integrated energy flux: "
    f"{integrated_energy_flux:.8e} erg cm^-2 s^-1"
)

print(
    f"Integrated photon flux: "
    f"{integrated_photon_flux:.8e} photons cm^-2 s^-1"
)

print(f"Saved calculated spectrum to: {OutputSpectrumFile}")
print(f"Saved photon flux plot to: {PlotFile}")