"""Read and plot the saved WR 140 NEI X-ray spectrum."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks


MINIMUM_PEAK_TO_CONTINUUM_RATIO = 1.05
MAXIMUM_PEAK_SEPARATION = 0.03  # Angstrom


def main():
    spectrum_file = Path(
        "/Users/tony/Desktop/CWBs-NEMOv1/Post-Processing/WR140Test/"
        "wr140_NEMO_d07e13_d2l6n128_NIE_ 1e6to1e9(4000).txt"
    )
    output_png = spectrum_file.with_name(
        f"{spectrum_file.stem}_identified_lines_publication.png"
    )
    output_pdf = spectrum_file.with_name(
        f"{spectrum_file.stem}_identified_lines_publication.pdf"
    )
    continuum_file = spectrum_file.with_name(
        f"{spectrum_file.stem}_approximate_continuum.txt"
    )
    line_list_file = spectrum_file.parent / "Soft&HardXrayList.txt"
    identified_line_file = spectrum_file.with_name(
        f"{spectrum_file.stem}_identified_high_luminosity_lines.txt"
    )

    wavelength, spectrum = np.loadtxt(spectrum_file, unpack=True)

    kev_angstrom = 12.39841984
    kev_to_erg = 1.602176634e-9
    photon_energy = kev_angstrom / wavelength
    energy_luminosity = spectrum * kev_angstrom / photon_energy**2
    photon_spectrum = energy_luminosity / (photon_energy * kev_to_erg)

    # Estimate the continuum with an edge-aware asymmetric smooth baseline.
    positive_floor = np.min(photon_spectrum[photon_spectrum > 0.0])
    log_spectrum = np.log10(np.maximum(photon_spectrum, positive_floor))
    number_of_points = log_spectrum.size
    second_difference = sparse.diags(
        (
            np.ones(number_of_points - 2),
            -2.0 * np.ones(number_of_points - 2),
            np.ones(number_of_points - 2),
        ),
        (0, 1, 2),
        shape=(number_of_points - 2, number_of_points),
        format="csc",
    )
    smoothness = 1.0e4
    line_weight = 0.01
    baseline = log_spectrum.copy()
    for _ in range(15):
        weights = np.where(
            log_spectrum > baseline,
            line_weight,
            1.0 - line_weight,
        )
        weight_matrix = sparse.diags(weights, format="csc")
        baseline = spsolve(
            weight_matrix
            + smoothness * (second_difference.T @ second_difference),
            weights * log_spectrum,
        )
    approximate_continuum = np.minimum(10.0**baseline, photon_spectrum)

    # Find candidate local maxima with positive emission above the continuum.
    continuum_excess = photon_spectrum - approximate_continuum
    peak_indices, _ = find_peaks(continuum_excess, height=0.0)

    # Read the physical emission lines from Soft&HardXrayList.
    line_pattern = re.compile(
        r"^([A-Z][a-z]?)\s+([IVXLCDM]+)\s+"
        r"([0-9.]+)\s+([0-9.eE+\-]+)\s+([0-9.]+)$"
    )
    catalogue_lines = []
    with line_list_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = line_pattern.match(line.strip())
            if match is None:
                continue
            line_name = f"{match.group(1)} {match.group(2)}"
            line_wavelength = float(match.group(3))
            line_luminosity = float(match.group(4))
            line_energy = float(match.group(5))
            if wavelength[0] <= line_wavelength <= wavelength[-1]:
                catalogue_lines.append((
                    line_name,
                    line_wavelength,
                    line_luminosity,
                    line_energy,
                ))

    # Retain visible peaks above the configured local-continuum ratio.
    visible_peak_indices = peak_indices[
        photon_spectrum[peak_indices]
        / approximate_continuum[peak_indices]
        >= MINIMUM_PEAK_TO_CONTINUUM_RATIO
    ]

    # Associate every visible peak with nearby physical catalogue lines.
    identified_lines = []
    for peak_index in visible_peak_indices:
        for (
            line_name,
            line_wavelength,
            line_luminosity,
            line_energy,
        ) in catalogue_lines:
            separation = abs(wavelength[peak_index] - line_wavelength)
            if separation > MAXIMUM_PEAK_SEPARATION:
                continue
            identified_lines.append((
                line_name,
                line_wavelength,
                line_energy,
                line_luminosity,
                peak_index,
                separation,
            ))

    np.savetxt(
        continuum_file,
        np.column_stack((wavelength, approximate_continuum)),
        header=(
            "Wavelength[A] ApproximateContinuum "
            "[photons s^-1 sr^-1 keV^-1]"
        ),
        fmt="%.8e",
    )

    # Save the physical line-to-spectrum peak associations.
    np.savetxt(
        identified_line_file,
        np.asarray([
            (
                line_name,
                line_wavelength,
                line_energy,
                line_luminosity,
                wavelength[peak_index],
                photon_spectrum[peak_index],
                separation,
            )
            for (
                line_name,
                line_wavelength,
                line_energy,
                line_luminosity,
                peak_index,
                separation,
            ) in identified_lines
        ], dtype=object),
        header=(
            "Ion LineWavelength[A] LineEnergy[keV] LineLuminosity "
            "MatchedPeakWavelength[A] PeakPhotonSpectrum "
            "Separation[A]"
        ),
        fmt=("%s", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e"),
    )

    # Apply compact double-column journal typography and vector-safe fonts.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })

    # Select the most luminous catalogue line for each unresolved peak blend.
    dominant_line_by_peak = {}
    for (
        line_name,
        line_wavelength,
        line_energy,
        line_luminosity,
        peak_index,
        separation,
    ) in identified_lines:
        previous_line = dominant_line_by_peak.get(peak_index)
        if previous_line is None or line_luminosity > previous_line[2]:
            dominant_line_by_peak[peak_index] = (
                line_name,
                line_energy,
                line_luminosity,
            )

    # Split the crowded wavelength range into three journal-width panels.
    panel_ranges = ((1.0, 8.1), (8.0, 14.0), (14.0, 20.0))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6))

    for axis, (minimum_wavelength, maximum_wavelength) in zip(
        axes,
        panel_ranges,
    ):
        panel_mask = (
            (wavelength >= minimum_wavelength)
            & (wavelength <= maximum_wavelength)
        )
        axis.plot(
            wavelength[panel_mask],
            photon_spectrum[panel_mask],
            color="#0072B2",
            linewidth=0.75,
            solid_capstyle="round",
            rasterized=False,
        )

        # Place every line ID in the panel containing its matched peak.
        for peak_index, (
            line_name,
            line_energy,
            line_luminosity,
        ) in dominant_line_by_peak.items():
            if not (
                minimum_wavelength
                <= wavelength[peak_index]
                <= maximum_wavelength
            ):
                continue
            label_artist = axis.annotate(
                f"{line_name} {line_energy:.4f} keV",
                (wavelength[peak_index], photon_spectrum[peak_index]),
                xytext=(0, 4),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=6.0,
                color="black",
                clip_on=True,
            )
            label_artist.set_clip_path(axis.patch)

        # Scale each panel to its own spectral dynamic range and label its band.
        panel_values = photon_spectrum[panel_mask]
        axis.set_yscale("log")
        axis.set_xlim(minimum_wavelength, maximum_wavelength)
        axis.set_ylim(
            bottom=np.min(panel_values[panel_values > 0.0]) * 0.7,
            top=np.max(panel_values) * 30.0,
        )
        axis.text(
            0.015,
            0.94,
            rf"{minimum_wavelength:.0f}--{maximum_wavelength:.0f} $\AA$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )

        # Use inward ticks on every side and publication-weight spines.
        axis.minorticks_on()
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
            width=0.8,
        )
        axis.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=2.5,
            width=0.6,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    # Apply shared physical-axis labels to the complete multi-panel figure.
    axes[-1].set_xlabel(r"Wavelength ($\AA$)")
    fig.supylabel(
        r"$dN_E/d\Omega$ "
        r"(photons s$^{-1}$ sr$^{-1}$ keV$^{-1}$)",
        x=0.07,
        fontsize=10,
    )

    # Preserve consistent margins and panel spacing for both export formats.
    fig.subplots_adjust(
        left=0.13,
        right=0.99,
        bottom=0.07,
        top=0.99,
        hspace=0.08,
    )

    # Save a high-resolution raster image and an editable vector PDF.
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
