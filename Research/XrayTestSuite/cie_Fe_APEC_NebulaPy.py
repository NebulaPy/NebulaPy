"""Compare Fe CIE ion fractions from NebulaPy and AtomDB/APEC."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import NebulaPy.src as nebula
from NebulaPy.src.LoggingConfig import configure_logging, get_logger


logger = get_logger(__name__)


atomic_number = 26
element = "Fe"
minimum_temperature = 1.0e4  # K
maximum_temperature = 3.0e8  # K
number_of_temperatures = 500
minimum_plotted_fraction = 1.0e-2

atomdb_directory = Path("/Users/tony/Desktop/XrayTest/atomdb")
atomdb_ion_balance = atomdb_directory / "eigenfe_v3.1.0.fits"
nebulapy_database = Path(__file__).resolve().parents[2] / "nebulapy-db-1.0.0"
output_directory = Path("/Users/tony/Desktop/XrayTest/xray-validation")
output_plot = output_directory / "cie_Fe_APEC_NebulaPy.png"
output_table = output_directory / "cie_Fe_APEC_NebulaPy.txt"
output_peaks = output_directory / "cie_Fe_APEC_NebulaPy_peaks.txt"


def ion_name(charge):
    """Return a PION-style Fe ion name for an integer charge."""
    return element if charge == 0 else f"{element}{charge}+"


def calculate_nebulapy_fractions(temperature):
    """Interpolate every Fe charge-state fraction from NebulaPy's CIE table."""
    os.environ.setdefault("NEBULAPY_DB", str(nebulapy_database))
    cie = nebula.cieMode()
    cie.load_cie_file()
    return np.column_stack([
        cie.get_cie_fraction(ion_name(charge), temperature)
        for charge in range(atomic_number + 1)
    ])


def calculate_apec_fractions(temperature):
    """Calculate every Fe charge-state fraction with AtomDB v3.1.0."""
    if not atomdb_ion_balance.is_file():
        raise FileNotFoundError(
            "The AtomDB Fe ion-balance file is absent: "
            f"{atomdb_ion_balance}"
        )

    os.environ["ATOMDB"] = str(atomdb_directory)
    from astropy.io import fits
    from pyatomdb import apec

    # PyAtomDB returns columns indexed by charge: column 0 is Fe, column 26
    # is Fe26+. PyAtomDB 1.2.1 cannot broadcast its equilibrium interpolator
    # over a temperature vector, so evaluate each temperature with one open
    # v3.1.0 FITS file.
    with fits.open(atomdb_ion_balance) as ion_balance_data:
        fractions = [
            apec.return_ionbal(
                atomic_number,
                value,
                teunit="K",
                filename=ion_balance_data,
            )
            for value in temperature
        ]
    return np.asarray(fractions, dtype=np.float64)


def write_peak_table(temperature, nebulapy_fractions, apec_fractions):
    """Save peak temperature and peak fraction for every Fe charge state."""
    with output_peaks.open("w", encoding="utf-8") as stream:
        stream.write(
            "# Ion NebulaPy_Tpeak[K] NebulaPy_peak_fraction "
            "APEC_Tpeak[K] APEC_peak_fraction\n"
        )
        for charge in range(atomic_number + 1):
            nebulapy_peak = np.argmax(nebulapy_fractions[:, charge])
            apec_peak = np.argmax(apec_fractions[:, charge])
            stream.write(
                f"{ion_name(charge):<6} "
                f"{temperature[nebulapy_peak]:.10e} "
                f"{nebulapy_fractions[nebulapy_peak, charge]:.10e} "
                f"{temperature[apec_peak]:.10e} "
                f"{apec_fractions[apec_peak, charge]:.10e}\n"
            )


def main():
    """Calculate, validate, plot, and save the Fe CIE comparison."""
    configure_logging(level="DEBUG", log_to_file=True)
    temperature = np.logspace(
        np.log10(minimum_temperature),
        np.log10(maximum_temperature),
        number_of_temperatures,
    )

    logger.info(
        "NebulaPy RUN START: Fe CIE fractions, %.3e-%.3e K",
        minimum_temperature,
        maximum_temperature,
    )
    nebulapy_fractions = calculate_nebulapy_fractions(temperature)
    logger.info("NebulaPy RUN COMPLETE: %s temperatures", temperature.size)

    logger.info(
        "APEC RUN START: AtomDB v3.1.0 Fe CIE fractions, %.3e-%.3e K",
        minimum_temperature,
        maximum_temperature,
    )
    apec_fractions = calculate_apec_fractions(temperature)
    logger.info("APEC RUN COMPLETE: %s temperatures", temperature.size)

    expected_shape = (temperature.size, atomic_number + 1)
    for name, fractions in (
            ("NebulaPy", nebulapy_fractions),
            ("APEC", apec_fractions),
    ):
        if fractions.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected {name} shape {fractions.shape}; "
                f"expected {expected_shape}."
            )
        if not np.all(np.isfinite(fractions)):
            raise RuntimeError(f"{name} CIE fractions contain non-finite values.")

    nebulapy_sum_error = np.max(np.abs(nebulapy_fractions.sum(axis=1) - 1.0))
    apec_sum_error = np.max(np.abs(apec_fractions.sum(axis=1) - 1.0))
    logger.info(
        "NORMALIZATION CHECK: maximum |sum(f)-1|: NebulaPy=%.3e, APEC=%.3e",
        nebulapy_sum_error,
        apec_sum_error,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    columns = [temperature]
    header = ["Temperature[K]"]
    for charge in range(atomic_number + 1):
        columns.extend((
            nebulapy_fractions[:, charge],
            apec_fractions[:, charge],
        ))
        header.extend((
            f"NebulaPy_{ion_name(charge)}",
            f"APEC_{ion_name(charge)}",
        ))
    np.savetxt(
        output_table,
        np.column_stack(columns),
        header=" ".join(header),
        fmt="%.10e",
    )
    write_peak_table(temperature, nebulapy_fractions, apec_fractions)

    plotted_charges = [
        charge
        for charge in range(atomic_number + 1)
        if max(
            nebulapy_fractions[:, charge].max(),
            apec_fractions[:, charge].max(),
        ) >= minimum_plotted_fraction
    ]
    colours = plt.cm.turbo(np.linspace(0.0, 1.0, len(plotted_charges)))

    figure, axis = plt.subplots(figsize=(12, 7))
    for colour, charge in zip(colours, plotted_charges):
        label = ion_name(charge)
        axis.plot(
            temperature,
            nebulapy_fractions[:, charge],
            color=colour,
            linewidth=1.5,
            label=label,
        )
        axis.plot(
            temperature,
            apec_fractions[:, charge],
            color=colour,
            linewidth=1.2,
            linestyle="--",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(minimum_temperature, maximum_temperature)
    axis.set_ylim(1.0e-4, 1.05)
    axis.set_xlabel("Temperature [K]")
    axis.set_ylabel("CIE ion fraction")
    axis.grid(alpha=0.2)
    axis.legend(
        ncol=9,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
        mode="expand",
        borderaxespad=0.0,
    )
    axis.text(
        0.02,
        0.03,
        "Solid: NebulaPy CIE table\nDashed: AtomDB/APEC v3.1.0",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.92,
        },
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    figure.savefig(output_plot, bbox_inches="tight", dpi=300)
    plt.close(figure)

    print("AtomDB ionization balance: v3.1.0")
    print(f"Temperature samples: {temperature.size}")
    print(f"NebulaPy maximum normalization error: {nebulapy_sum_error:.6e}")
    print(f"APEC maximum normalization error: {apec_sum_error:.6e}")
    print(f"Saved CIE table: {output_table}")
    print(f"Saved peak table: {output_peaks}")
    print(f"Saved comparison plot: {output_plot}")


if __name__ == "__main__":
    main()
