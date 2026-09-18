"""Generate a PoWR atmosphere SED table for use with PION."""

from argparse import ArgumentParser
from pathlib import Path

import NebulaPy.src as nebula


ENERGY_BINS_EV = [
    [7.902470, 11.260300], [11.260300, 13.598400],
    [13.598400, 14.534100], [14.534100, 16.199200],
    [16.199200, 21.564500], [21.564500, 24.383100],
    [24.383100, 29.601300], [29.601300, 30.651000],
    [30.651000, 35.121100], [35.121100, 40.963000],
    [40.963000, 45.141800], [45.141800, 47.887800],
    [47.887800, 54.417800], [54.417800, 63.423300],
    [63.423300, 77.000000],
]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="directory for plots and PION-formatted tables",
    )
    parser.add_argument("--metallicity", default="SMC")
    parser.add_argument("--composition", default="WNL-H20")
    parser.add_argument("--mdot", type=float, default=-5.0)
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    powr_sed = nebula.sed(
        energy_bins=ENERGY_BINS_EV,
        progress=True,
        plot=str(output_dir),
        pion=str(output_dir),
    )
    powr_sed.PotsdamWolfRayet(
        args.metallicity,
        args.composition,
        args.mdot,
    )


if __name__ == "__main__":
    main()
