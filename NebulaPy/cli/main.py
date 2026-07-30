"""Top-level NebulaPy command-line interface."""

from __future__ import annotations

import argparse

from NebulaPy.database.installer import main as install_database


def _parser():
    parser = argparse.ArgumentParser(prog="nebulapy")
    commands = parser.add_subparsers(dest="command", required=True)
    database = commands.add_parser(
        "database",
        help="manage the NebulaPy auxiliary database",
    )
    database_commands = database.add_subparsers(
        dest="database_command",
        required=True,
    )
    database_commands.add_parser(
        "install",
        add_help=False,
        help="download and install database components",
    )
    return parser


def main(argv=None):
    parser = _parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command == "database" and args.database_command == "install":
        return install_database(remaining)
    parser.error("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
