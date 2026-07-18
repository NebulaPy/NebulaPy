"""Ion-symbol conversion helpers shared across NebulaPy."""

from __future__ import annotations

from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import NebulaError


__all__ = [
    "get_element_symbol",
    "get_spectroscopic_symbol",
    "getPionSymbol",
]


def get_element_symbol(ion):
    """Extract the element symbol from a PION ion identifier."""
    return "".join(filter(str.isalpha, ion))


def get_spectroscopic_symbol(ion):
    """Convert a PION ion identifier to spectroscopic notation."""
    normalized = ion.strip().replace("+", "")
    element = "".join(filter(str.isalpha, normalized)).capitalize()
    number = "".join(filter(str.isdigit, normalized))
    level = int(number) + 1 if number else 1

    if not 1 <= level <= len(const.ROMAN_ION_STAGES):
        raise NebulaError(
            f"Ionisation level {level} out of supported range "
            f"(1–{len(const.ROMAN_ION_STAGES)})"
        )

    return f"{element} {const.ROMAN_ION_STAGES[level - 1]}"


def getPionSymbol(chianti_symbol):
    """Convert a CHIANTI ion symbol such as fe_26 to a PION symbol."""
    normalized = chianti_symbol.lower()

    try:
        element, stage = normalized.split("_")
        stage = int(stage)
    except ValueError as exc:
        raise NebulaError(
            f"Invalid CHIANTI symbol '{normalized}'. Expected format like 'fe_25'."
        ) from exc

    if stage < 1:
        raise NebulaError(f"Invalid CHIANTI ion stage: {stage}")
    if stage == 1:
        return element.capitalize()

    return f"{element.capitalize()}{stage - 1}+"
