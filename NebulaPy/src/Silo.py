"""Discovery and time-based batching of PION silo files."""

from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
import re

import astropy.units as unit
from pypion.ReadData import ReadData
from pypion.SiloHeader_data import OpenData

from NebulaPy.src.LoggingConfig import NebulaError, get_logger


logger = get_logger(__name__)


class Silo:
    """Organize uniform or nested-grid silo files into simulation snapshots."""

    _TIME_UNITS = {
        "s": unit.s,
        "sec": unit.s,
        "second": unit.s,
        "seconds": unit.s,
        "min": unit.min,
        "minute": unit.min,
        "minutes": unit.min,
        "h": unit.h,
        "hr": unit.h,
        "hour": unit.h,
        "hours": unit.h,
        "d": unit.day,
        "day": unit.day,
        "days": unit.day,
        "yr": unit.yr,
        "year": unit.yr,
        "years": unit.yr,
        "kyr": unit.kyr,
        "myr": unit.Myr,
        "gyr": unit.Gyr,
    }
    _NESTED_NAME = re.compile(
        r"_level(?P<level>\d{2})_(?P<instant>.+)\.silo$"
    )

    @classmethod
    def _time_unit(cls, time_unit):
        """Return an Astropy time unit and its preferred display label."""
        if time_unit is None:
            return unit.s, "sec"

        key = str(time_unit).strip().lower()
        selected_unit = cls._TIME_UNITS.get(key)
        if selected_unit is None:
            supported = "sec, min, hour, day, yr, kyr, Myr, Gyr"
            raise NebulaError(
                f"Unsupported time unit '{time_unit}'. Supported units: {supported}"
            )

        display_labels = {"myr": "Myr", "gyr": "Gyr"}
        return selected_unit, display_labels.get(key, key)

    @staticmethod
    def _validate_inputs(directory, filebase, start_time, finish_time, out_frequency):
        if not isinstance(directory, (str, Path)):
            raise NebulaError("directory must be a path-like string")
        if not isinstance(filebase, str) or not filebase.strip():
            raise NebulaError("filebase must be a non-empty string")

        for name, value in (("start_time", start_time), ("finish_time", finish_time)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real)
            ):
                raise NebulaError(f"{name} must be a real number or None")

        if (
            start_time is not None
            and finish_time is not None
            and start_time > finish_time
        ):
            raise NebulaError(
                "start_time must be smaller than or equal to finish_time"
            )

        if out_frequency is not None and (
            isinstance(out_frequency, bool)
            or not isinstance(out_frequency, Integral)
            or out_frequency <= 0
        ):
            raise NebulaError(
                "out_frequency must be a positive integer or None"
            )

    @staticmethod
    def _simulation_time(silo_set, coordinate_system):
        """Read one snapshot time in seconds."""
        data = ReadData([str(path) for path in silo_set])
        try:
            readers = {
                1: data.get_3Darray,
                2: data.get_2Darray,
                3: data.get_1Darray,
            }
            reader = readers.get(coordinate_system)
            if reader is None:
                raise NebulaError(
                    f"Unsupported coordinate-system identifier: {coordinate_system}"
                )
            return float((reader("Density")["sim_time"] * unit.s).value)
        finally:
            data.close()

    @staticmethod
    def _header_details(all_silos):
        """Read grid metadata once from the silo collection."""
        header = OpenData([str(path) for path in all_silos])
        try:
            header.db.SetDir("/header")
            coordinate_system = int(header.db.GetVar("coord_sys"))
            level_count = int(header.db.GetVar("grid_nlevels"))
        finally:
            header.close()

        if level_count < 1:
            raise NebulaError(
                f"Invalid number of grid levels in silo header: {level_count}"
            )
        if coordinate_system not in {1, 2, 3}:
            raise NebulaError(
                f"Unsupported coordinate-system identifier: {coordinate_system}"
            )
        return coordinate_system, level_count

    @classmethod
    def _nested_index(cls, all_silos):
        """Index nested files by simulation instant and grid level."""
        index = {}
        for path in all_silos:
            match = cls._NESTED_NAME.search(path.name)
            if match is None:
                continue
            key = (match.group("instant"), int(match.group("level")))
            if key in index:
                raise NebulaError(
                    f"Duplicate silo file for instant {key[0]}, level {key[1]}"
                )
            index[key] = path
        return index

    @staticmethod
    def _sample_snapshots(snapshots, out_frequency):
        """Apply output frequency while retaining the first and final snapshots."""
        if out_frequency is None or len(snapshots) <= 1:
            return snapshots

        indices = set(range(0, len(snapshots), out_frequency))
        indices.add(len(snapshots) - 1)
        return [snapshots[index] for index in sorted(indices)]

    @classmethod
    def batch(
        cls,
        directory,
        filebase,
        start_time=None,
        finish_time=None,
        time_unit=None,
        out_frequency=None,
    ):
        """Return silo files grouped by time and nested-grid level.

        Supported time scales include seconds, minutes, hours, days, years,
        kiloyears (kyr), megayears (Myr), and gigayears (Gyr).
        """
        cls._validate_inputs(
            directory,
            filebase,
            start_time,
            finish_time,
            out_frequency,
        )
        selected_unit, display_unit = cls._time_unit(time_unit)
        seconds_per_unit = selected_unit.to(unit.s)
        start_seconds = (
            start_time * seconds_per_unit if start_time is not None else None
        )
        finish_seconds = (
            finish_time * seconds_per_unit if finish_time is not None else None
        )

        root = Path(directory).expanduser()
        if not root.is_dir():
            raise NebulaError(f"Silo directory does not exist: {root}")

        filename_prefix = f"{filebase}_"
        all_silos = sorted(
            path
            for path in root.iterdir()
            if path.is_file()
            and path.suffix == ".silo"
            and path.name.startswith(filename_prefix)
        )
        if not all_silos:
            raise NebulaError(
                f"No '{filebase}' silo files found in '{root}'"
            )

        coordinate_system, level_count = cls._header_details(all_silos)
        logger.info("Batching silo files into time instances")
        logger.info(
            "Grid: %s",
            "uniform" if level_count == 1 else f"nested with {level_count} levels",
        )
        logger.debug(
            "Batch selection: directory=%s, filebase=%s, files=%s, start=%s %s, "
            "finish=%s %s, frequency=%s",
            root,
            filebase,
            len(all_silos),
            start_time,
            display_unit,
            finish_time,
            display_unit,
            out_frequency,
        )

        nested_index = cls._nested_index(all_silos) if level_count > 1 else None
        if level_count == 1:
            candidates = [([path], path.name) for path in all_silos]
        else:
            level_zero = sorted(
                (instant, path)
                for (instant, level), path in nested_index.items()
                if level == 0
            )
            if not level_zero:
                raise NebulaError(
                    "No level-00 silo files found for the nested grid"
                )
            candidates = [([path], instant) for instant, path in level_zero]

        timed_snapshots = [
            (cls._simulation_time(snapshot, coordinate_system), snapshot, instant)
            for snapshot, instant in candidates
        ]
        timed_snapshots.sort(key=lambda item: item[0])
        simulation_start = timed_snapshots[0][0]
        simulation_finish = timed_snapshots[-1][0]

        if finish_seconds is not None and finish_seconds > simulation_finish:
            logger.warning(
                f"Specified finish time {float(finish_time):.3f} {display_unit} "
                f"exceeds the simulation walltime "
                f"{simulation_finish / seconds_per_unit:.3f} {display_unit}"
            )

        selected = [
            item
            for item in timed_snapshots
            if (start_seconds is None or item[0] >= start_seconds)
            and (finish_seconds is None or item[0] <= finish_seconds)
        ]
        if not selected:
            raise NebulaError(
                "No silo files found in the specified time range; "
                "check the selection criteria"
            )

        selected = cls._sample_snapshots(selected, out_frequency)

        snapshots = []
        for _, snapshot, instant in selected:
            if level_count > 1:
                snapshot = []
                for level in range(level_count):
                    level_file = nested_index.get((instant, level))
                    if level_file is None:
                        raise NebulaError(
                            f"Missing silo file for level {level} instant {instant}"
                        )
                    snapshot.append(level_file)
            snapshots.append([str(path) for path in snapshot])

        selected_start = selected[0][0] / seconds_per_unit
        selected_finish = selected[-1][0] / seconds_per_unit
        logger.info(
            "Batched %s time instances spanning %.3f–%.3f %s",
            len(snapshots),
            selected_start,
            selected_finish,
            display_unit,
        )
        logger.debug(
            "Simulation coverage: %.3f–%.3f %s",
            simulation_start / seconds_per_unit,
            simulation_finish / seconds_per_unit,
            display_unit,
        )
        return snapshots
