"""Consistent Rich live-progress displays for NebulaPy."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from multiprocessing import current_process
from typing import TypeVar

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
)
from rich.text import Text
from NebulaPy.src.LoggingConfig import CONSOLE, get_logger


T = TypeVar("T")
logger = get_logger(__name__)


class _CompactElapsedColumn(ProgressColumn):
    """Render elapsed time as MM:SS, expanding to HH:MM:SS when needed."""

    def render(self, task: Task) -> Text:
        elapsed = int(task.elapsed or 0)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        value = (
            f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            if hours
            else f"{minutes:02d}:{seconds:02d}"
        )
        return Text(f"elapsed {value}", style="progress.elapsed")


class _CompletedColumn(ProgressColumn):
    """Render completed and total counts with a meaningful unit."""

    def render(self, task: Task) -> Text:
        total = "?" if task.total is None else str(int(task.total))
        unit = task.fields.get("unit", "items")
        return Text(f"{int(task.completed)}/{total} {unit}", style="progress.download")


def create_progress(*, enabled: bool = True) -> Progress:
    """Create the standard NebulaPy live-progress display."""
    live_enabled = (
        enabled
        and current_process().name == "MainProcess"
        and (CONSOLE.is_terminal or CONSOLE.is_jupyter)
    )
    return Progress(
        TextColumn("{task.description}"),
        TaskProgressColumn(),
        BarColumn(bar_width=24),
        _CompletedColumn(),
        _CompactElapsedColumn(),
        refresh_per_second=10,
        transient=False,
        disable=not live_enabled,
        console=CONSOLE,
    )


def track(
    sequence: Iterable[T],
    *,
    description: str,
    total: int | None = None,
    unit: str = "items",
    enabled: bool = True,
) -> Iterator[T]:
    """Iterate over a sequence while displaying standard NebulaPy progress."""
    if total is None:
        try:
            total = len(sequence)  # type: ignore[arg-type]
        except TypeError:
            total = None

    with create_progress(enabled=enabled) as progress:
        task_id = progress.add_task(description, total=total, unit=unit)
        for item in sequence:
            yield item
            progress.advance(task_id)
        task = progress.tasks[task_id]
        completed = int(task.completed)
        elapsed = task.elapsed or 0.0

    logger.debug(
        "Progress completed: %s, %s/%s in %.2f seconds",
        description,
        completed,
        total if total is not None else "unknown",
        elapsed,
    )


class NebulaProgress:
    """Context manager for progress updated by queues or callback-driven work."""

    def __init__(
        self,
        description: str,
        total: int,
        *,
        unit: str = "tasks",
        enabled: bool = True,
    ):
        self.description = description
        self.total = total
        self.unit = unit
        self.progress = create_progress(enabled=enabled)
        self.task_id: int | None = None

    def __enter__(self) -> "NebulaProgress":
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.description,
            total=self.total,
            unit=self.unit,
        )
        return self

    def advance(self, amount: int = 1) -> None:
        if self.task_id is None:
            raise RuntimeError("Progress has not been started")
        self.progress.advance(self.task_id, amount)

    def update(self, completed: int) -> None:
        if self.task_id is None:
            raise RuntimeError("Progress has not been started")
        self.progress.update(self.task_id, completed=completed)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        progress_record = None
        if self.task_id is not None:
            task = self.progress.tasks[self.task_id]
            status = "completed" if task.completed >= self.total else "stopped"
            progress_record = (
                status,
                int(task.completed),
                task.elapsed or 0.0,
            )

        self.progress.stop()

        if progress_record is not None:
            status, completed, elapsed = progress_record
            logger.debug(
                "Progress %s: %s, %s/%s in %.2f seconds",
                status,
                self.description,
                completed,
                self.total,
                elapsed,
            )


_active_updates: dict[str, NebulaProgress] = {}


def update_progress(
    *,
    key: str,
    description: str,
    completed: int,
    total: int,
    unit: str = "items",
    enabled: bool = True,
) -> None:
    """Update progress from legacy callback-style loops."""
    if not enabled:
        return

    state = _active_updates.get(key)
    if state is None:
        state = NebulaProgress(description, total, unit=unit, enabled=True)
        state.__enter__()
        _active_updates[key] = state

    state.update(min(completed, total))

    if completed >= total:
        state.__exit__(None, None, None)
        _active_updates.pop(key, None)
