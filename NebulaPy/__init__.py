"""NebulaPy public package metadata and startup banner."""

from multiprocessing import current_process
import os
import sys

from . import version

__version_info__ = version.__version_info__
__version__ = version.__version__


def _display_startup_banner() -> None:
    """Display the package identity once in an interactive main process."""
    banner_enabled = os.environ.get("NEBULAPY_NO_BANNER", "").lower() not in {
        "1",
        "true",
        "yes",
    }
    if (
        banner_enabled
        and current_process().name == "MainProcess"
        and sys.stderr.isatty()
    ):
        print(f"NebulaPy {__version__}", file=sys.stderr, flush=True)


_display_startup_banner()
