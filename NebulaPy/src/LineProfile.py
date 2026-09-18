"""Stateless helpers for shifted and broadened spectral-line profiles."""

from dataclasses import dataclass

import numpy as np

from NebulaPy.src.LoggingConfig import NebulaError


SPEED_OF_LIGHT_KM_S = 2.99792458e5
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class GlobalLineProfileConfig:
    """Immutable configuration for one snapshot's global line profile."""

    resolving_power: float
    global_velocity: float = 0.0
    global_velocity_sigma: float = 0.0
    relativistic: bool = False

    def __post_init__(self):
        resolving_power = float(self.resolving_power)
        global_velocity = float(self.global_velocity)
        global_velocity_sigma = float(self.global_velocity_sigma)

        if not np.isfinite(resolving_power) or resolving_power < 0.0:
            raise NebulaError(
                "resolving_power must be finite and non-negative."
            )
        if not np.isfinite(global_velocity):
            raise NebulaError("global_velocity must be finite.")
        if abs(global_velocity) >= SPEED_OF_LIGHT_KM_S:
            raise NebulaError(
                "global_velocity magnitude must be less than the speed of light."
            )
        if (
                not np.isfinite(global_velocity_sigma)
                or global_velocity_sigma < 0.0
        ):
            raise NebulaError(
                "global_velocity_sigma must be finite and non-negative."
            )
        if not isinstance(self.relativistic, (bool, np.bool_)):
            raise NebulaError("relativistic must be a boolean.")

        object.__setattr__(self, "resolving_power", resolving_power)
        object.__setattr__(self, "global_velocity", global_velocity)
        object.__setattr__(
            self,
            "global_velocity_sigma",
            global_velocity_sigma,
        )
        object.__setattr__(self, "relativistic", bool(self.relativistic))


class LineProfile:
    """Stateless engine for evaluating spectral-line profiles.

    Low-level methods expose the individual Gaussian, Doppler-shift, and width
    calculations.  High-level methods compose those operations for a specific
    physical scope, currently a snapshot-global profile.
    """

    def gaussian(self, wavelength, line_center, sigma):
        """Return a normalized Gaussian with a wavelength-space width."""
        wavelength = np.asarray(wavelength, dtype=np.float64)
        line_center = float(line_center)
        sigma = float(sigma)

        if not np.all(np.isfinite(wavelength)):
            raise NebulaError("wavelength must contain only finite values.")
        if not np.isfinite(line_center):
            raise NebulaError("line_center must be finite.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise NebulaError("sigma must be finite and positive.")

        offset = (wavelength - line_center) / sigma
        return (
            np.exp(-0.5 * offset**2)
            / (np.sqrt(2.0 * np.pi) * sigma)
        )

    def doppler_shift(
            self,
            line_wavelength,
            velocity,
            relativistic=False,
    ):
        """Return the non-relativistic or relativistic Doppler-shifted centre.

        ``velocity`` is in km/s; positive values produce a
        redshift and negative values produce a blueshift.  Set
        ``relativistic=True`` to use the longitudinal relativistic Doppler
        factor; otherwise the classical approximation is used.
        """
        line_wavelength = float(line_wavelength)
        velocity = float(velocity)

        if not np.isfinite(line_wavelength) or line_wavelength <= 0.0:
            raise NebulaError("line_wavelength must be finite and positive.")
        if not np.isfinite(velocity):
            raise NebulaError("global velocity must be finite.")
        if abs(velocity) >= SPEED_OF_LIGHT_KM_S:
            raise NebulaError(
                "global velocity magnitude must be less than the speed of light."
            )

        if not isinstance(relativistic, (bool, np.bool_)):
            raise NebulaError("relativistic must be a boolean.")

        beta = velocity / SPEED_OF_LIGHT_KM_S
        if relativistic:
            doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
        else:
            doppler_factor = 1.0 + beta

        return line_wavelength * doppler_factor

    def instrumental_sigma(self, line_wavelength, resolving_power):
        """Convert resolving power ``R = wavelength / FWHM`` to sigma."""
        line_wavelength = float(line_wavelength)
        resolving_power = float(resolving_power)

        if not np.isfinite(line_wavelength) or line_wavelength <= 0.0:
            raise NebulaError(
                "line_wavelength must be finite and positive."
            )
        if not np.isfinite(resolving_power) or resolving_power < 0.0:
            raise NebulaError(
                "resolving_power must be finite and non-negative."
            )
        if resolving_power == 0.0:
            return 0.0

        return line_wavelength / (resolving_power * FWHM_TO_SIGMA)

    def velocity_broadening_sigma(
            self,
            line_wavelength,
            velocity_sigma,
    ):
        """Convert a Gaussian velocity dispersion in km/s to sigma."""
        line_wavelength = float(line_wavelength)
        velocity_sigma = float(velocity_sigma)

        if not np.isfinite(line_wavelength) or line_wavelength <= 0.0:
            raise NebulaError(
                "line_wavelength must be finite and positive."
            )
        if not np.isfinite(velocity_sigma) or velocity_sigma < 0.0:
            raise NebulaError(
                "global_velocity_sigma must be finite and non-negative."
            )

        return line_wavelength * velocity_sigma / SPEED_OF_LIGHT_KM_S

    def global_line_profile(
            self,
            wavelength,
            line_wavelength,
            resolving_power,
            global_velocity,
            global_velocity_sigma,
            relativistic=False,
    ):
        """Return a globally shifted and Gaussian-broadened line profile.

        Instrumental and global velocity widths are independent Gaussian
        sigmas and are therefore combined in quadrature.
        """
        shifted_center = self.doppler_shift(
            line_wavelength=line_wavelength,
            velocity=global_velocity,
            relativistic=relativistic,
        )

        instrument_sigma = self.instrumental_sigma(
            line_wavelength=shifted_center,
            resolving_power=resolving_power,
        )
        velocity_sigma = self.velocity_broadening_sigma(
            line_wavelength=shifted_center,
            velocity_sigma=global_velocity_sigma,
        )
        total_sigma = np.hypot(instrument_sigma, velocity_sigma)
        if total_sigma == 0.0:
            raise NebulaError(
                "global line profile requires instrumental or velocity broadening."
            )

        return self.gaussian(
            wavelength=wavelength,
            line_center=shifted_center,
            sigma=total_sigma,
        )

__all__ = ["GlobalLineProfileConfig", "LineProfile"]
