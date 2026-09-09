"""Configuration objects for public pipeline entry points."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class FitConfig:
    """Numerical and workflow settings shared by all pipeline stages."""

    mode: str = "auto"
    input_format: str = "auto"
    vexp: float | None = None
    clip_outliers: bool = True
    vel_width: float = 4.0
    sil6355_red_initial: float = 0.0
    sil6355_blue_initial: float = -20.0
    sil5972_blue_initial: float = -20.0
    background_search_window_kms: float = 20_000.0
    background_search_step_kms: float = 1_000.0
    background_search_attempts: int = 5
    sil6355_red_bounds_kms: tuple[float, float] = (-15_000.0, 15_000.0)
    sil6355_blue_bounds_kms: tuple[float, float] = (-40_000.0, -15_000.0)
    sil5972_blue_bounds_kms: tuple[float, float] = (-30_000.0, -10_000.0)
    n_resol_neighbors: int = 3
    k_vel12_bounds: tuple[float, float] = (0.60, 1.30)
    k_fwhm12_bounds: tuple[float, float] = (0.60, 1.25)
    n_iterations: int = 100
    mc_mode: str = "resample"
    continuum_shift: float = 1.3
    seed: int | None = None

    def validate(self) -> None:
        def finite_number(name: str, value: object) -> None:
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")

        def integer_at_least(name: str, value: object, minimum: int) -> None:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")

        def ordered_bounds(name: str, values: object, *, positive: bool = False) -> None:
            try:
                lower, upper = values
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain two numeric bounds") from exc
            finite_number(f"{name} lower bound", lower)
            finite_number(f"{name} upper bound", upper)
            if lower >= upper:
                raise ValueError(f"{name} lower bound must be less than its upper bound")
            if positive and lower <= 0:
                raise ValueError(f"{name} bounds must be positive")
            if not positive and (lower <= -299_792.0 or upper >= 299_792.0):
                raise ValueError(f"{name} bounds must lie strictly between ±299792 km/s")

        if self.mode not in {"auto", "review", "manual"}:
            raise ValueError("mode must be 'auto', 'review', or 'manual'")
        if self.input_format not in {"auto", "flux", "flux-error"}:
            raise ValueError("input_format must be 'auto', 'flux', or 'flux-error'")
        positive_settings = {
            "vel_width": self.vel_width,
            "background_search_window_kms": self.background_search_window_kms,
            "background_search_step_kms": self.background_search_step_kms,
        }
        if self.vexp is not None:
            positive_settings["vexp"] = self.vexp
        for name, value in positive_settings.items():
            finite_number(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "sil6355_red_initial",
            "sil6355_blue_initial",
            "sil5972_blue_initial",
        ):
            value = getattr(self, name)
            finite_number(name, value)
            if abs(value) >= 299.792:
                raise ValueError(f"{name} must lie strictly between ±299.792 in 10³ km/s")
        for name in (
            "sil6355_red_bounds_kms",
            "sil6355_blue_bounds_kms",
            "sil5972_blue_bounds_kms",
        ):
            ordered_bounds(name, getattr(self, name))
        for name in ("k_vel12_bounds", "k_fwhm12_bounds"):
            ordered_bounds(name, getattr(self, name), positive=True)
        integer_at_least("background_search_attempts", self.background_search_attempts, 0)
        integer_at_least("n_resol_neighbors", self.n_resol_neighbors, 2)
        integer_at_least("n_iterations", self.n_iterations, 0)
        if self.seed is not None:
            integer_at_least("seed", self.seed, 0)
        if self.mc_mode not in {"resample", "shift", "shift_resample", "all"}:
            raise ValueError("invalid mc_mode")
        finite_number("continuum_shift", self.continuum_shift)
        if self.continuum_shift < 0:
            raise ValueError("continuum_shift cannot be negative")
