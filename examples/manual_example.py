"""Open the four-click manual continuum selector for one example spectrum."""

from pathlib import Path

from sii_fitter import FitConfig, run_catalog

ROOT = Path(__file__).resolve().parents[1]

run_catalog(
    ROOT / "example_catalog.csv",
    ROOT / "example_output" / "manual",
    config=FitConfig(mode="manual", n_iterations=20, seed=42),
    start=0,
    end=1,
)
