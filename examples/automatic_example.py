"""Run all bundled example spectra without interactive prompts."""

from pathlib import Path

from sii_fitter import FitConfig, run_catalog

ROOT = Path(__file__).resolve().parents[1]

run_catalog(
    ROOT / "example_catalog.csv",
    ROOT / "example_output" / "automatic",
    config=FitConfig(mode="auto", n_iterations=20, seed=42),
)
