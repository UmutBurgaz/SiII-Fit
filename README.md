# Si II Fitter

`sii-fitter` measures the Si II λ6355 and λ5972 features in supernova spectra. It supports a fully automatic batch workflow, a lightweight review workflow, and manual continuum placement. Background selection and fitting can run together or as two separately inspectable stages.

This is research software. Automatic fits should not be treated as scientifically valid without inspecting the diagnostic plots and quality columns.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate             # macOS / Linux
python -m pip install --upgrade pip
python -m pip install -e .
```

In Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` in place of `source`.
If script activation is restricted, run `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`
and use `.\.venv\Scripts\python.exe -m sii_fitter` in place of `sii-fit`.

For development and tests, use `python -m pip install -e '.[dev]'`.

## Input data

### Spectrum files

Spectrum files are whitespace- or comma-separated ASCII tables. Wavelengths must be observed-frame **air** Ångströms and sorted wavelength/flux values must cover 5250–6750 Å in the rest frame. Convert vacuum wavelengths to air before use; the pipeline does not perform that conversion.

- Two columns: wavelength, flux. Uncertainties are estimated from the spectrum.
- Three or more columns: by default the first three are wavelength, flux, and 1σ flux uncertainty; additional columns are ignored.
- If a third column is not an uncertainty, set `input_format` to `flux` in the catalogue or pass `--input-format flux`.
- Missing, non-finite, or non-positive uncertainties (including `-99` sentinels) are estimated from the spectrum. Outputs record `uncertainty_source` and `n_estimated_uncertainties`. Extra systematic-error columns are not combined automatically.

### Catalogue

Only three columns are required:

```csv
spectrum,redshift,mwebv
my_spectrum.dat,0.031,0.018
```

Spectrum paths may be absolute, relative to the catalogue, or bare filenames inside a `spectra/` folder beside the catalogue. `--spectra-dir PATH` supplies another fallback directory.

Saved catalogues use relative spectrum paths where possible. Windows paths on a different drive remain absolute; update them or supply valid replacement paths when moving a run to another computer.

The included [example catalogue](example_catalog.csv) uses optional `name` and `phase` columns. Optional metadata is copied into output tables and may be used in plot labels.

| Canonical column | Accepted legacy aliases | Purpose |
| --- | --- | --- |
| `spectrum` | `spec_name`, `spectrum_file`, `file` | Spectrum path |
| `redshift` | `z` | Systemic redshift |
| `mwebv` | `mw_ebv`, `ebv` | Milky Way E(B−V) |
| `name` | `object`, `object_name`, `ztfname`, `sn_name` | Display name; defaults to file stem |
| `record_id` | `id`, `spectrum_id` | Unique restart key; generated if absent |
| `phase` | `spec_phase` | Optional phase in days |
| `input_format` | `type_of_spec`, `spectrum_format` | `auto`, `flux`, or `flux-error` |
| `mode` | `fit_mode` | Per-spectrum `auto`, `review`, or `manual` |
| `vexp` | `smoothing_width` | Optional σ(λ)/λ smoothing width |

The old `original` and `homogenised` format values are accepted as aliases for `flux` and `flux-error`.

## Choosing a workflow

| Situation | Suggested workflow |
| --- | --- |
| A few spectra or difficult/low-S/N data | `manual` or `review` |
| A moderate sample | `review`, or a mixed catalogue |
| A large sample | `auto`, inspect QC/plots, then rerun flagged spectra in `review` or `manual` |
| Team-curated continua | Separate `background` and `fit` stages |

`auto` never asks questions. `review` shows automatic anchors and asks only whether to accept the background and fit; rejecting the background opens the four-click selector. `manual` opens that selector immediately. There are no Hα, sodium, or unrelated classification prompts.

For a mixed run, add a `mode` column:

```csv
spectrum,redshift,mwebv,mode
clean.dat,0.031,0.018,auto
noisy.dat,0.044,0.026,review
peculiar.dat,0.019,0.011,manual
```

## Command-line examples

End-to-end automatic fitting:

```bash
sii-fit run example_catalog.csv --output-dir sii_results --mode auto --seed 42
```

Interactive review:

```bash
sii-fit run example_catalog.csv --output-dir sii_results_review --mode review
```

Manual fitting for the first catalogue row:

```bash
sii-fit run example_catalog.csv --output-dir sii_results_manual --mode manual --start 0 --end 1
```

Two-stage processing:

```bash
sii-fit background example_catalog.csv --output-dir sii_backgrounds --mode review
sii-fit fit sii_backgrounds/backgrounds.csv --output-dir sii_fits --seed 42
```

`backgrounds.csv` is deliberately editable. This makes it possible to inspect, version, or curate the four velocity-space continuum anchors before running an expensive batch fit.

For long runs, `--resume` skips successful `record_id` values already present in the output. `--start` and `--end` support catalogue slicing. Progress and errors are checkpointed after every spectrum, so a bad file does not discard earlier work.

Use a fresh output directory for a new analysis or changed settings. Existing checkpoints are protected from accidental replacement; `--resume` continues the same analysis and does not refit completed IDs. Catalogue validation (including missing files or invalid required values) runs before processing; errors encountered while fitting individual spectra are logged per record.

Useful fitting options:

```text
--niter N                    Monte Carlo draws; 0 disables MC
--mc-mode MODE               resample, shift, shift_resample, or all
--continuum-shift VALUE      anchor-shift σ in 10³ km/s
--vexp VALUE                 fixed σ(λ)/λ; default is S/N-based
--vel-width VALUE            continuum-window width in 10³ km/s
--no-clip                    disable outlier clipping
--input-format FORMAT        auto, flux, or flux-error
```

Run `sii-fit COMMAND --help` for the full interface.

## Python examples

The repository contains directly runnable examples:

```bash
python examples/automatic_example.py
python examples/manual_example.py
```

The same API can be used in a notebook:

```python
from sii_fitter import FitConfig, run_catalog

config = FitConfig(
    mode="auto",
    n_iterations=200,
    mc_mode="shift_resample",
    seed=42,
)
summary = run_catalog("catalog.csv", "results", config=config, resume=True)
print(summary)
```

Advanced automatic-anchor bounds and fit constraints are fields on `FitConfig`, rather than private constants hidden in runner scripts.

## Outputs

An end-to-end run creates:

```text
results/
├── backgrounds.csv
├── fit_results.csv
├── fit_log.csv
└── plots/
    └── <safe_record_id>-<hash>.png
```

- `backgrounds.csv`: four continuum-anchor velocities, window width, selection status, preprocessing settings, and metadata.
- `fit_results.csv`: background values, deterministic best-fit measurements, and Monte Carlo summaries.
- `fit_log.csv`: success/failure, exception type/message, and runtime for every attempted spectrum.
- `plots/`: continuum, deterministic fit, and Monte Carlo diagnostics.

Velocities and FWHM values are in 10³ km s⁻¹. Equivalent widths are in Å. Wavelengths are converted to the rest frame using the catalogue redshift. Milky Way extinction is corrected with R(V)=3.1.

`background_qc_flag=1` means at least one automatic anchor used a fallback, clipping, or another non-ideal selection status. `mc_qc_flag=1` means fewer than 80% of requested Monte Carlo draws succeeded; `mc_success_fraction` gives the exact fraction. Monte Carlo columns also include per-mode successful/failed draw counts, compact failure reasons, and distribution flags. Optimizer status and fit statistics are saved with the deterministic measurements. These are triage signals, not substitutes for visual inspection.

## Code map

- `sii_fitter/core.py`: spectrum preparation, anchor search, profile fitting, Monte Carlo calculation, and plotting primitives.
- `sii_fitter/catalog.py`: minimal/legacy catalogue validation and path resolution.
- `sii_fitter/pipeline.py`: restartable batch and interactive orchestration.
- `sii_fitter/cli.py`: `sii-fit` command-line interface.
- `sii_fitter/style.py`: self-contained Matplotlib settings; no private `georgios` style dependency.
- `examples/`: automatic and manual runnable examples.
- `tests/`: catalogue, style, and numerical utility checks.

## Scientific assumptions and current limits

- The input wavelength scale is assumed to be observed-frame air Ångströms, matching the [NIST Si II air line wavelengths](https://www.physics.nist.gov/PhysRefData/Handbook/Tables/silicontable2.htm). Verify the wavelength convention of each data source, including the example spectra, whose headers do not specify it.
- The first uncertainty column, when used, is assumed to be a 1σ error.
- The continuum is locally linear in the selected windows.
- The two Si II doublets are represented by coupled Gaussian components with configurable velocity/FWHM ratios and equal component amplitudes within each doublet.
- The minimum fitted width is estimated from pixel sampling near Si II 6355. It is not a measured instrumental resolution, and widths are not deconvolved from an instrumental profile.
- Monte Carlo resampling assumes independent Gaussian pixel errors. Uncertainties estimated for flux-only inputs come from smoothed absolute residuals and are approximate, not calibrated 1σ measurements. Deterministic parameter errors are conditional on the selected continuum; shift-based MC explores anchor-placement sensitivity.
- Outlier clipping is enabled by default and is applied consistently to background selection and fitting. Disable it when narrow real features could be affected.
- Automatic smoothing is estimated from S/N. Record a fixed `vexp` when exact cross-run consistency is more important than adapting to data quality.
- Interactive modes require a Matplotlib GUI backend and therefore should not be run on a headless worker.
- Parallel workers are intentionally not enabled for interactive modes. For large automatic jobs, split the catalogue with `--start/--end`, use separate output directories, and combine reviewed tables only after checking unique `record_id` values.

Before a public release, choose a repository license and add the preferred citation/data-release attribution, including permission and attribution for the bundled example spectra and any adapted algorithms. Those are authorship decisions and are intentionally not guessed here. See [the review and release notes](docs/REVIEW.md) for validation evidence and remaining checks.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest
```

GitHub Actions runs installation, lint, tests, and package builds on Linux (Python 3.10 and 3.13) and Windows (Python 3.13). GUI interaction and scientific validation against independently reviewed measurements still require manual checks.
