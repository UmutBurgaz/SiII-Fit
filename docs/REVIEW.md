# Private repository review

Review date: 2026-09-09. Status: suitable for private development and further validation; not yet a scientifically validated public release.

## Changes made before initial upload

- Restricted automatic continuum extrema to their requested search windows and observed wavelength coverage.
- Fixed negative uncertainty sentinels being converted to positive uncertainties. Invalid errors are estimated, with the source and count recorded in output catalogues.
- Added validation of finite configuration values, ordered bounds, integer counts, and random seeds.
- Preserved record IDs such as `001` and `NA` when loading and resuming catalogues.
- Made saved spectrum paths work across Windows drives and plot filenames resist collisions and reserved names.
- Made the fitting-stage continuum-width fallback work as documented.
- Prevented failed plot/CSV writes from being treated as completed results on resume, and protected existing output checkpoints from accidental replacement by a fresh run.
- Corrected the documented input wavelength convention to air, matching the constants in the code.
- Added portable font selection, repository exclusions, source-distribution examples, and Linux/Windows GitHub Actions checks.

## Local validation

Environment: Windows, Python 3.13.12, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, Matplotlib 3.11.1, lmfit 1.3.4, PyAstronomy 0.25.0, pytest 9.1.1, Ruff 0.16.6.

| Check | Result |
| --- | --- |
| Installation into a project-local virtual environment | Passed |
| `python -m pip check` | Passed |
| `python -m ruff check .` | Passed |
| `python -m pytest -q` | 54 passed |
| Synthetic coupled-doublet recovery | Passed within the specified test tolerances |
| Four bundled spectra, automatic workflow | 4 succeeded, 0 failed |
| MC `all`, 10 draws per mode per spectrum, seed 42 | 120 successful trials across resample, shift, and shift-resample |
| Source distribution and wheel build | Passed |
| Built-wheel import outside the source directory, shared installed dependencies | Passed |
| Four saved backgrounds fitted using the built wheel, MC disabled | 4 succeeded, 0 failed |
| Two-stage comparison | All 21 numeric deterministic output columns matched within relative tolerance 1e-5 and absolute tolerance 1e-7 |
| Installed `sii-fit --version` | `sii-fit 0.1.0` |

The MC run is a small execution check, not evidence that ten draws are sufficient for science. All four runs had background and MC QC flags of zero; these flags do not validate the scientific interpretation. One rendered diagnostic was inspected for layout and legibility.

PyAstronomy emitted one dependency deprecation warning about `scipy.misc`. It did not cause test or fitting failures. The GitHub Actions page records separate hosted-run results; local checks do not establish compatibility with every operating system or future dependency version.

Local validation outputs are under `verification_output/` and deliberately excluded from version control. They can be regenerated with:

```bash
sii-fit run example_catalog.csv --output-dir verification_output/all_modes --mc-mode all --niter 10 --seed 42
```

Use a new output directory when repeating this command, or `--resume` only to continue an incomplete run with unchanged inputs/settings. Catalogue validation fails before processing if required metadata or a referenced file is missing. Failures during individual spectrum fitting are logged and checkpointed.

## What remains before a public release

1. Compare measurements and uncertainty intervals against independently reviewed spectra, including low-S/N data and spectra with blends or weak Si II 5972. Test sensitivity to continuum placement and the model's constraints.
2. Test the real GUI workflows on intended desktop backends. The automated manual-selector test simulates clicks; it does not validate an actual mouse/keyboard session.
3. Verify input air/vacuum conventions, redshifts, extinction values, flux-error meanings, and the provenance of bundled data. See [the spectrum notes](../spectra/README.md).
4. Decide the project license and preferred citation, confirm redistribution permissions for data and attribution/license requirements for adapted algorithms. Review contributor and commit attribution before making history public.

## Scientific scope

Each doublet uses equal component amplitudes and shared Gaussian velocity/width parameters. The two doublets are coupled through bounded ratios. The width floor depends on pixel sampling, and widths are not instrument-deconvolved. Equivalent widths are finite-window pseudo-equivalent widths relative to locally fitted continua. `r_sil` is EW5972/EW6355.

Deterministic standard errors condition on the chosen continuum. MC resampling assumes independent Gaussian pixel errors. Flux-only and missing-error estimates use smoothed absolute residuals and have not been calibrated as 1σ uncertainties. Additional systematic-error columns are not automatically combined. No observational accuracy or uncertainty-coverage guarantee follows from passing software tests.
