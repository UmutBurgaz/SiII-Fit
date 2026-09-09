# Bundled example spectra

These four spectra were supplied with the project. Their headers identify the
observations listed below, but do not establish a public data source, publication,
redistribution license, or wavelength convention.

| File | Header object ID | Observation MJD | Telescope / instrument | Pixels |
| --- | --- | --- | --- | --- |
| `2024srp_0_20240807_SEDM.DAT` | `2024srp` | 60529.20 | P60 / SEDM | 214 |
| `ZTF18aaaooqj_0_20250125_SEDM.DAT` | `ZTF18aaaooqj` | 60700.31 | P60 / SEDM | 214 |
| `ZTF18aaaooqj_1_20250203_SPRAT.DAT` | `ZTF18aaaooqj` | 60709.04 | LT / SPRAT | 891 |
| `ZTF18aabstmw_0_20180307_SEDM.DAT` | `ZTF18aabstmw` | 58184.00 | P60 / SEDM | 208 |

The SPRAT header credits observers Daniel Perley, Jacob Wise, K-Ryan Hinds,
Aleksandra Bochenek, and Zoe McGrath. The other headers contain no observer credit.
These header entries are preserved as supplied; they do not establish ownership
or permission to redistribute the files.

Every file has four columns: `WAVE`, `FLUX`, `FLUX_STATERR`, and `FLUX_SYSERR`.
The default fitter input uses the third column as the statistical 1-sigma flux
uncertainty and ignores the fourth, systematic-uncertainty column. It therefore
does not propagate the supplied systematic uncertainties. In
`ZTF18aabstmw_0_20180307_SEDM.DAT`, all 208 statistical-uncertainty entries are
`-99`, a missing-data sentinel. Invalid uncertainties are estimated from the flux
by the fitter; this example does not supply usable statistical uncertainties.
The other three files contain finite, positive statistical-uncertainty entries.

All four files span the fitter's required 5250–6750 Å rest-frame interval at the
redshifts in `example_catalog.csv`. The air/vacuum wavelength convention, redshift
and extinction sources, phase reference, and preprocessing provenance have not
been verified from the supplied files.

Before making these data public, record the original source or release link,
confirm redistribution permission and the applicable data license, add the
required citation and acknowledgements, and verify the wavelength convention and
catalogue metadata. A software license alone does not settle these data questions.
