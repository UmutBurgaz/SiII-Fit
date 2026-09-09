from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from sii_fitter import core
from sii_fitter.style import PLOT_STYLE

ROOT = Path(__file__).resolve().parents[1]


def test_relativistic_velocity_wavelength_round_trip():
    velocity = np.array([-30.0, -11.0, 0.0, 8.0])
    wavelength = core.vel_to_wl(velocity, core.rest_wl_sil1_1)
    recovered = core.wl_to_vel(wavelength, core.rest_wl_sil1_1)
    np.testing.assert_allclose(recovered, velocity, atol=1e-10)


def test_requested_plot_style_is_self_contained():
    assert PLOT_STYLE["xtick.direction"] == "in"
    assert PLOT_STYLE["ytick.right"] is True
    assert PLOT_STYLE["figure.figsize"] == (9.7, 6)
    assert PLOT_STYLE["font.family"][0] == "Times New Roman"


@pytest.mark.parametrize("root_velocity, expected_count", [(0.5, 1), (7.0, 0)])
def test_extrema_search_rejects_roots_outside_its_velocity_window(root_velocity, expected_count):
    root_wavelength = core.vel_to_wl(root_velocity, core.rest_wl_sil1_1)
    roots, second_derivatives = core._scan_extrema_in_window(
        lambda wavelength: -(wavelength - root_wavelength),
        lambda wavelength: -1.0,
        v_center_1e3=0.0,
        rest_wl=core.rest_wl_sil1_1,
        v_window_kms=1000.0,
        dv_kms=1000.0,
    )
    assert len(roots) == expected_count
    if expected_count:
        assert roots[0] == pytest.approx(root_wavelength)
        assert second_derivatives[0] < 0


def test_anchor_selection_rejects_extrapolated_extrema():
    wavelength_grid = np.linspace(6000.0, 6400.0, 100)
    # The closest candidate is outside observed coverage, but inside velocity bounds.
    candidates = [(6500.0, 7.13), (6300.0, -2.23)]
    selected = core._pick_with_index_exclusion_and_bounds(
        candidates,
        v_center_1e3=7.0,
        wl_grid=wavelength_grid,
        exclude_idx=[],
        min_sep=5,
        bounds_kms=(-15_000.0, 15_000.0),
    )
    assert selected == 6300.0


def test_joint_fit_recovers_known_synthetic_line_parameters():
    wavelength = np.linspace(5200.0, 6800.0, 801)
    flux = (
        1.0
        + core.g_sil1_1(wavelength, -0.2, -11.0, 3.5)
        + core.g_sil1_2(wavelength, -0.2, -11.0, 3.5)
        + core.g_sil2_1(wavelength, -0.08, -11.5, 3.0)
        + core.g_sil2_2(wavelength, -0.08, -11.5, 3.0)
    )
    spectrum = pd.DataFrame(
        {
            "rest_wl": wavelength,
            "norm_fl": flux,
            "norm_flerr": np.full(wavelength.size, 0.01),
            "vel_sil1_1": core.wl_to_vel(wavelength, core.rest_wl_sil1_1),
            "vel_sil2_1": core.wl_to_vel(wavelength, core.rest_wl_sil2_1),
        }
    )
    background = {
        "sil6355_blue_side": -24.0,
        "sil6355_red_side": 0.0,
        "sil5972_blue_side": -24.0,
        "sil5972_red_side": -1.0,
    }
    metrics, figure, *_ = core.fit_features(spectrum, background, vel_width=2.0)
    try:
        assert metrics["optimizer_success"]
        assert metrics["sil6355_vel"] == pytest.approx(-11.0, abs=0.01)
        assert metrics["sil5972_vel"] == pytest.approx(-11.5, abs=0.01)
        assert metrics["sil6355_fwhm"] == pytest.approx(3.5, abs=0.01)
        assert metrics["sil5972_fwhm"] == pytest.approx(3.0, abs=0.01)
        assert metrics["sil6355_ew"] > metrics["sil5972_ew"] > 0
    finally:
        core.plt.close(figure)


@pytest.mark.parametrize("bad_count", [2, 301])
def test_negative_uncertainty_sentinels_are_estimated_and_recorded(tmp_path, bad_count):
    wavelength = np.linspace(5000.0, 7500.0, 301)
    flux = 1.0 + 0.1 * np.sin(wavelength / 100.0) + 0.01 * np.sin(wavelength)
    uncertainty = np.full(wavelength.size, 0.03)
    uncertainty[:bad_count] = -99.0
    path = tmp_path / "negative-errors.dat"
    np.savetxt(path, np.column_stack((wavelength, flux, uncertainty)))

    spectrum = core.prepare_spectrum(path, redshift=0.0, mwebv=0.0, vexp=0.004)
    estimated_errors = core.spec_create_errors(
        spectrum["wl"].to_numpy(), spectrum["fl"].to_numpy(), vexp=0.004
    )
    np.testing.assert_allclose(spectrum["fl_err"].iloc[:bad_count], estimated_errors[:bad_count])
    np.testing.assert_allclose(spectrum["fl_err"].iloc[bad_count:], 0.03)
    assert spectrum.attrs["input_format"] == "flux-error"
    assert spectrum.attrs["uncertainty_source"] == (
        "estimated" if bad_count == len(wavelength) else "mixed"
    )
    assert spectrum.attrs["n_estimated_uncertainties"] == bad_count


def test_manual_selector_records_four_clicked_anchors(monkeypatch):
    spectrum = core.prepare_spectrum(
        ROOT / "spectra" / "2024srp_0_20240807_SEDM.DAT",
        redshift=0.06037676,
        mwebv=0.13490324,
        vexp=0.004,
        type_of_spec="flux",
    )
    monkeypatch.setattr(core.plt, "get_backend", lambda: "test-interactive")
    monkeypatch.setattr(Figure, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(core.plt, "pause", lambda *args, **kwargs: None)
    wavelengths = iter(
        [
            core.vel_to_wl(0.0, core.rest_wl_sil1_1),
            core.vel_to_wl(-20.0, core.rest_wl_sil1_1),
            core.vel_to_wl(-10.0, core.rest_wl_sil2_1),
            core.vel_to_wl(-22.0, core.rest_wl_sil2_1),
        ]
    )
    monkeypatch.setattr(Figure, "ginput", lambda *args, **kwargs: [(next(wavelengths), 1.0)])

    background = core.select_background_manually(spectrum)
    np.testing.assert_allclose(background["sil6355_red_side"], 0.0, atol=1e-10)
    assert background["sil6355_blue_side"] < background["sil6355_red_side"]
    assert all(value == "manual" for key, value in background.items() if key.startswith("status_"))
