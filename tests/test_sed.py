"""Reference validation for NebulaPy ATLAS spectral-energy distributions."""

import numpy as np

from NebulaPy.src.SED import sed


ENERGY_BINS_EV = [
    [7.902470, 11.26030], [11.26030, 13.59840], [13.59840, 14.53410],
    [14.53410, 16.19920], [16.19920, 21.56450], [21.56450, 24.38310],
    [24.38310, 29.60130], [29.60130, 30.65100], [30.65100, 35.12110],
    [35.12110, 40.96300], [40.96300, 45.14180], [45.14180, 47.88780],
    [47.88780, 54.41780], [54.41780, 63.42330], [63.42330, 77.00000],
]
REFERENCE_TEMPERATURE_K = 35000.0
REFERENCE_METALLICITY = 0.0
REFERENCE_GRAVITY = 4.5
REFERENCE_TOTAL_FLUX = 8.52655614197760000e13
REFERENCE_BIN_FRACTIONS = np.array(
    [
        3.15029591321945190e-01,
        2.15582922101020813e-01,
        1.71370487660169601e-02,
        2.44659129530191422e-02,
        6.36906027793884277e-02,
        1.62787884473800659e-02,
        2.93372268788516521e-03,
        2.00708222109824419e-04,
        3.91597714042291045e-04,
        4.70529957965482026e-05,
    ],
    dtype=np.float64,
)
REFERENCE_ALL_BIN_SUM = 6.55759871006011963e-01


def test_sed(record_property):
    """ATLAS binning reproduces 10 fixed reference flux fractions."""
    atlas = sed(energy_bins=ENERGY_BINS_EV, progress=False)
    atlas.CastelliKuruczAtlas(
        metallicity=REFERENCE_METALLICITY,
        gravity=REFERENCE_GRAVITY,
    )

    temperatures = np.asarray(atlas.Teff, dtype=np.float64)
    binned_flux = np.asarray(atlas.container["binned_flux"], dtype=np.float64)
    total_flux = np.asarray(atlas.container["total_flux"], dtype=np.float64)

    assert atlas.Model == "Atlas"
    assert atlas.container["model"] == "Atlas"
    assert atlas.container["metallicity"] == REFERENCE_METALLICITY
    assert atlas.container["gravity"] == REFERENCE_GRAVITY
    assert atlas.Nmodels == 76
    assert temperatures.shape == (76,)
    assert np.all(np.diff(temperatures) > 0)
    assert binned_flux.shape == (76, len(ENERGY_BINS_EV))
    assert total_flux.shape == (76,)
    assert np.isfinite(binned_flux).all()
    assert np.isfinite(total_flux).all()
    assert np.all(binned_flux >= 0.0)

    zero_flux_indices = np.flatnonzero(total_flux == 0.0)
    np.testing.assert_array_equal(zero_flux_indices, np.array([75]))
    assert temperatures[zero_flux_indices[0]] == 50000.0
    assert np.all(total_flux[:75] > 0.0)
    np.testing.assert_array_equal(binned_flux[zero_flux_indices[0]], 0.0)

    reference_matches = np.flatnonzero(temperatures == REFERENCE_TEMPERATURE_K)
    assert reference_matches.size == 1
    reference_index = int(reference_matches[0])
    actual_fractions = binned_flux[reference_index]

    np.testing.assert_allclose(
        actual_fractions[:10],
        REFERENCE_BIN_FRACTIONS,
        rtol=5.0e-7,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        total_flux[reference_index],
        REFERENCE_TOTAL_FLUX,
        rtol=5.0e-7,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual_fractions.sum(),
        REFERENCE_ALL_BIN_SUM,
        rtol=5.0e-7,
        atol=1.0e-12,
    )

    record_property(
        "test_summary",
        "  Model      : Castelli-Kurucz ATLAS9"
        f"\n  Atmosphere : T={REFERENCE_TEMPERATURE_K:.0f} K, "
        f"[M/H]={REFERENCE_METALLICITY:+.1f}, log(g)={REFERENCE_GRAVITY:.1f}"
        f"\n  Grid       : {atlas.Nmodels} temperatures, "
        f"{len(ENERGY_BINS_EV)} energy bins"
        f"\n  Reference  : {REFERENCE_BIN_FRACTIONS.size} normalized flux points"
        "\n  Checks     : ordering, shape, finite values, total flux, bin fractions",
    )
