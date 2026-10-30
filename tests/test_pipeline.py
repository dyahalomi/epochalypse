#!/usr/bin/env python
"""Self-check for the parts of the pipeline that can go wrong silently.

    python test_pipeline.py

Needs only numpy and pandas -- no jax, no scan law, no generated catalog. What
it guards:

* the seeding scheme (a change here silently changes the whole catalog);
* per-source determinism, which is what makes the pipeline shardable;
* the priors and the Hill screen actually holding on real draws, the a-prior
  floor clearing every star's Roche lobe, and the periodogram grid bracketing
  every period the prior can inject;
* the mass-radius interpolation edge behaviour (NaN, not extrapolation);
* that every path resolves inside the checkout -- `config.ROOT` is found by
  walking up from `__file__`, and a path escaping it is not otherwise visible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from epochalypse import config as C
from epochalypse import planets as P
from epochalypse import stars as S

REPO = Path(__file__).resolve().parents[1]  # tests/ -> the checkout


def star(source_id, *, mass=0.55, radius=0.52):
    """One synthetic parent-sample row, the median star of the 250 pc sample."""
    return pd.Series(
        {
            "gaia_source_id": str(source_id),
            "mass_interp": mass,
            "radius_interp": radius,
            "parallax": 8.1,
            "sig_AL": 0.21,
            "sig_cal": 0.11,
            "sig_att_radec": 0.05,
            "astrometric_n_good_obs_al_dr3": 380.0,
            "astrometric_matched_transits_dr3": 44.0,
            "astrometric_params_solved_dr3": 31,
        }
    )


def test_defaults_resolve_inside_the_checkout():
    """With nothing configured, every path lands in the checkout.

    `config.ROOT` is found by walking up from `__file__`, so a wrong answer here
    is silent. The delivered inputs and the outputs are both relocatable now
    (`--data-root`, `--output-root`) -- this pins the DEFAULTS, which is what a
    fresh clone and the smoke test rely on.
    """
    paths = [
        C.ROOT,
        C.DATA_ROOT,
        C.REFERENCE_DIR,
        C.g23h_sample(),
        C.scanlaw_dr4(),
        C.PECAUT_MAMAJEK,
        C.GOST_FOV_MAP,
        C.OUTPUT_ROOT,
        C.stars_csv(),
        C.index_dir(),
        C.truths("1_companion"),
        C.shard_epochs("2_companion", 3, 512),
    ]
    for path in paths:
        assert REPO in Path(path).parents or Path(path) == REPO, (
            f"{path} escapes {REPO} -- the defaults must stay in the checkout"
        )


def test_roots_are_relocatable():
    """`--data-root` and `--output-root` move what they claim to, and no more.

    The reference data must NOT follow the data root: it is committed, and the
    tests and a fresh clone depend on finding it in the checkout.
    """
    data_before, out_before = C.DATA_ROOT, C.OUTPUT_ROOT
    try:
        C.set_data_root("/tmp/ceph-data")
        C.set_output_root("/tmp/ceph-out")

        assert str(C.g23h_sample()).startswith("/private/tmp/ceph-data") or str(
            C.g23h_sample()
        ).startswith("/tmp/ceph-data"), C.g23h_sample()
        assert C.scanlaw_dr4().parent == C.g23h_sample().parent
        assert C.stars_csv().is_relative_to(C.OUTPUT_ROOT)
        assert C.shard_epochs("1_companion", 0, 1).is_relative_to(C.OUTPUT_ROOT)

        # reference data stays put
        assert C.PECAUT_MAMAJEK.is_relative_to(REPO)
        assert C.GOST_FOV_MAP.is_relative_to(REPO)
        assert C.PECAUT_MAMAJEK.exists() and C.GOST_FOV_MAP.exists()
    finally:
        C.set_data_root(data_before)
        C.set_output_root(out_before)


def test_seeding():
    """Pinned seeds: any change here regenerates a different catalog."""
    assert P.system_seed(42, "0_companion", "12345") == 428296313
    assert P.system_seed(42, "1_companion", "12345") == 1461313147
    assert P.system_seed(42, "0_companion", "5484066448309985152") == 2648412945
    # keyed on the id, not a row index: order of use is irrelevant
    assert P.system_seed(42, "1_companion", 12345) == P.system_seed(
        42, "1_companion", "12345"
    )
    # and the streams are separated per population and per master seed
    seeds = {
        P.system_seed(m, pop, "12345")
        for m in (C.SEED_PLANETS, C.SEED_ASTROMETRY)
        for pop in C.POPULATIONS
    }
    assert len(seeds) == 6, "population/master seed streams collide"


def test_determinism():
    """The same source gives the same companions, every time, in any order."""
    for population in ("1_companion", "2_companion"):
        first = P.draw_companions(
            population, star(777), n_transits=44, sigma_single=0.21
        )
        P.draw_companions(population, star(888), n_transits=44, sigma_single=0.21)
        again = P.draw_companions(
            population, star(777), n_transits=44, sigma_single=0.21
        )
        assert first == again, (
            f"{population} draw is not a pure function of the source id"
        )
    # a control population injects nothing
    assert (
        P.draw_companions("0_companion", star(777), n_transits=44, sigma_single=0.21)
        == []
    )


def test_priors_and_screens():
    """Draw a real sample and check every bound and screen it must satisfy."""
    n = 600
    one = pd.DataFrame(
        [
            P.draw_companions(
                "1_companion", star(10**15 + i), n_transits=44, sigma_single=0.21
            )[0]
            for i in range(n)
        ]
    )

    assert one["sma"].between(C.A_MIN_AU, C.A_MAX_AU).all()
    assert one["mass_pl"].between(C.MASS_MIN_MJUP, C.MASS_MAX_MJUP).all()
    assert one["ecc"].between(C.ECC_MIN, C.ECC_MAX).all()
    assert one["inc"].between(0.0, 180.0).all()
    for angle in ("Omega", "omega", "M_anom"):
        assert one[angle].between(0.0, 360.0).all()

    # log-uniform in a: equal probability per decade, over the full decades
    decades = np.floor(np.log10(one.loc[one["sma"] > 0.1, "sma"]))
    counts = decades.value_counts(normalize=True)
    assert set(counts.index) == {-1.0, 0.0, 1.0}, counts
    assert counts.max() - counts.min() < 0.10, (
        f"log10(a) not flat per decade:\n{counts}"
    )

    # isotropic orbits: cos i uniform, so half the systems sit above 90 deg
    assert abs((one["inc"] > 90).mean() - 0.5) < 0.08

    # Kepler's third law round-trips
    mtot = 0.55 + one["mass_pl"] * C.MJUP_IN_MSUN
    assert np.allclose(one["period"], np.sqrt(one["sma"] ** 3 / mtot))


def test_snr_suppression_is_period_over_baseline():
    """snr_eff = snr_single / (1 + (P/T)^2) for every companion, at any mass.

    a_crit carries the companion mass, so the (a/a_crit)^3 form the generator
    evaluates IS the (P/T)^2 form by Kepler's third law. A star-only a_crit
    would inflate the suppression by (1 + M_p/M_star), up to a factor of two
    for the heaviest companions around the lightest stars.
    """
    rows = []
    for i in range(400):
        rows += P.draw_companions(
            "2_companion", star(10**16 + i), n_transits=44, sigma_single=0.21
        )
    two = pd.DataFrame(rows)
    supp = 1.0 + (two["period"] / C.BASELINE_YEARS) ** 2
    assert np.allclose(two["snr_eff"], two["snr_single"] / supp, rtol=1e-12)
    assert np.allclose(two["snr_total"], np.sqrt(44) * two["snr_eff"], rtol=1e-12)
    # the sample must actually exercise the heavy-companion regime
    assert (two["mass_pl"] * C.MJUP_IN_MSUN / 0.55 > 0.05).any()


def pecaut_track():
    """(abs_G, mass, radius) densely along the interpolated Pecaut & Mamajek track.

    Mass and radius come only from interpolating this table (off-table stars
    are dropped), so extremes over the track bound every star in every sample.
    """
    pecaut = S.read_pecaut()
    pecaut = pecaut.dropna(subset=["M_G", "Msun", "R_Rsun"]).sort_values("M_G")

    # dense in M_G: the interpolated track, not just the tabulated spectral types
    abs_g = np.linspace(pecaut["M_G"].min(), pecaut["M_G"].max(), 20_001)
    mass = np.interp(abs_g, pecaut["M_G"], pecaut["Msun"])
    radius = np.interp(abs_g, pecaut["M_G"], pecaut["R_Rsun"])
    return abs_g, mass, radius


def test_a_min_clears_every_roche_lobe():
    """No star the pipeline can build overfills its Roche lobe at A_MIN_AU.

    This is why the draw has no per-star Roche screen. Raise the radius model
    past the main sequence or lower A_MIN_AU and this fails rather than letting
    contact systems through.
    """
    abs_g, mass, radius = pecaut_track()
    # the floor grows with companion mass, so the heaviest companion is the worst
    floor = P.roche_lobe_min_separation(radius, mass, C.MASS_MAX_MJUP * C.MJUP_IN_MSUN)
    assert floor.max() < C.A_MIN_AU, (
        f"a star at M_G={abs_g[floor.argmax()]:.2f} overfills its Roche lobe out "
        f"to {floor.max():.4f} AU, beyond A_MIN_AU = {C.A_MIN_AU} AU"
    )


def test_period_grid_brackets_every_injectable_period():
    """The periodogram grid covers every period the prior can inject, any star.

    Shortest: A_MIN_AU around the heaviest star with the heaviest companion.
    Longest: A_MAX_AU around the lightest star with the lightest companion.
    Change the a prior or the mass table without moving the grid and this fails.
    """
    from epochalypse.periodogram import config as PC

    _, mass, _ = pecaut_track()
    shortest = P.semimajor_axis_to_period(
        C.A_MIN_AU, mass.max() + C.MASS_MAX_MJUP * C.MJUP_IN_MSUN
    )
    longest = P.semimajor_axis_to_period(
        C.A_MAX_AU, mass.min() + C.MASS_MIN_MJUP * C.MJUP_IN_MSUN
    )
    assert PC.P_MIN < shortest, f"P_MIN = {PC.P_MIN} yr > shortest {shortest:.3e} yr"
    assert longest < PC.P_MAX, f"P_MAX = {PC.P_MAX} yr < longest {longest:.1f} yr"


def test_two_companion_pairs():
    """Pairs come out inner-first, non-crossing, and Hill-stable."""
    pairs = [
        P.draw_companions(
            "2_companion", star(10**15 + i), n_transits=44, sigma_single=0.21
        )
        for i in range(300)
    ]
    coplanar = 0
    for inner, outer in pairs:
        assert inner["sma"] <= outer["sma"], "companion 1 is not the inner one"
        assert inner["coplanar"] == outer["coplanar"]
        mu = [c["mass_pl"] * C.MJUP_IN_MSUN / 0.55 for c in (inner, outer)]
        label = P.classify_with_resonance(
            mu[0], mu[1], inner["sma"], outer["sma"], inner["ecc"], outer["ecc"]
        )
        assert label in ("stable", "resonant_stable_possible"), label
        if inner["coplanar"]:
            coplanar += 1
            assert inner["inc"] == outer["inc"] and inner["Omega"] == outer["Omega"]
    # a coin flip decides coplanarity
    assert abs(coplanar / len(pairs) - C.COPLANAR_PROBABILITY) < 0.10

    columns = P.companion_columns(pairs[0])
    assert columns["n_planets"] == 2
    assert columns["P_ratio"] == pairs[0][1]["period"] / pairs[0][0]["period"]
    for key in (
        "sma_1",
        "sma_2",
        "snr_total_1",
        "snr_total_2",
        "coplanar",
        "near_2_1",
        "near_3_2",
        "near_resonance",
    ):
        assert key in columns, key
    assert P.companion_columns([])["n_planets"] == 0


def test_noise_model():
    """sigma_single is finite for a normal star and NaN when unusable."""
    sigma, n_dof = P.star_noise_terms(star(1))
    assert n_dof == C.N_DOF_FIVE_PARAM and np.isfinite(sigma) and sigma > 0

    other = star(1).copy()
    other["astrometric_params_solved_dr3"] = 95
    assert P.star_noise_terms(other)[1] == C.N_DOF_OTHER

    # no transits, or fewer good observations than degrees of freedom
    assert np.isnan(P.single_datum_sigma(380, 0, 5, 0.11, 0.21))
    assert np.isnan(P.single_datum_sigma(4, 44, 5, 0.11, 0.21))
    assert np.isnan(P.single_datum_sigma(np.nan, 44, 5, 0.11, 0.21))


def test_mass_radius_interpolation():
    """Linear in absolute G, NaN outside the table -- never extrapolated."""
    frame = pd.DataFrame(
        {
            "phot_g_mean_mag_dr3": [8.0, 12.5, 15.9, 4.0, 20.0, np.nan, 11.0, 13.0],
            "parallax": [8.1, 24.0, 4.2, 15.5, 40.0, 10.0, np.nan, -3.0],
        }
    )
    got = S.add_mass_radius_from_pecaut(frame, verbose=False)
    # pinned against the Pecaut & Mamajek table shipped in data/
    assert np.allclose(
        got["mass_interp"][:4],
        [1.590184664537, 0.428309872783, 0.46649227884, 3.434485302474],
    )
    # off the faint end of the sequence, and rows with no usable G or parallax
    assert got["mass_interp"][4:].isna().all()
    assert got["radius_interp"][4:].isna().all()


def test_main_sequence_offset():
    """Zero on the ridge, signed by brightness, NaN off the colour range."""
    # G2V and M0V rows of the table: Bp-Rp 0.823 / 1.84, M_G 4.635 / 8.16
    on_ridge = S.main_sequence_offset([4.635, 8.16], [0.823, 1.84])
    assert np.allclose(on_ridge, 0.0), on_ridge
    # brighter than the ridge is negative
    assert np.isclose(S.main_sequence_offset(4.635 - 1.5, 0.823), -1.5)
    # bluer than B9V, redder than M8.5V, or no colour: never extrapolated
    assert np.isnan(S.main_sequence_offset([5.0, 5.0, 5.0], [-0.5, 6.0, np.nan])).all()


def test_source_list_cuts():
    """Both noise terms are required, and only the main-sequence band survives."""
    from epochalypse.sources import simulated_sources

    frame = pd.DataFrame(
        {
            "sig_AL": [0.2, np.nan, 0.2, 0.2, 0.2, 0.2],
            "sig_att_radec": [0.05, 0.05, np.nan, 0.05, 0.05, 0.05],
            # on the G2V ridge, then 0.9 above / 3 above / 1.5 below it
            "abs_G": [4.635, 4.635, 4.635, 3.735, 1.635, 6.135],
            "bp_rp_dr3": [0.823] * 6,
        }
    )
    keep = simulated_sources(frame, verbose=False)
    assert keep.tolist() == [True, False, False, True, False, False], keep

    before = C.MAIN_SEQUENCE_ONLY
    try:
        C.MAIN_SEQUENCE_ONLY = False
        keep = simulated_sources(frame, verbose=False)
        assert keep.tolist() == [True, False, False, True, True, True], keep
    finally:
        C.MAIN_SEQUENCE_ONLY = before


def test_config_exposes_every_constant_the_package_reads():
    """`config.X` names are re-exports; a linter once stripped four of them.

    They are unused *inside* config.py, so an unused-import pass removes them
    and the simulator dies on the first source -- inside a try/except, on a
    512-rank job. Cheaper to notice here.
    """
    import re

    package = Path(__file__).resolve().parents[1] / "src" / "epochalypse"
    referenced = set()
    for module in package.glob("*.py"):
        referenced |= set(re.findall(r"\bC\.([A-Z][A-Z0-9_]+)\b", module.read_text()))
    assert referenced, "found no C.CONSTANT references -- did the alias change?"
    missing = sorted(name for name in referenced if not hasattr(C, name))
    assert not missing, f"config.py does not define: {missing}"


def test_high_snr_is_a_floor_on_every_companion():
    """The high-SNR sample is a physical floor, not a quantile.

    It was `nlargest(1% by max SNR_tot)` once. A system now qualifies only if
    EVERY injected companion clears `HIGH_SNR_MIN`, so a two-companion system
    with one strong and one weak companion is out -- which a max-based or
    quantile rule would have kept.
    """
    from epochalypse.sources import select_high_snr

    frame = pd.DataFrame(
        {
            "gaia_source_id": ["1", "2", "3", "4", "5"],
            "snr_total_1": [9.0, 9.0, 1.0, 9.0, 9.0],
            "snr_total_2": [9.0, 1.0, 1.0, np.nan, C.HIGH_SNR_MIN],
        }
    )
    kept = set(select_high_snr(frame)["gaia_source_id"])
    assert kept == {"1", "5"}, kept  # 5 sits exactly on the floor: >= not >
    # not a fixed fraction of the input
    assert len(select_high_snr(frame)) == 2

    # one-companion populations have no snr_total_2 column at all
    one = frame[["gaia_source_id", "snr_total_1"]]
    assert set(select_high_snr(one)["gaia_source_id"]) == {"1", "2", "4", "5"}

    # an explicit floor overrides the configured one
    assert (
        len(select_high_snr(frame, snr_min=0.5)) == 4
    )  # all but row 4, whose NaN never clears


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")
