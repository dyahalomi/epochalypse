"""Shared machinery for the paper-figure notebooks (``Fig*_*.ipynb``).

Everything the nine notebooks have in common lives here, so a figure cannot
quietly drift from its neighbours: where the catalog is, the plotting style,
the paper-wide 10,000-star subsample rule, the parent-sample cuts, the sky-map
helpers, the population and characterization loaders (in the column vocabulary
the figure code reads), the three-class periodogram classification with its
palette and labels, the constant-SNR_tot loci, the joint mass-period figure,
the nine-periodogram gallery, and a reader that pulls one system's epochs off
the catalog server rather than requiring the 28 GB of epoch shards locally.

The physics and the periodogram come from the generator package itself
(``src/epochalypse`` at the repository root): the main-sequence label, the
per-star noise model, the trial-period grid and ``kepmodel`` search are
imported, not copied, so the figures are built with the code that built the
catalog.

Layout this module assumes (the repository root)::

    epochalypse/
        src/epochalypse/        the generator + characterization package
        catalog/                mirror of the server (download_catalog.sh)
            data/               stars.csv, injected_solutions_*.parquet, index/
            periodograms/       characterization_*.parquet, calibration.json, ...
        analysis/
            paperfig.py         this file
            Fig*_*.ipynb        one notebook per paper figure
            data/               external inputs (DR4 prerelease, LW26 release)
            cache/              column extracts and fetched epochs (rebuilt on demand)
            paper_figures/          the figures as they go in the paper (each notebook's last cell)
            supplementary_figures/  intermediate and alternative versions shown earlier
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================================
# Paths
# ==========================================================================
ROOT = Path(__file__).resolve().parent.parent   # the repository root
REPO = ROOT
ANALYSIS = ROOT / "analysis"
CATALOG = ROOT / "catalog"
DATA_DIR = CATALOG / "data"
PGRAM_DIR = CATALOG / "periodograms"
EXTERNAL = ANALYSIS / "data"
CACHE = ANALYSIS / "cache"
PAPER_FIG_DIR = ANALYSIS / "paper_figures"           # the figures as they go in the paper
SUPP_FIG_DIR = ANALYSIS / "supplementary_figures"    # intermediate / alternative versions
SERVER = "https://users.flatironinstitute.org/~apricewhelan/epochalypse"
N_SHARDS = 320

for _d in (CACHE, PAPER_FIG_DIR, SUPP_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The generator package is installed (editable) in the locked environment, so it
# imports directly; the fallback covers a bare interpreter run from a checkout.
try:
    import epochalypse  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO / "src"))

from epochalypse import config as C  # noqa: E402  (priors, column names, reference data)
from epochalypse import constants as K  # noqa: E402
from epochalypse import stars as S  # noqa: E402  (Pecaut & Mamajek, the main-sequence label)
from epochalypse.periodogram import config as PC  # noqa: E402

# The generator's own path helpers, pointed at the server mirror: C.stars_csv(),
# C.truths(population, high_snr) and PC.calibration_path() then resolve inside
# catalog/, which is the same layout the server has.
C.set_output_root(CATALOG)
PC.set_output_root(PGRAM_DIR)

# Inputs that are too large to mirror and only feed the funnel counts of the
# schematic. Read if present; otherwise the recorded counts are used.
G23H_500PC = Path(
    "/Users/daniel/Desktop/gaia_tooling_workshop/epochalypse_parallel/data/"
    "g23h_epochalypse_stars/G23H_within_500pc.arrow"
)
G23H_FULL = Path("~/Desktop/gaia_tess_ttvs/G23H-v1.0.feather").expanduser()
N_G23H_FULL_RECORDED = 18_061_961       # rows of the full G23H release
N_G23H_500PC_RECORDED = 5_789_521       # rows within 500 pc (one per DR2 cross-match)
N_G23H_500PC_UNIQUE_RECORDED = 5_729_075  # unique DR3 sources among them

MJUP = K.MJUP_IN_MSUN
BASELINE = K.DR4_BASELINE_YEARS
YR = K.DAYS_PER_YEAR

# ==========================================================================
# Style
# ==========================================================================
# The palette of the catalog-description figures (the generator's FIGURES block).
RANDOM_COLOR = "#050CDB"      # the whole population, drawn from the unbiased prior
HIGH_SNR_COLOR = "#DC144D"    # the SNR_tot >= 5 selection
INNER_COLOR = "#01019D"
OUTER_COLOR = "#BB3DF1"
INK = "#1a1a1a"


def setup_style(usetex=True):
    """The typography every figure in the paper uses."""
    import matplotlib

    matplotlib.rc("xtick", labelsize=18)
    matplotlib.rc("ytick", labelsize=18)
    matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    matplotlib.rc("text", usetex=usetex)
    # matplotlib 3.11.0 dropped the math minus sign from usetex PDF output
    # (Type1Font.subset clobbered CMSY code 0, so every "-" in a label vanished
    # in the vector version). 3.11.1 fixed it.
    if matplotlib.__version__ == "3.11.0":
        print("WARNING: matplotlib 3.11.0 has the usetex minus-sign bug; upgrade, or check "
              "every PDF's axis labels before trusting it")


def save(fig, stem, dpi=200, figdir=None, paper=False):
    """Write ``stem.pdf`` (the paper version) and ``stem.png`` (what the notebook shows).

    ``paper=True`` writes into ``paper_figures/`` (a figure as it goes in the paper); the
    default is ``supplementary_figures/`` (intermediate and alternative versions). ``figdir``
    overrides both.
    """
    if figdir is None:
        figdir = PAPER_FIG_DIR if paper else SUPP_FIG_DIR
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "png"):
        path = figdir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **({"dpi": dpi} if suffix == "png" else {}))
        written.append(path)
    print("  wrote " + ", ".join(str(p.relative_to(ROOT)) for p in written))
    return written


def tex_num(n):
    """A thousands-separated integer that survives math mode (``5{,}040{,}328``)."""
    return f"{n:,}".replace(",", "{,}")


# ==========================================================================
# The paper-wide subsample: the n smallest blake2s ranks of the source ids
# ==========================================================================
# Keyed on gaia_source_id, never on a row index, so the same stars come out of
# stars.csv, the truth tables and the characterization tables alike; nested in
# n; invariant to row order. One rule, one seed, every figure.
DEFAULT_SUBSAMPLE_SEED = 20260824


def source_id_ranks(gaia_source_ids, *, seed=DEFAULT_SUBSAMPLE_SEED):
    """A uniform 64-bit rank per source id: ``blake2s(f"{seed}:{id}")``."""
    prefix = f"{int(seed)}:".encode("utf-8")
    ids = np.asarray(gaia_source_ids)
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2s(prefix + str(int(i)).encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            for i in ids
        ),
        dtype=np.uint64,
        count=len(ids),
    )


def subsample_mask(gaia_source_ids, n, *, seed=DEFAULT_SUBSAMPLE_SEED):
    """Boolean mask selecting ``n`` source ids at random, reproducibly."""
    ids = np.asarray(gaia_source_ids)
    if n >= len(ids):
        return np.ones(len(ids), dtype=bool)
    if n <= 0:
        return np.zeros(len(ids), dtype=bool)
    ranks = source_id_ranks(ids, seed=seed)
    order = np.lexsort((ids.astype("int64", copy=False), ranks))
    mask = np.zeros(len(ids), dtype=bool)
    mask[order[: int(n)]] = True
    return mask


def subsample_frame(frame, n, *, seed=DEFAULT_SUBSAMPLE_SEED, id_col="gaia_source_id"):
    """``n`` rows of ``frame`` drawn at random on ``id_col``, in the frame's own order."""
    mask = subsample_mask(frame[id_col].to_numpy().astype("int64"), n, seed=seed)
    return frame.loc[mask].copy()


def draw(frame, n=10_000, seed=DEFAULT_SUBSAMPLE_SEED):
    """``frame`` down-selected to ``n`` systems for plotting as individual markers."""
    return subsample_frame(frame, n, seed=seed).reset_index(drop=True)


# ==========================================================================
# The stellar sample
# ==========================================================================
# The columns of stars.csv the figures read. stars.csv is 7.3 GB and ~117
# columns; these are cached to parquet on first use (a couple of minutes
# through pyarrow.csv) and memory-mapped afterwards.
STAR_COLUMNS = [
    "gaia_source_id", "source_id_dr2", "hip_id", "ra", "dec", "parallax",
    "parallax_error", "phot_g_mean_mag_dr3", "bp_rp_dr3", "ruwe_dr3",
    "abs_G", "mass_interp", "radius_interp",
    "sig_AL", "sig_att_radec", "sig_cal",
    "astrometric_n_good_obs_al_dr3", "astrometric_matched_transits_dr3",
    "astrometric_params_solved_dr3", "astrometric_excess_noise_dr3",
    "predicted_class_msms",
]
STARS_CACHE = CACHE / "stars_columns.parquet"


def star_columns(columns=None, rebuild=False):
    """The requested columns of stars.csv, from the parquet cache (built on demand)."""
    import pyarrow.parquet as pq

    columns = list(columns or STAR_COLUMNS)
    cached = pq.ParquetFile(STARS_CACHE).schema_arrow.names if STARS_CACHE.exists() else []
    if rebuild or not set(columns).issubset(cached):
        import pyarrow.csv as pacsv

        want = list(dict.fromkeys(STAR_COLUMNS + columns))
        print(f"building {STARS_CACHE.name} from stars.csv "
              f"({C.stars_csv().stat().st_size / 1e9:.1f} GB) ...", flush=True)
        table = pacsv.read_csv(
            C.stars_csv(),
            read_options=pacsv.ReadOptions(block_size=64 << 20),
            convert_options=pacsv.ConvertOptions(
                include_columns=want,
                column_types={"gaia_source_id": "int64", "source_id_dr2": "int64",
                              "hip_id": "float64"},
            ),
        )
        pq.write_table(table.select(want), STARS_CACHE, compression="zstd")
        del table
    return pd.read_parquet(STARS_CACHE, columns=columns)


def parent_mask(stars):
    """The noise-complete sample: finite astrometry, positive parallax and mass, both
    per-CCD noise terms. This is the 'parent sample' of the paper's Section 2 before
    the main-sequence cut (5,724,572 stars at 500 pc)."""
    astrometry_ok = (
        np.isfinite(stars["ra"]) & np.isfinite(stars["dec"])
        & np.isfinite(stars["parallax"]) & (stars["parallax"] > 0)
        & np.isfinite(stars["mass_interp"]) & (stars["mass_interp"] > 0)
    )
    return (astrometry_ok & stars["sig_AL"].notna() & stars["sig_att_radec"].notna()).to_numpy()


def main_sequence_offset(stars):
    """Delta M_G from the Pecaut & Mamajek dwarf ridge, per star (the generator's own label)."""
    return S.main_sequence_offset(stars["abs_G"].to_numpy(float), stars[C.BPRP_COL].to_numpy(float))


def main_sequence_class(offset):
    """``main_sequence`` / ``above_ms`` / ``below_ms`` / ``outside_colour_range`` per star."""
    return np.select(
        [np.isnan(offset), offset < C.MS_BRIGHT_EDGE, offset > C.MS_FAINT_EDGE],
        ["outside_colour_range", "above_ms", "below_ms"],
        default="main_sequence",
    )


def simulated_ids():
    """The gaia_source_ids the catalog simulated, from the control population's truth table."""
    return pd.read_parquet(C.truths("0_companion"), columns=["gaia_source_id"])[
        "gaia_source_id"
    ].to_numpy().astype("int64")


def noise_terms(stars):
    """(sigma_UEVA,single, sigma_formal) [mas] per star, vectorized from the DR3 ingredients.

    The first is what the simulator injects (``epochalypse.planets.single_datum_sigma``),
    the second what the catalog reports: the per-CCD error sqrt(sig_att^2 + sig_AL^2)
    inverse-variance-combined over the CCDs of one transit.
    """
    n_dof = np.where(stars["astrometric_params_solved_dr3"] == C.PARAMS_SOLVED_FIVE_PARAM,
                     C.N_DOF_FIVE_PARAM, C.N_DOF_OTHER)
    n_good = stars["astrometric_n_good_obs_al_dr3"].to_numpy(float)
    n_fov = stars["astrometric_matched_transits_dr3"].to_numpy(float)
    sig_al = stars["sig_AL"].to_numpy(float)
    ueva = np.sqrt(((n_fov - n_dof) * stars["sig_cal"].to_numpy(float) ** 2 + n_fov * sig_al**2)
                   / (n_good - n_dof))
    formal = np.hypot(stars["sig_att_radec"].to_numpy(float), sig_al) / np.sqrt(n_good / n_fov)
    return ueva, formal


# ==========================================================================
# Sky map helpers (ported from the generator's figures.py)
# ==========================================================================
def add_curved_colorbar(ax, vmin, vmax, cmap, *, side="top", half_angle=20.0,
                        r_in=1.03, r_out=1.07, nseg=256, ticks=None,
                        tick_labels=None, label="", fontsize=10, text_color="k",
                        edge_color="k"):
    """Colorbar drawn as an annular arc hugging the top or bottom of a healpy
    Mollweide ellipse (drawn as (2 cos t, sin t) in axes data coordinates)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    top = side == "top"
    centre = 90.0 if top else -90.0
    t0 = np.radians(centre + (half_angle if top else -half_angle))
    t1 = np.radians(centre - (half_angle if top else -half_angle))
    t = np.linspace(t0, t1, nseg + 1)
    xi, yi = 2.0 * r_in * np.cos(t), r_in * np.sin(t)
    xo, yo = 2.0 * r_out * np.cos(t), r_out * np.sin(t)
    verts = [[(xi[i], yi[i]), (xi[i + 1], yi[i + 1]), (xo[i + 1], yo[i + 1]), (xo[i], yo[i])]
             for i in range(nseg)]
    colors = cmap((np.arange(nseg) + 0.5) / nseg)
    ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors=colors,
                                     linewidths=1, zorder=5))
    ax.plot(np.r_[xi, xo[::-1], xi[:1]], np.r_[yi, yo[::-1], yi[:1]],
            color=edge_color, lw=1, zorder=6)
    if ticks is None:
        ticks = np.linspace(vmin, vmax, 5)
    if tick_labels is None:
        tick_labels = [f"{value:g}" for value in ticks]
    elif len(tick_labels) != len(ticks):
        raise ValueError(f"{len(tick_labels)} tick labels for {len(ticks)} ticks")
    va = "bottom" if top else "top"
    for value, text in zip(ticks, tick_labels):
        fraction = (value - vmin) / (vmax - vmin)
        tt = t0 + fraction * (t1 - t0)
        ct, st = np.cos(tt), np.sin(tt)
        ax.plot([2 * r_out * ct, 2 * (r_out + 0.015) * ct], [r_out * st, (r_out + 0.015) * st],
                color=edge_color, lw=1, zorder=6)
        angle = np.degrees(np.arctan2(-ct, 2 * st))
        angle = angle - 180 if angle > 90 else (angle + 180 if angle < -90 else angle)
        ax.text(2 * (r_out + 0.03) * ct, (r_out + 0.03) * st, text, rotation=angle,
                rotation_mode="anchor", ha="center", va=va, fontsize=fontsize,
                color=text_color, zorder=6)
    if label:
        ylab = (r_out + 0.135) if top else -(r_out + 0.135)
        ax.text(0.0, ylab, label, ha="center", va=va, fontsize=fontsize + 1,
                color=text_color, zorder=6)
    y0, y1 = ax.get_ylim()
    if top:
        ax.set_ylim(min(y0, -1.05), max(y1, r_out + 0.24))
    else:
        ax.set_ylim(min(y0, -(r_out + 0.24)), max(y1, 1.05))
    x0, x1 = ax.get_xlim()
    ax.set_xlim(min(x0, -2.05), max(x1, 2.05))


def decorate_skymap(ax, display_grat, foreign_grat, label_coord, *, fontsize=9,
                    transit_ticks=(0, 100, 200), transit_tick_labels=None):
    """Graticules, meridian/parallel labels, and the FoV-transit colorbar."""
    import healpy as hp

    hp.visufunc.graticule(coord=foreign_grat, c="silver", linestyle="solid",
                          lw=0.3, dpar=30, dmer=30, alpha=0.5)
    hp.visufunc.graticule(coord=display_grat, c="w", linestyle="dashed",
                          lw=0.4, dpar=30, dmer=30)
    for lat in [-90, -60, -30, 30, 60]:
        hp.visufunc.projtext(0, lat, rf"{lat}$^\circ$", lonlat=True, coord=label_coord,
                             fontweight="bold", fontsize=fontsize, color="silver",
                             ha="left", va="bottom")
    for lon in [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]:
        hp.visufunc.projtext(lon % 360, 0, rf"{lon}$^\circ$", lonlat=True, coord=label_coord,
                             fontweight="bold", fontsize=fontsize, color="silver",
                             ha="left", va="bottom")
    ax.text(-2.0, 0.0, r"180$^\circ$", fontweight="bold", ha="left", va="bottom",
            fontsize=fontsize, color="silver")
    ax.text(2.005, 0.0, r"-180$^\circ$", fontweight="bold", ha="left", va="bottom",
            fontsize=fontsize, color="silver")
    image = ax.get_images()[0]
    vmin, vmax = image.get_clim()
    add_curved_colorbar(ax, vmin, vmax, "bone", side="top", ticks=list(transit_ticks),
                        tick_labels=transit_tick_labels, label="number of FoV transits",
                        fontsize=18)


# ==========================================================================
# The injected populations (truth tables), for the catalog-description figures
# ==========================================================================
POPULATION_LABELS = {
    (0, False): "companion-free control",
    (1, False): "one-companion",
    (2, False): "two-companions",
    (1, True): "one-companion, high-SNR",
    (2, True): "two-companions, high-SNR",
}


def select_high_snr(frame, snr_min=None):
    """Systems in which *every* injected companion clears SNR_tot >= ``snr_min`` (5.0)."""
    from epochalypse.periodogram.calibrate import select_high_snr as _select

    return _select(frame, snr_min).sort_values("gaia_source_id").reset_index(drop=True)


def read_population(n_companions, high_snr=False, columns=None, check=True):
    """A truth table in the column names the catalog figures use.

    ``sma_1 -> sma`` etc. for one-companion populations, ``alpha_mas_k -> alpha_k_mas``,
    ``mass_st_msun -> mass_st``, ``parallax_mas -> parallax``. The high-SNR tables on
    disk are the SNR_tot >= 5 selection the generator wrote; with ``check`` the
    selection is recomputed from the parent table and the two must agree.
    """
    population = f"{n_companions}_companion"
    path = C.truths(population, high_snr=high_snr)
    frame = pd.read_parquet(path, columns=columns)
    if high_snr and check:
        snr_cols = [f"snr_total_{k}" for k in range(1, n_companions + 1)]
        parent = pd.read_parquet(C.truths(population), columns=["gaia_source_id", *snr_cols])
        selected = select_high_snr(parent)
        got = set(frame["gaia_source_id"].astype(str))
        want = set(selected["gaia_source_id"].astype(str))
        assert got == want, (f"{path.name} is not the SNR_tot >= {C.HIGH_SNR_MIN:g} selection "
                             f"of {population}: {len(got):,} rows vs {len(want):,} selected")
    frame = frame.rename(columns={"mass_st_msun": "mass_st", "parallax_mas": "parallax"})
    for k in (1, 2):
        if f"alpha_mas_{k}" in frame.columns:
            frame = frame.rename(columns={f"alpha_mas_{k}": f"alpha_{k}_mas"})
    if n_companions == 1:
        frame = frame.rename(columns={c: c[:-2] for c in frame.columns
                                      if c.endswith("_1") and c != "alpha_1_mas"})
        frame = frame.rename(columns={"alpha_1_mas": "alpha_mas"})
    return frame


# ==========================================================================
# The characterization tables, in the vocabulary of the figure code
# ==========================================================================
# key -> (file population, high-SNR view?, n companions, panel title)
CHAR_SOURCES = {
    "0_companion":            ("0_companion", False, 0, "companion-free control"),
    "1_companion":            ("1_companion", False, 1, "one companion"),
    "2_companion":            ("2_companion", False, 2, "two companions"),
    "1_companion_high_snr":   ("1_companion", True,  1, "one companion: high-SNR"),
    "2_companion_high_snr":   ("2_companion", True,  2, "two companions: high-SNR"),
}
CHAR_LABELS = {k: v[3] for k, v in CHAR_SOURCES.items()}
N_COMPANIONS = {k: v[2] for k, v in CHAR_SOURCES.items()}

# The columns the figures read; pass more through ``columns``.
SYSTEM_COLUMNS = [
    "gaia_source_id", "shard", "shard_row", "n_epochs",
    "klass", "top_power", "best_period", "width_dex", "best_at_edge", "n_competitive",
    "peak1_period", "peak2_period", "accel_delta_chi2", "chi2_5par",
    "n_transits_dr4", "parallax_mas", "mass_st_msun", "sigma_single_mas", "n_planets",
]
COMPANION_COLUMNS = ["period_{k}", "sma_{k}", "ecc_{k}", "mass_pl_{k}", "inc_{k}",
                     "alpha_mas_{k}", "snr_total_{k}", "snr_eff_{k}", "snr_single_{k}",
                     "period_{k}_in_bound", "period_{k}_recovered"]
# generator vocabulary -> what the figure code reads (and the unit conversion)
ALIASES = {"sma_{k}": ("a_{k}_au", 1.0), "ecc_{k}": ("e_{k}", 1.0),
           "mass_pl_{k}": ("Mp_{k}_msun", MJUP), "alpha_mas_{k}": ("alpha_{k}_mas", 1.0)}


def char_path(key):
    population, high_snr, _, _ = CHAR_SOURCES[key]
    return PGRAM_DIR / f"characterization_{population}{'_high_snr' if high_snr else ''}.parquet"


def calibration():
    """calibration.json: the thresholds measured on all companion-free systems."""
    return json.loads(PC.calibration_path().read_text())


def manifest():
    """manifest.json: the grid, the classifier settings and the code version of the run."""
    return json.loads(PC.manifest_path().read_text())


def add_aliases(frame, n_companions=2):
    """Rename the generator's columns to the ones the figure code reads, in place."""
    renames = {}
    for k in range(1, n_companions + 1):
        for source, (target, factor) in ALIASES.items():
            col = source.format(k=k)
            if col in frame.columns:
                if factor != 1.0:
                    frame[col] = frame[col].to_numpy(float) * factor
                renames[col] = target.format(k=k)
        inc = f"inc_{k}"
        if inc in frame.columns:
            frame[inc] = np.radians(frame[inc].to_numpy(float))
            renames[inc] = f"i_{k}_rad"
    frame.rename(columns=renames, inplace=True)
    if "period_1_recovered" in frame.columns and "period_recovered" not in frame.columns:
        frame["period_recovered"] = frame["period_1_recovered"].to_numpy(float)
    return frame


def apply_calibration(frame, thr_orbit=None, thr_accel=None, baseline=BASELINE):
    """The null-calibrated flags (``detected_cal`` & co.), from the run's calibration.json
    unless thresholds are given. The generator's own function."""
    from epochalypse.periodogram.calibrate import apply_calibration as _apply

    cal = calibration()
    return _apply(frame, cal["thr_orbit"] if thr_orbit is None else float(thr_orbit),
                  cal["thr_accel"] if thr_accel is None else float(thr_accel), baseline)


def load_chars(key, columns=None, thr_orbit=None, thr_accel=None, verbose=True):
    """One characterization table, flags applied, in the figure vocabulary.

    ``columns`` defaults to the lean set the figures read (``SYSTEM_COLUMNS`` plus the
    per-companion family); pass ``"all"`` for everything in the file.
    """
    import pyarrow.parquet as pq

    population, high_snr, npl, _ = CHAR_SOURCES[key]
    path = char_path(key)
    if columns == "all":
        use = None
    else:
        want = list(columns) if columns is not None else list(SYSTEM_COLUMNS)
        for k in range(1, npl + 1):
            want += [c.format(k=k) for c in COMPANION_COLUMNS]
        want += ["top_power", "accel_delta_chi2", "klass", "best_period"]
        have = set(pq.ParquetFile(path).schema_arrow.names)
        use = [c for c in dict.fromkeys(want) if c in have]
    frame = pd.read_parquet(path, columns=use)
    if "klass" in frame.columns:
        frame["klass"] = frame["klass"].astype("category")
    add_aliases(frame, npl)
    apply_calibration(frame, thr_orbit, thr_accel)
    if verbose:
        print(f"  {key:22s} {len(frame):>9,} systems   {path.name}"
              + (f"   (SNR_tot >= {C.HIGH_SNR_MIN:g} view)" if high_snr else ""))
    return frame


# ==========================================================================
# The three-class periodogram classification (ported from epochalypse_figures.py)
# ==========================================================================
# Tol colorblind-friendly; the null class is a neutral grey.
CLASS_COLORS = {"undet": "0.8", "narrow": "#332288", "broad": "#796BBA"}
SHORT_3 = {"undet": "no significant peak or acceleration",
           "narrow": "significant peak, period localized",
           "broad": "significant peak or acceleration, period not localized"}
LABELS_3 = {
    "undet": ("no significant peak or acceleration: neither channel exceeds the threshold "
              "calibrated on the companion-free control at a 1\\% false-positive rate"),
    "narrow": ("significant peak, period localized: the competitive region spans less than "
               "0.05 dex in period"),
    "broad": ("significant peak or acceleration, period not localized: the competitive region "
              "is wide, or the significance comes from the acceleration channel alone")}
DRAW_ORDER = ["undet", "broad", "narrow"]      # back to front: the null class is the background
LEGEND_ORDER = ["undet", "narrow", "broad"]
ROW_LABELS = {"undet": "not significant", "narrow": "period localized",
              "broad": "period not localized"}
WIDTH_THR = 0.05   # dex; "narrow" threshold the classifier used (manifest: width_constrained_dex)

# One marker area for every map in the paper, rescaled to a 13-inch reference width so the
# three characterization figures print at matching marker sizes.
MARKER_SIZE, MARKER_SIZE_TRI, MARKER_REF_WIDTH_IN = 23, 29, 13.0
ACCENT = "k"       # the SNR_tot loci and the example badges


def marker_size(fig_width_in, tri=False):
    base = MARKER_SIZE_TRI if tri else MARKER_SIZE
    return base * (float(fig_width_in) / MARKER_REF_WIDTH_IN) ** 2


def is_primary(df, k):
    """A single periodogram addresses at most the dominant (higher snr_total) companion.

    The other companion is left grey -- a one-planet periodogram never tested for it.
    snr_total only RANKS the two, it is never a threshold.
    """
    other = 2 if k == 1 else 1
    oc = f"snr_total_{other}"
    if oc not in df.columns or not df[oc].notna().any():
        return np.ones(len(df), bool)
    stk = np.nan_to_num(df[f"snr_total_{k}"].to_numpy(float), nan=-np.inf)
    sto = np.nan_to_num(df[oc].to_numpy(float), nan=-np.inf)
    return stk >= sto


def classes(df, k, det_col="detected_cal", peak_col="peak_significant_cal"):
    """Class key per row for companion ``k``: ``undet`` / ``narrow`` / ``broad``.

    Data-only -- no truth column is read, so the identical classification can be run on
    real DR4 epoch astrometry. A row is ``narrow`` only if the PERIODOGRAM channel itself
    clears its threshold and the competitive region is narrow (``klass == "unimodal"``);
    the acceleration channel alone can make a row ``broad`` but never ``narrow``.
    """
    active = df[det_col].to_numpy(bool) & is_primary(df, k)
    narrow = (df["klass"].to_numpy(object) == "unimodal") & df[peak_col].to_numpy(bool)
    out = np.full(len(df), "undet", object)
    out[active & narrow] = "narrow"
    out[active & ~narrow] = "broad"
    return out


# --------------------------------------------------------------------------
# axes
# --------------------------------------------------------------------------
def _axis(col, label, scale, lim, nbins=34, line=None, factor=1.0):
    bins = (np.logspace(np.log10(lim[0]), np.log10(lim[1]), nbins) if scale == "log"
            else np.linspace(lim[0], lim[1], nbins))
    return dict(col=col, label=label, scale=scale, lim=lim, bins=bins, line=line, factor=factor)


def period_axis(lim=None):
    """The period axis of the maps: the search grid's own bounds, which bracket the
    injected prior by construction (``period_grid.parquet``)."""
    if lim is None:
        grid = manifest()["grid"]
        lim = (grid["p_min_yr"], grid["p_max_yr"])
    return _axis("period_{k}", "orbital period [yr]", "log", lim, line=BASELINE)


AX_M = _axis("Mp_{k}_msun", r"companion mass [$M_{\rm Jup}$]", "log", (2e-4, 140.0),
             factor=1.0 / MJUP)
AX_A = _axis("alpha_{k}_mas", r"reflex semi-major axis $\alpha$ [mas]", "log", (1e-4, 5e3))


# --------------------------------------------------------------------------
# constant-SNR_tot loci in the mass-period plane
# --------------------------------------------------------------------------
# The catalog's own detectability proxy, rebuilt from a characterization frame so the
# curves drawn on a figure are the SAME quantity the high-SNR populations were selected on:
#     alpha = [Mp/(M*+Mp)] a parallax;  snr_single = alpha/sigma;
#     snr_eff = snr_single / (1 + (a/a_crit)^3), a_crit = (T^2 M*)^(1/3);  snr_total = sqrt(N) snr_eff
# With a ~ P^(2/3) a locus is Mp ~ P^(-2/3) [1 + (P/T)^2]: a V with its minimum at T/sqrt(2).
MAX_LOCUS_STARS = 20_000


def star_terms(df, k=1):
    """Per-row (M*, parallax, sigma, N_fov) rebuilt from a characterization frame."""
    P = df[f"period_{k}"].to_numpy(float)
    Mp = df[f"Mp_{k}_msun"].to_numpy(float)
    a = df[f"a_{k}_au"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mtot = a**3 / P**2
        parallax = df[f"alpha_{k}_mas"].to_numpy(float) * mtot / (Mp * a)
    return (mtot - Mp, parallax, df["sigma_single_mas"].to_numpy(float),
            df["n_transits_dr4"].to_numpy(float))


def _star_terms_capped(df, k=1, max_stars=MAX_LOCUS_STARS):
    terms = star_terms(df, k)
    n = len(terms[0])
    if max_stars is None or n <= max_stars:
        return terms
    take = np.linspace(0, n - 1, int(max_stars)).astype(int)
    return tuple(v[take] for v in terms)


def snr_locus(df, snr, periods, k=1, quantile=0.5, baseline=BASELINE, iters=8,
              max_stars=MAX_LOCUS_STARS):
    """Companion mass [Msun] whose snr_total equals ``snr``, per trial period, at the
    ``quantile`` over the stars of ``df`` (the median star by default)."""
    Ms, par, sig, nfov = (v[:, None] for v in _star_terms_capped(df, k, max_stars))
    p = np.asarray(periods, float)[None, :]
    ok = np.isfinite(Ms) & np.isfinite(par) & (Ms > 0) & (par > 0) & np.isfinite(sig)
    Ms, par, sig, nfov = (np.where(ok, v, np.nan) for v in (Ms, par, sig, nfov))
    mp = np.zeros_like(Ms * p)
    for _ in range(iters):
        mtot = Ms + mp
        a = (p**2 * mtot) ** (1.0 / 3.0)
        supp = 1.0 + (p / baseline) ** 2 * mtot / Ms
        mp = np.clip(snr * sig * supp / (np.sqrt(nfov) * par * a) * mtot, 0.0, 0.5 * Ms)
    return np.nanquantile(mp, quantile, axis=0)


def snr_total(df, k=1, baseline=BASELINE):
    """The catalog's snr_total recomputed from a frame (reproduces its column to ~1e-6)."""
    Ms, par, sig, nfov = star_terms(df, k)
    P = df[f"period_{k}"].to_numpy(float)
    Mp = df[f"Mp_{k}_msun"].to_numpy(float)
    a = (P**2 * (Ms + Mp)) ** (1.0 / 3.0)
    alpha = Mp / (Ms + Mp) * a * par
    return np.sqrt(nfov) * (alpha / sig) / (1.0 + (P / baseline) ** 2 * (Ms + Mp) / Ms)


def snr_at(df, period, mass, k=1, quantile=0.5, baseline=BASELINE, max_stars=MAX_LOCUS_STARS):
    """snr_total of a hypothetical companion on the median star of ``df``."""
    Ms, par, sig, nfov = (v[:, None] for v in _star_terms_capped(df, k, max_stars))
    p = np.atleast_1d(np.asarray(period, float))[None, :]
    m = np.atleast_1d(np.asarray(mass, float))[None, :]
    mtot = Ms + m
    a = (p**2 * mtot) ** (1.0 / 3.0)
    alpha = m / mtot * a * par
    s = np.sqrt(nfov) * (alpha / sig) / (1.0 + (p / baseline) ** 2 * mtot / Ms)
    out = np.nanquantile(s, quantile, axis=0)
    return out if np.ndim(period) or np.ndim(mass) else float(out)


def class_boundary(df, flag, k=1, frac=0.5, pbins=None, mbins=None, nmin=25):
    """The mass at which ``frac`` of companions carry ``flag``, per period bin (NaN where
    the fraction never reaches ``frac``)."""
    P = df[f"period_{k}"].to_numpy(float)
    M = df[f"Mp_{k}_msun"].to_numpy(float)
    flag = np.asarray(flag, bool)
    pbins = np.logspace(-1.3, 3.0, 12) if pbins is None else np.asarray(pbins, float)
    mbins = (np.logspace(np.log10(0.03 * MJUP), np.log10(80 * MJUP), 27) if mbins is None
             else np.asarray(mbins, float))
    pc, mc_out = [], []
    for lo, hi in zip(pbins[:-1], pbins[1:]):
        sel = (P >= lo) & (P < hi)
        pc.append(np.sqrt(lo * hi))
        f, mc = [], []
        for mlo, mhi in zip(mbins[:-1], mbins[1:]):
            cell = sel & (M >= mlo) & (M < mhi)
            if cell.sum() >= nmin:
                f.append(flag[cell].mean())
                mc.append(np.sqrt(mlo * mhi))
        f, mc = np.asarray(f), np.asarray(mc)
        if len(f) < 4 or f.max() < frac:
            mc_out.append(np.nan)
            continue
        j = int(np.argmax(f >= frac))
        if j == 0:
            mc_out.append(mc[0])
            continue
        w = (frac - f[j - 1]) / (f[j] - f[j - 1])
        mc_out.append(10 ** (np.log10(mc[j - 1]) + w * (np.log10(mc[j]) - np.log10(mc[j - 1]))))
    return np.asarray(pc), np.asarray(mc_out)


def companions(df, npl, ax_x, ax_y, classify):
    """Pool (x, y, class, companion index) over every companion in one panel."""
    X, Y, Cl, Kk = [], [], [], []
    for k in ([1, 2] if npl == 2 else [1]):
        cx, cy = ax_x["col"].format(k=k), ax_y["col"].format(k=k)
        d = df.dropna(subset=[cx, cy, f"period_{k}"])
        X.append(d[cx].to_numpy(float) * ax_x["factor"])
        Y.append(d[cy].to_numpy(float) * ax_y["factor"])
        Cl.append(classify(d, k))
        Kk.append(np.full(len(d), k))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Cl), np.concatenate(Kk)


# --------------------------------------------------------------------------
# drawing the loci
# --------------------------------------------------------------------------
SNR_LABEL_ALONG, SNR_LABEL_CORNER = "along", "corner"


def _label_along(ax, x, y, text, along=0.2, fontsize=13, color="0.1", perp=0.0):
    """Write ``text`` on the curve, along its screen slope, ``along`` of the way from where
    the curve enters the axes to its minimum; ``perp`` lifts it off the curve in points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ylo, yhi = ax.get_ylim()
    ok = np.isfinite(y) & (y > ylo * 2.0) & (y < yhi / 2.0)
    if ok.sum() < 3:
        return
    idx = np.flatnonzero(ok)
    i0, imin = idx[0], int(np.nanargmin(np.where(ok, y, np.inf)))
    j = int(round(i0 + along * (imin - i0))) if imin > i0 else idx[len(idx) // 2]
    j = int(np.clip(j, max(idx[0], 1), min(idx[-1], len(x) - 2)))
    (x0, y0), (x1, y1) = ax.transData.transform([(x[j - 1], y[j - 1]), (x[j + 1], y[j + 1])])
    angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    transform = ax.transData
    if perp:
        from matplotlib.transforms import offset_copy

        rad = np.radians(angle)
        transform = offset_copy(ax.transData, fig=ax.figure, units="points",
                                x=-np.sin(rad) * perp, y=np.cos(rad) * perp)
    ax.text(x[j], y[j], text, rotation=angle, rotation_mode="anchor", ha="center", va="bottom",
            fontsize=fontsize, color=color, zorder=9, fontweight="bold", transform=transform,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8, alpha=0.92))


def draw_snr_lines(ax, df, levels, ax_x, ax_y, color=ACCENT, lw=2.4, alpha=0.9,
                   along=None, fontsize=13, perp=0.0, mode=SNR_LABEL_ALONG,
                   corner=(0.025, 0.965), corner_dy=0.075, corner_anchor=0.46):
    """Overlay constant-snr_total loci on a period-vs-mass axes. Call AFTER the limits are
    set. ``mode="corner"`` stacks the labels in a corner with an arrow out to each curve."""
    if not levels or not (ax_x["col"].startswith("period") and ax_y["col"].startswith("Mp_")):
        return
    levels = np.atleast_1d(levels)
    pgrid = np.logspace(np.log10(ax_x["lim"][0]), np.log10(ax_x["lim"][1]), 240)
    if mode == SNR_LABEL_CORNER:
        curves = []
        for s in levels:
            mloc = snr_locus(df, float(s), pgrid) * ax_y["factor"]
            ax.plot(pgrid, mloc, color=color, lw=lw, ls="-", alpha=alpha, zorder=8)
            curves.append(mloc)
        ylo, yhi = ax.get_ylim()
        vis = [np.flatnonzero(np.isfinite(m) & (m > ylo) & (m < yhi)) for m in curves]
        if not all(v.size for v in vis):
            return
        j0 = max(int(v[0]) for v in vis)
        jv = int(np.nanargmin(np.where(np.isfinite(curves[0]), curves[0], np.inf)))
        j = int(np.clip(j0 + round(corner_anchor * max(jv - j0, 1)), j0, len(pgrid) - 1))
        anchors = [(pgrid[j], m[j]) for m in curves]
        for i, (s, (ax_, ay_)) in enumerate(sorted(zip(levels, anchors), key=lambda t: -t[0])):
            if not np.isfinite(ay_):
                continue
            ax.annotate(rf"$\mathrm{{SNR}}_{{\rm tot}} = {s:g}$", xy=(ax_, ay_), xycoords="data",
                        xytext=(corner[0], corner[1] - i * corner_dy), textcoords="axes fraction",
                        ha="left", va="top", fontsize=fontsize, color=color, fontweight="bold",
                        zorder=9,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8,
                                  alpha=0.92),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=3,
                                        shrinkB=2, alpha=0.9))
        return
    if along is None:
        along = 0.18 + 0.22 * np.arange(len(levels))
    along = np.broadcast_to(np.atleast_1d(along), levels.shape)
    perp = np.broadcast_to(np.atleast_1d(perp).astype(float), levels.shape)
    for s, al, pp in zip(levels, along, perp):
        mloc = snr_locus(df, float(s), pgrid) * ax_y["factor"]
        ax.plot(pgrid, mloc, color=color, lw=lw, ls="-", alpha=alpha, zorder=8)
        _label_along(ax, pgrid, mloc, rf"$\mathrm{{SNR}}_{{\rm tot}} = {s:g}$", along=float(al),
                     fontsize=fontsize, color=color, perp=float(pp))


# --------------------------------------------------------------------------
# the joint mass-period figure: scatter + stacked class histograms on both margins
# --------------------------------------------------------------------------
def joint(specs, ax_x, ax_y, stem, *, frame, classify, include_undet=False, name_tag="",
          figsize=(17, 12.5), save_fig=True, snr_lines=None, snr_along=None, snr_fontsize=None,
          snr_perp=0.0, fontscale=1.0, show_undet_fraction=False, marginal_log=False,
          snr_mode=SNR_LABEL_ALONG, snr_corner=(0.025, 0.965), snr_corner_dy=0.075,
          snr_corner_anchor=0.46, paper=False):
    """Grid of joint plots, one per ``(key, n_companions, title)`` in ``specs``.

    ``frame(spec)`` returns the DataFrame a panel draws (the markers) and ``classify(df, k)``
    its class keys. ``include_undet`` stacks the null class into the marginals as well.
    ``snr_lines`` draws constant-snr_total loci on each panel's own stars. ``paper=True``
    saves into ``paper_figures/`` instead of ``supplementary_figures/``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    fs = float(fontscale)
    order, legend, short = DRAW_ORDER, LEGEND_ORDER, SHORT_3
    stack = order if include_undet else [k for k in order if k != "undet"]
    nrow = int(np.ceil(len(specs) / 2))
    ncol_panels = 2 if len(specs) > 1 else 1
    fig = plt.figure(figsize=(figsize[0], figsize[1] * nrow / 2))
    outer = GridSpec(nrow, ncol_panels, figure=fig, hspace=0.26, wspace=0.16,
                     left=0.06, right=0.985, top=0.90, bottom=0.06)
    for idx, spec in enumerate(specs):
        pop, npl, title = spec
        r, c = divmod(idx, ncol_panels)
        inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[r, c], width_ratios=[4, 1],
                                        height_ratios=[4, 1], hspace=0.04, wspace=0.04)
        axm = fig.add_subplot(inner[0, 0])
        axr = fig.add_subplot(inner[0, 1], sharey=axm)
        axb = fig.add_subplot(inner[1, 0], sharex=axm)
        X, Y, Cl, Kk = companions(frame(spec), npl, ax_x, ax_y, classify)
        for key in order:
            m = Cl == key
            for kk, mk, sz in [(1, "s", marker_size(figsize[0])),
                               (2, "^", marker_size(figsize[0], tri=True))]:
                mm = m & (Kk == kk)
                axm.scatter(X[mm], Y[mm], marker=mk, s=sz, c=CLASS_COLORS[key],
                            alpha=0.45 if key == "undet" else 0.7, linewidths=0, rasterized=True)
        axm.set(xscale=ax_x["scale"], yscale=ax_y["scale"], xlim=ax_x["lim"], ylim=ax_y["lim"])
        draw_snr_lines(axm, frame(spec), snr_lines, ax_x, ax_y, along=snr_along, perp=snr_perp,
                       fontsize=13 if snr_fontsize is None else snr_fontsize, mode=snr_mode,
                       corner=snr_corner, corner_dy=snr_corner_dy, corner_anchor=snr_corner_anchor)
        if ax_x["line"] is not None:
            axm.axvline(ax_x["line"], color="k", ls="--", lw=1.6)
            axb.axvline(ax_x["line"], color="k", ls="--", lw=1.6)
        if ax_y["line"] is not None:
            axm.axhline(ax_y["line"], color="k", ls="--", lw=1.6)
            axr.axhline(ax_y["line"], color="k", ls="--", lw=1.6)
        axm.set_title(title, fontsize=19 * fs, pad=8)
        axm.tick_params(labelbottom=False, labelsize=12 * fs)
        axm.set_ylabel(ax_y["label"], fontsize=15 * fs)
        axm.grid(True, which="both", alpha=0.12)
        axb.hist([X[Cl == k] for k in stack], bins=ax_x["bins"], stacked=True,
                 color=[CLASS_COLORS[k] for k in stack], edgecolor="none")
        axb.set(xscale=ax_x["scale"], xlim=ax_x["lim"])
        axb.set_xlabel(ax_x["label"], fontsize=15 * fs)
        axb.tick_params(labelsize=11 * fs)
        axb.set_ylabel("N", fontsize=12 * fs)
        axr.hist([Y[Cl == k] for k in stack], bins=ax_y["bins"], stacked=True,
                 orientation="horizontal", color=[CLASS_COLORS[k] for k in stack], edgecolor="none")
        axr.set(yscale=ax_y["scale"], ylim=ax_y["lim"])
        axr.tick_params(labelleft=False, labelsize=10 * fs)
        if marginal_log:
            axb.set_yscale("log"); axb.set_ylim(bottom=0.5)
            axr.set_xscale("log"); axr.set_xlim(left=0.5)
        if not include_undet and show_undet_fraction:
            axc = fig.add_subplot(inner[1, 1]); axc.axis("off")
            axc.text(0.5, 0.60, f"{100 * np.mean(Cl == 'undet'):.0f}\\%", ha="center", va="center",
                     fontsize=17 * fs, fontweight="bold", color="0.35")
            axc.text(0.5, 0.18, f"{short['undet']}\n(of companions)", ha="center", va="center",
                     fontsize=9.5 * fs, color="0.35")
    marks = [Line2D([0], [0], marker="s", color="0.35", lw=0, ms=9, label="inner"),
             Line2D([0], [0], marker="^", color="0.35", lw=0, ms=10, label="outer")]
    if ax_x["line"] is not None or ax_y["line"] is not None:
        marks.append(Line2D([0], [0], color="k", ls="--", lw=1.6, label=r"\textit{Gaia} DR4 baseline"))
    if snr_lines and ax_x["col"].startswith("period") and ax_y["col"].startswith("Mp_"):
        marks.append(Line2D([0], [0], color=ACCENT, lw=2.4, alpha=0.9,
                            label=r"constant $\mathrm{SNR}_{\rm tot}$ (median over stars)"))
    classes_h = [Line2D([0], [0], marker="s", color=CLASS_COLORS[k], lw=0, ms=12, label=short[k])
                 for k in legend]
    fig.legend(handles=marks, loc="upper center", ncol=len(marks), frameon=False,
               fontsize=13 * fs, bbox_to_anchor=(0.5, 0.995))
    fig.legend(handles=classes_h, loc="upper center", ncol=len(classes_h), frameon=False,
               fontsize=13 * fs, bbox_to_anchor=(0.5, 0.964))
    if save_fig:
        save(fig, f"{stem}{name_tag}{'_wundet' if include_undet else '_noundet'}", dpi=130,
             paper=paper)
    return fig


def census(specs, *, frame, classify):
    """Per-population class fractions [%] of companions, as a dict of dicts."""
    out = {}
    for spec in specs:
        pop, npl, _ = spec
        _, _, Cl, _ = companions(frame(spec), npl, period_axis(), AX_M, classify)
        out[pop] = {k: 100.0 * float(np.mean(Cl == k)) for k in DRAW_ORDER}
    return out


# --------------------------------------------------------------------------
# the gallery: nine periodograms placed on the map
# --------------------------------------------------------------------------
# Where to look for each example: {class: [(period [yr], mass [M_Jup][, {col: value}])]}.
# The nearest system OF THAT CLASS in the log-log plane is taken. Two of the three broad
# panels are pinned to a significant periodogram peak and the third to the acceleration
# channel alone, so the row shows both ways a system ends up "significant but not localized".
EXAMPLE_ANCHORS = {
    "narrow": [(2.43, 15.8), (1.4, 5.0), (2.59, 42.1)],
    "broad": [(7.0, 10.0, {"peak_significant_cal": True}),
              (25.0, 30.0, {"peak_significant_cal": True}),
              (400.0, 60.0, {"peak_significant_cal": False, "accel_significant_cal": True})],
    "undet": [(0.1, 0.01), (3.0, 0.5), (500.0, 20.0)],
}


def _flat_anchors(anchors):
    def norm(pt, key):
        p, m, extra = (list(pt) + [None])[:3]
        return (p, m, key, extra if isinstance(extra, dict) else None)
    rows = [(key, len(pts)) for key, pts in anchors.items()]
    return rows, [norm(pt, key) for key, pts in anchors.items() for pt in pts]


def select_examples(df, anchors=EXAMPLE_ANCHORS, *, classify, k=1):
    """One system per anchor: nearest in (log P, log M) inside the anchor's class."""
    _, flat = _flat_anchors(anchors)
    d = df.dropna(subset=[f"period_{k}", f"Mp_{k}_msun"]).copy()
    d["cls"] = classify(d, k)
    lp = np.log10(d[f"period_{k}"].to_numpy(float))
    lm = np.log10(d[f"Mp_{k}_msun"].to_numpy(float) / MJUP)
    picked, taken = [], set()
    for i, (p_a, m_a, want, require) in enumerate(flat):
        cost = (lp - np.log10(p_a)) ** 2 + (lm - np.log10(m_a)) ** 2
        if want is not None:
            cost = np.where(d["cls"].to_numpy(object) == want, cost, np.inf)
        for col, val in (require or {}).items():
            cost = np.where(d[col].to_numpy() == val, cost, np.inf)
        for j in taken:
            cost[j] = np.inf
        if not np.isfinite(cost).any():
            raise ValueError(f"no system left for anchor {(p_a, m_a, want, require)}")
        j = int(np.argmin(cost))
        taken.add(j)
        row = d.iloc[j].copy()
        row["tag"] = chr(ord("A") + i)
        picked.append(row)
    return pd.DataFrame(picked).reset_index(drop=True)


def _line_color(cls):
    return "0.45" if cls == "undet" else CLASS_COLORS[cls]


def _param_title(row, k=1):
    """The injected parameters of one example, as a three-line panel title."""
    g = lambda v, n=2: f"{float(v):.{n}g}"  # noqa: E731
    return (rf"$M_{{\rm c}} = {g(row[f'Mp_{k}_msun'] / MJUP)}\,M_{{\rm Jup}}$, "
            rf"$M_\star = {g(row['mass_st_msun'])}\,M_\odot$" "\n"
            rf"$P = {g(row[f'period_{k}'], 3)}$ yr, "
            rf"$d = {g(1000.0 / row['parallax_mas'], 3)}$ pc" "\n"
            rf"$e = {g(row[f'e_{k}'])}$, "
            rf"$i = {g(np.degrees(row[f'i_{k}_rad']), 3)}^\circ$")


def _draw_periodogram(ax, periods, power, row, color, thr_orbit, baseline, k=1, fs=1.0):
    ax.plot(periods, np.clip(power, 1e-3, None), lw=0.9 * fs, color=color, zorder=3)
    ax.axvline(float(row[f"period_{k}"]), color="#2e6f95", ls="-", lw=1.4 * fs, zorder=2)
    ax.axvline(baseline, color="k", ls="--", lw=1.2 * fs, zorder=2)
    if thr_orbit is not None:
        ax.axhline(thr_orbit, color="#CC6677", ls=":", lw=1.4 * fs, zorder=2)
    top = max(float(np.nanmax(power)), 1.0 if thr_orbit is None else float(thr_orbit))
    pos = power[np.isfinite(power) & (power > 0)]
    bot = min(1.0, 0.8 * float(np.nanmin(pos))) if pos.size else 1.0
    ax.set(xscale="log", yscale="log", xlim=(periods[0], periods[-1]),
           ylim=(max(bot, 1e-2), 6.0 * top))
    ax.tick_params(labelsize=10 * fs, pad=2)
    ax.grid(True, which="major", alpha=0.12)


def examples(df, population_title, *, classify, periodogram, epochs_for, ax_x, ax_y=AX_M,
             anchors=EXAMPLE_ANCHORS, k=1, thr_orbit=None, baseline=BASELINE,
             stem="characterizability_examples", name_tag="", figsize=(18.0, 10.5),
             save_fig=True, verbose=True, snr_lines=None, snr_along=None, snr_fontsize=None,
             snr_perp=0.0, map_df=None, snr_mode=SNR_LABEL_ALONG, snr_corner=(0.025, 0.965),
             snr_corner_dy=0.075, snr_corner_anchor=0.46, paper=False):
    """The population map beside the periodograms of nine systems drawn from it.

    ``df`` holds the candidate systems (the examples are picked from it), ``map_df`` the
    population drawn behind them (``df`` if None). ``epochs_for(row) -> (t, psi, pf, y, yerr)``
    fetches a system's epochs and ``periodogram(t, psi, pf, y, yerr) -> (periods, power)``
    runs the search -- the one the catalog was characterized with. ``paper=True`` saves into
    ``paper_figures/`` instead of ``supplementary_figures/``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    fs = float(figsize[0]) / MARKER_REF_WIDTH_IN
    rows, _ = _flat_anchors(anchors)
    picks = select_examples(df, anchors, classify=classify, k=k)
    nrow, ncol = len(rows), max(n for _, n in rows)
    fig = plt.figure(figsize=figsize)
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.5], wspace=0.17,
                     left=0.058, right=0.995, top=0.875, bottom=0.075)

    axm = fig.add_subplot(outer[0, 0])
    cx, cy = ax_x["col"].format(k=k), ax_y["col"].format(k=k)
    d = (df if map_df is None else map_df).dropna(subset=[cx, cy, f"period_{k}"])
    Cl = classify(d, k)
    X = d[cx].to_numpy(float) * ax_x["factor"]
    Y = d[cy].to_numpy(float) * ax_y["factor"]
    for key in DRAW_ORDER:
        m = Cl == key
        axm.scatter(X[m], Y[m], marker="s", s=marker_size(figsize[0]), c=CLASS_COLORS[key],
                    linewidths=0, alpha=0.4 if key == "undet" else 0.75, rasterized=True)
    if ax_x["line"] is not None:
        axm.axvline(ax_x["line"], color="k", ls="--", lw=1.6 * fs)
    axm.set(xscale=ax_x["scale"], yscale=ax_y["scale"], xlim=ax_x["lim"], ylim=ax_y["lim"])
    draw_snr_lines(axm, d, snr_lines, ax_x, ax_y, lw=2.4 * fs, along=snr_along, perp=snr_perp,
                   fontsize=13 * fs if snr_fontsize is None else snr_fontsize, mode=snr_mode,
                   corner=snr_corner, corner_dy=snr_corner_dy, corner_anchor=snr_corner_anchor)
    axm.set_xlabel(ax_x["label"], fontsize=15 * fs)
    axm.set_ylabel(ax_y["label"], fontsize=15 * fs)
    axm.tick_params(labelsize=12 * fs)
    axm.grid(True, which="both", alpha=0.12)
    axm.set_title(population_title, fontsize=17 * fs, pad=8)
    for _, row in picks.iterrows():
        px = float(row[cx]) * ax_x["factor"]
        py = float(row[cy]) * ax_y["factor"]
        badge = _line_color(row["cls"])
        axm.annotate(row["tag"], (px, py), ha="center", va="center", fontsize=15 * fs,
                     fontweight="bold", color=badge, zorder=8.5,
                     bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec=badge, lw=1.4 * fs,
                               alpha=0.95))
    handles = [Line2D([0], [0], marker="s", color=CLASS_COLORS[key], lw=0, ms=11 * fs,
                      label=SHORT_3[key]) for key in LEGEND_ORDER]
    handles.append(Line2D([0], [0], color="k", ls="--", lw=1.6 * fs, label=r"\textit{Gaia} DR4 baseline"))
    if snr_lines:
        handles.append(Line2D([0], [0], color=ACCENT, lw=2.4 * fs, alpha=0.9,
                              label=r"constant $\mathrm{SNR}_{\rm tot}$ (median over stars)"))
    axm.legend(handles=handles, loc="lower left", frameon=True, fontsize=10.5 * fs,
               handletextpad=0.5, borderaxespad=0.3, fancybox=False, framealpha=0.92,
               facecolor="white", edgecolor="0.6")

    grid = GridSpecFromSubplotSpec(nrow, ncol, subplot_spec=outer[0, 1], hspace=0.85, wspace=0.34)
    i = 0
    for r, (key, npts) in enumerate(rows):
        axes_r = []
        for c in range(npts):
            row = picks.iloc[i]
            t, psi, pf, y, yerr = epochs_for(row)
            periods, power = periodogram(t, psi, pf, y, yerr)
            color = _line_color(row["cls"])
            ax = fig.add_subplot(grid[r, c])
            _draw_periodogram(ax, periods, power, row, color, thr_orbit, baseline, k, fs=fs)
            ax.set_title(_param_title(row, k), fontsize=10.5 * fs, linespacing=1.5, pad=5)
            ax.text(0.045, 0.90, row["tag"], transform=ax.transAxes, fontsize=14 * fs,
                    fontweight="bold", va="top", ha="left", color=color, zorder=9,
                    bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec=color, lw=1.2 * fs,
                              alpha=0.95))
            if bool(row.get("accel_significant_cal", False)) and \
                    not bool(row.get("peak_significant_cal", True)):
                ax.text(0.97, 0.05, "significant via acceleration", ha="right", va="bottom",
                        transform=ax.transAxes, fontsize=9.5 * fs, color=color)
            ax.set_xlabel("period [yr]", fontsize=12 * fs, labelpad=2)
            if c == 0:
                ax.set_ylabel(r"$\Delta\chi^2$", fontsize=13 * fs)
            axes_r.append(ax)
            if verbose:
                print(f"  {row['tag']}  {int(row['gaia_source_id']):20d}  "
                      f"P = {row[f'period_{k}']:9.3f} yr  M = {row[f'Mp_{k}_msun'] / MJUP:7.3f} MJup  "
                      f"{row['cls']}  (peak {float(np.nanmax(power)):.1f})")
            i += 1
        if key is not None:
            box = axes_r[0].get_position()
            fig.text(box.x0 - 0.060, box.y0 + 0.5 * box.height, ROW_LABELS.get(key, key),
                     rotation=90, va="center", ha="center", fontsize=13 * fs, fontweight="bold",
                     color=_line_color(key))
    keys = [Line2D([0], [0], color="#2e6f95", lw=1.8 * fs, label="injected period"),
            Line2D([0], [0], color="k", ls="--", lw=1.4 * fs, label=r"\textit{Gaia} DR4 baseline")]
    if thr_orbit is not None:
        keys.append(Line2D([0], [0], color="#CC6677", ls=":", lw=1.6 * fs,
                           label="null-calibrated detection threshold"))
    right = outer[0, 1].get_position(fig)
    fig.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5 * (right.x0 + right.x1), 0.995),
               ncol=len(keys), frameon=False, fontsize=11.0 * fs, handletextpad=0.5,
               columnspacing=1.6)
    if save_fig:
        save(fig, f"{stem}{name_tag}", dpi=130, paper=paper)
    return fig, picks


# ==========================================================================
# Epochs off the server, one system at a time
# ==========================================================================
# The epoch shards are 28.6 GB and are not mirrored. A system's epochs are a few
# hundred rows of one ~25 MB shard, addressed by the (shard, shard_row) the
# characterization tables carry, so they are read over HTTP range requests --
# only the row groups that hold the system -- and cached locally.
EPOCH_COLUMNS = ["obs_time_tcb", "scan_pos_angle", "parallax_factor_al",
                 "centroid_pos_al", "centroid_pos_error_al"]


def _http_fs():
    import fsspec

    return fsspec.filesystem("https", block_size=4 * 2**20)


def shard_url(population, kind, shard, n_shards=N_SHARDS):
    """URL of one epochs/truths shard on the server."""
    return (f"{SERVER}/data/simulated_astrometry/{population}/"
            f"{kind}_rank{shard:05d}_of_{n_shards:05d}.parquet")


def fetch_epochs(population, shard, gaia_source_id, n_shards=N_SHARDS, verbose=True):
    """One system's epoch table (sorted by time), from the server or the local cache."""
    import pyarrow.parquet as pq

    cache_dir = CACHE / "epochs" / population
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{int(gaia_source_id)}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    target = str(int(gaia_source_id))
    url = shard_url(population, "epochs", shard, n_shards)
    if verbose:
        print(f"  fetching epochs of {target} from shard {shard:05d} ...", flush=True)
    with _http_fs().open(url) as handle:
        pf = pq.ParquetFile(handle)
        pieces = []
        for group in range(pf.num_row_groups):
            ids = pf.read_row_group(group, columns=["gaia_source_id"]).column(0)
            ids = ids.to_numpy(zero_copy_only=False).astype(str)
            hit = np.flatnonzero(ids == target)
            if hit.size:
                block = pf.read_row_group(group, columns=EPOCH_COLUMNS).to_pandas()
                pieces.append(block.iloc[hit])
    if not pieces:
        raise KeyError(f"{target} is not in {url}")
    epochs = pd.concat(pieces).sort_values("obs_time_tcb").reset_index(drop=True)
    epochs.to_parquet(cached, index=False)
    return epochs


def epoch_arrays(epochs):
    """(t [yr from the DR4 reference epoch], psi [rad], parallax factor, y, yerr [mas])."""
    from epochalypse.periodogram.fitting import epoch_arrays as _arrays

    return _arrays(*[epochs[c].to_numpy(float) for c in EPOCH_COLUMNS])


def epochs_for_row(population):
    """``epochs_for`` for :func:`examples`: reads the (shard, gaia_source_id) off a row."""
    def _get(row):
        return epoch_arrays(fetch_epochs(population, int(row["shard"]), int(row["gaia_source_id"])))
    return _get


def kepmodel_search():
    """``(periods, periodogram)``: the catalog's trial grid and the kepmodel search on it,
    checked against the run's manifest so a curve drawn here is the one that classified
    the system."""
    from epochalypse.periodogram.grid import frequency_segments, segment_periods
    from epochalypse.periodogram.periodogram import kepmodel_power

    segments = frequency_segments()
    periods = segment_periods(segments)
    grid = manifest()["grid"]
    assert len(periods) == grid["n_periods"], (
        f"this checkout's grid has {len(periods)} trial periods, the run used {grid['n_periods']}")
    stored = pd.read_parquet(PC.period_grid_path())["period_yr"].to_numpy(float)
    assert np.allclose(stored, periods, rtol=1e-10), "period_grid.parquet differs from the grid"

    def periodogram(t, psi, pf, y, yerr):
        p, power, _ = kepmodel_power(t, psi, pf, y, yerr, segments=segments)
        return p, power

    return periods, periodogram


# ==========================================================================
# The five-parameter astrometric fit (for the DR4 prerelease comparison)
# ==========================================================================
def weighted_linear_solution(t, scan_angle, pf, values, errors):
    """Weighted LSQ of the five-parameter astrometric model. Returns (beta, cov, X)."""
    from epochalypse.periodogram.fitting import astrometric_design_matrix

    X = astrometric_design_matrix(t, scan_angle, pf)
    w = 1.0 / np.square(errors)
    normal = X.T @ (w[:, None] * X)
    covariance = np.linalg.pinv(normal)
    beta = covariance @ (X.T @ (w * values))
    return beta, covariance, X


def sigma_reported_cache(population="0_companion", n_shards=N_SHARDS, rebuild=False, workers=8):
    """Per-star median of the reported per-epoch uncertainty, read off the epoch shards.

    The catalog writes ``sigma_reported_i = (1 + 0.1 xi_i) sigma_formal`` per epoch; the
    figure compares DR4 against what a user reads off the epoch tables, so the median over
    each star's epochs is taken from the shards themselves (two columns of each, streamed
    from the server) rather than recomputed from the noise-model ingredients.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq

    path = CACHE / f"sigma_reported_median_{population}.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    # One part file per shard, so an interrupted build resumes where it stopped; the
    # reads are HTTP-latency bound, so several run at once.
    part_dir = CACHE / f"sigma_reported_parts_{population}"
    part_dir.mkdir(parents=True, exist_ok=True)
    fs = _http_fs()

    def one(shard):
        part = part_dir / f"shard{shard:05d}.parquet"
        if part.exists():
            return part
        url = shard_url(population, "epochs", shard, n_shards)
        with fs.open(url) as handle:
            table = pq.ParquetFile(handle).read(columns=["gaia_source_id", "centroid_pos_error_al"])
        d = table.to_pandas()
        d["gaia_source_id"] = d["gaia_source_id"].astype("int64")
        g = d.groupby("gaia_source_id")["centroid_pos_error_al"]
        out = pd.DataFrame({"sigma_reported_med": g.median(), "n_epochs": g.size()}).reset_index()
        out.to_parquet(part.with_suffix(".tmp"), index=False)
        part.with_suffix(".tmp").replace(part)
        return part

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(one, range(n_shards))):
            if i % 40 == 0:
                print(f"  shard {i:03d} of {n_shards} ...", flush=True)
    out = pd.concat([pd.read_parquet(part_dir / f"shard{s:05d}.parquet") for s in range(n_shards)],
                    ignore_index=True)
    out.to_parquet(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}: {len(out):,} stars")
    return out
