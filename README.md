# epochalypse

Gaia's fourth data release (DR4, expected December 2026) will for the first time include epoch
astrometry: the individual along-scan measurements of each star, rather than a single fitted solution.
This repository is a test bed for the orbit-modeling tools being built for that release. It holds the
`epochalypse` catalog: simulated Gaia DR4-like epoch astrometry for substellar companions injected onto
real Gaia stars, with every injected parameter known exactly, so that any pipeline can be run on it and
scored against the truth.

The catalog is described in *Preparing for the Gaia DR4 Epochalypse: Simulated Epoch Astrometry for
Substellar Companion Orbit-Modeling Benchmarks* (Yahalomi et al. 2026, submitted). This repository holds
the code that generated it, the periodogram analysis of every system, an orbit-fitting tutorial, and the
notebook behind each figure in the paper.

## What is in the catalog

Onto each of the 5,040,328 main-sequence Gaia stars within 500 pc in the G23H catalog, we inject a zero-,
one-, and two-companions system, for 15,120,984 systems in total. Companion masses and semimajor axes are
drawn log-uniformly over 1 M_Mars to 80 M_Jup and 0.03 to 100 au, with no occurrence model and no
detectability cut. Orbital periods therefore extend well beyond the DR4 baseline, and the companion-free
population serves as a matched null for the other two. The per-epoch noise follows the Gaia DR3 noise
model of Kiefer et al. (2025) and the epochs follow the predicted DR4 scanning law.

For users who want only companions expected to be detectable, we also flag high-SNR sub-populations of
the one- and two-companions systems: the 439,503 and 26,379 systems whose every companion has an
estimated total signal-to-noise ratio of at least 5.

For every system the catalog provides:

* an **epoch table**, one row per Gaia field-of-view transit, with the observation time, the along-scan
  position and its reported uncertainty, the parallax factor, the scan angle, and the field of view, in
  the format expected for Gaia DR4;
* an **injected-solutions table**, one row per system, with the star's parallax, proper motion, mass and
  radius, each companion's orbital elements and mass, the signal-to-noise estimates, and the random
  seeds that produced the system;
* a **periodogram characterization table**, the result of running the `kepmodel` astrometric periodogram
  on the system, with thresholds calibrated on the companion-free population.

The tables are plain Parquet files with flat column types and can be read with any Parquet library.
Section 6 and Tables 1 and 2 of the paper describe every column.

## Getting the catalog

The catalog is archived on Zenodo at <https://doi.org/10.5281/zenodo.23789041>. The script
[`catalog/download_catalog.sh`](catalog/download_catalog.sh) fetches it into `catalog/` in the layout the
rest of this repository expects. It can be rerun to finish an interrupted download.

```bash
cd catalog
./download_catalog.sh                        # the tables, about 13.5 GB
./download_catalog.sh --shards 1_companion   # plus the epoch tables of one population, about 10 GB
./download_catalog.sh --shards all           # all three populations, about 29 GB unpacked
```

The first command is enough for most of the figure notebooks. The epoch tables are split into 320 files
per population (this reflects how the simulation was run and carries no physical meaning). The tutorial
needs the one- and two-companions epochs; Figure 6 needs the one-companion epochs and Figure 2 the
zero-companions epochs. Nothing in this repository reads the catalog over the network: a notebook that
cannot find what it needs under `catalog/` stops and says what to download.

## Start here: the orbit-fitting tutorial

[`tutorials/fit_astrometric_orbits.ipynb`](tutorials/fit_astrometric_orbits.ipynb) is the recommended
entry point. It loads a single simulated system, builds the five-parameter astrometric model and a
`jaxoplanet` Keplerian reflex model, finds a starting period with the same `kepmodel` periodogram used in
the paper (the injected orbit is never used in the fit), and fits zero-, one-, and two-companion models
with NumPyro. The competing models are compared with the WAIC information criterion, and the recovered
period and mass posteriors are plotted against the injected truth. Two worked examples, one with a single
companion and one with two, run in about two minutes on a laptop. See
[`tutorials/README.md`](tutorials/README.md) for setup.

## The paper's figures

[`analysis/`](analysis/) has one notebook per figure, `Fig1_*.ipynb` through `Fig9_*.ipynb`. Each ends
with the figure as it appears in the paper, and the numbers quoted in the text are printed along the way.
The final figures are committed in [`analysis/paper_figures/`](analysis/paper_figures/). Code shared by
the notebooks lives in [`analysis/paperfig.py`](analysis/paperfig.py); the physics (the main-sequence
selection, the noise model, the period grid, the periodogram) is imported from the generator package, so
the figures are made with the same code that made the catalog. See
[`analysis/README.md`](analysis/README.md) for inputs and run times.

## Generating the catalog yourself

The generator is the `epochalypse` package in [`src/`](src/epochalypse/), driven by the scripts in
[`scripts/`](scripts/). Once Gaia DR4 is public, the same companion populations can be injected onto the
real DR4 epoch astrometry and noise with this code.

* [`PIPELINE.md`](PIPELINE.md) describes how the parent stellar sample is built, how the companions are
  drawn, how the epoch astrometry is simulated, and how to run it on a laptop or a cluster.
* [`PERIODOGRAMS.md`](PERIODOGRAMS.md) describes the `kepmodel` period search, how its detection
  thresholds are set on the companion-free population, and the tables it writes.

Every random draw is seeded from the Gaia source id, so any system can be regenerated exactly, on its
own, in any order.

## Layout

```
src/epochalypse/   the generator and periodogram package
scripts/           catalog generation and periodogram search (laptop or cluster)
tests/             self-checks that need only the small inputs committed here
data/              small committed inputs (Pecaut & Mamajek table, scanning-law transit map, a sample subset)
catalog/           the catalog, fetched by download_catalog.sh (not tracked by git)
tutorials/         the orbit-fitting tutorial
analysis/          one notebook per paper figure, paperfig.py, paper_figures/, external inputs in data/
```

## Environment

Dependencies are pinned in `uv.lock`, which records the package versions the catalog was generated with:

```bash
uv sync --frozen
uv run python tests/test_pipeline.py
```

The notebooks need a few extra packages (Jupyter, NumPyro, corner) and a registered kernel; the commands
are in [`analysis/README.md`](analysis/README.md). The figures use LaTeX text rendering, so a TeX
installation with Computer Modern is required to reproduce them.

## Citation

If you use the catalog, please cite the paper (Yahalomi et al. 2026, submitted; arXiv link to follow) and
the Zenodo record, <https://doi.org/10.5281/zenodo.23789041>.

Two external datasets are included for the comparisons in Figures 8 and 9: the Gaia DR4 forecast of
Lammers & Winn (2026), under its MIT license, in `analysis/data/LW25_GaiaForecasts/`, and the Gaia DR4
prerelease epoch astrometry from ESA in `analysis/data/dr4_prerelease/` (see the `readme.txt` there for
credit and license).

## Team

The `epochalypse` catalog was built by a collaboration working on analysis tools for Gaia DR4. The
project started at the "Tooling up for Gaia DR4" workshop at the Flatiron Institute, Center for
Computational Astrophysics, in April 2026.
