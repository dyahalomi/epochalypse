# The paper's figures, one notebook each

Every figure of *Preparing for the Gaia DR4 Epochalypse* is produced by one notebook here, whose
**last cell is the figure as it goes in the paper** (earlier cells show the inputs, the numbers the
text quotes, and intermediate versions worth looking at). The notebooks read the regenerated
catalog mirrored in `../catalog/` and write the final figures to `paper_figures/`; every other
figure a notebook draws (an intermediate step, an alternative version) goes to
`supplementary_figures/`. In the code, `pf.save(fig, stem, paper=True)` is the former and
`pf.save(fig, stem)` the latter.

| notebook | paper figure | written to `paper_figures/` as |
| --- | --- | --- |
| `Fig1_sky_scanlaw_cmd_main_sequence.ipynb` | Fig. 1: the parent sample over the DR4 scan law + its CMD with the main-sequence band | `star_sky_scanlaw_equatorial_density_cmd_absG_ms_full311` |
| `Fig2_dr4_prerelease_noise.ipynb` | Fig. 2: injected and reported per-epoch uncertainty (medians, 16--84%) vs the DR4 prerelease | `dr4_prerelease_medians_vs_g` |
| `Fig3_population_schematic.ipynb` | Fig. 3: the catalog schematic (funnel, populations, sub-catalogs) | `population_schematic_500pc` |
| `Fig4_mass_period_snr_contours.ipynb` | Fig. 4: companion mass vs period, coloured by $\alpha$, with the SNR$_{\rm tot}$ loci | `simulated_planets_mass_period_snr_contours` |
| `Fig5_population_diagnostics.ipynb` | Fig. 5a/b: the injected prior distributions, all vs high-SNR | `pop_diagnostics_1planet`, `pop_diagnostics_2planet` |
| `Fig6_characterizability_examples.ipynb` | Fig. 6: nine periodograms placed on the map | `characterizability_examples_kepmodel_3class` |
| `Fig7_characterizability_joint.ipynb` | Fig. 7: characterizability across the mass--period plane | `characterizability_joint_kepmodel_3class_noundet` |
| `Fig8_injected_vs_recovered_period.ipynb` | Fig. 8: injected vs recovered period, with LW26 | `injected_vs_recovered_period_lw26` |
| `Fig9_lw26_fig1_comparison.ipynb` | Fig. 9: LW26's Figure 1 with the `epochalypse` populations | `lw25_fig1_with_unlocalized_counts_contours_mixedmargins` |

`paperfig.py` holds everything the notebooks share: paths, the plotting style, the paper-wide
10,000-star subsample rule, the parent-sample cuts, the sky-map helpers, the loaders that put the
truth and characterization tables into the vocabulary the figure code reads, the three-class
periodogram classification with its palette, the constant-SNR$_{\rm tot}$ loci, the joint and
gallery figures, and a reader that fetches one system's epochs off the catalog server. The
physics -- the main-sequence label, the noise model, the trial-period grid and the `kepmodel`
search -- is imported from the generator package in `../src/epochalypse`, not copied, so
the figures are built with the code that built the catalog.

## Inputs

* `../catalog/` -- the server mirror, gitignored (`../catalog/download_catalog.sh` fetches it; see the
  comment there for what is deliberately left out: the 711 GB of raw periodogram curves and the
  28.6 GB of epoch shards). Fig. 6 and Fig. 2 read the few epoch rows they need straight off the
  server, by HTTP range request, and cache them under `cache/`.
* `data/dr4_prerelease/` -- ESA's DR4 prerelease epoch astrometry for 12 sources (Fig. 2).
* `data/LW25_GaiaForecasts/` -- the DR4 files of the Lammers & Winn (2026) release and the
  exoplanet-archive export their Figure 1 notebook used (Figs. 8, 9); MIT-licensed, their
  `LICENSE` and `README.md` are kept alongside.
* The Pecaut & Mamajek table and the GOST FoV-transit map come from the repository's `data/`.
* Fig. 3 reads the row counts of the full `G23H` release and of its 500 pc cut from the original
  inputs when they are on this machine (`paperfig.G23H_FULL`, `paperfig.G23H_500PC`), and uses
  the recorded values otherwise.

`cache/` is rebuilt on demand (the column extract of `stars.csv`, the per-star median of
$\sigma_{\rm reported}$ streamed from the epoch shards, fetched epochs) and can be deleted.

## Running

The notebooks run in the generator's locked environment plus a few extras (remote reads and
Jupyter), set up once from the repository root as

```bash
cd ..
uv sync --frozen                                   # the locked environment, in .venv/
uv pip install --python .venv/bin/python fsspec aiohttp requests jupyterlab ipykernel nbconvert
.venv/bin/python -m ipykernel install --user --name epochalypse-figs \
    --display-name "Python (epochalypse figures)"
```

Every notebook's metadata names that kernel (`epochalypse-figs`), so they run as-is from any
Jupyter (`jupyter lab` in this directory); the machine's default `python3` kernel is a different
environment and will not have `pyarrow`, `healpy` or `kepmodel`. To execute one in place from the
command line:

```bash
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 Fig7_characterizability_joint.ipynb
```

The figures use `usetex`, so a TeX installation with Computer Modern is required; matplotlib
3.11.0 drops the math minus sign from usetex PDFs and must be avoided (the locked 3.11.1 is fine).

Run times on a laptop: Figs. 3, 5, 8, 9 under a minute; Figs. 6, 7 about a minute; Fig. 1 a few
minutes; Fig. 4 (15 M rasterized points) ~6 min; Fig. 2 ~15 min the first time (streaming two
columns of the 320 control-population epoch shards over eight connections, checkpointed per
shard under `cache/sigma_reported_parts_0_companion/`), seconds afterwards.
