# Tutorials

| notebook | what it does |
| --- | --- |
| `fit_astrometric_orbits.ipynb` | Loads one simulated system from the catalog, rebuilds the along-scan five-parameter model plus a `jaxoplanet` Keplerian reflex model, finds a starting orbit for each companion with the catalog's own `kepmodel` periodogram search (the injected orbit is never used in the fit), samples 0-, 1- and 2-companion models by NUTS in NumPyro, compares them with WAIC, and plots the recovered orbits against the injected truth. Ships two worked examples, a one-companion and a two-companion high-SNR system, selected by a rule from the injected-solutions tables rather than by index. |

## Inputs

The notebook reads the catalog mirrored in `../catalog/` (gitignored; `../catalog/download_catalog.sh` fetches it):
the injected-solutions tables (`data/injected_solutions_*.parquet`) for the truth and the
characterization tables (`periodograms/characterization_*.parquet`) for the shard that holds each
system's epochs. The epoch shards themselves (28.6 GB) are not mirrored; a system's few hundred
epochs are read straight off the catalog server by HTTP range request and cached under `cache/epochs/`.
If `../catalog/data/simulated_astrometry/` exists locally, it is used instead.

## Running

Same environment as the figure notebooks in `../analysis/` (the generator's locked environment, see
`../analysis/README.md`) plus `numpyro` and `corner`, set up from the repository root:

```bash
cd ..
uv sync --frozen
uv pip install --python .venv/bin/python fsspec aiohttp requests jupyterlab ipykernel nbconvert numpyro corner
.venv/bin/python -m ipykernel install --user --name epochalypse-figs --display-name "Python (epochalypse figures)"
```

The notebook's metadata names that kernel (`epochalypse-figs`). To execute it in place from the
command line:

```bash
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 fit_astrometric_orbits.ipynb
```

With the shipped chain lengths (500 warmup + 500 samples, two chains, three model orders per
example) the whole notebook runs in about two minutes on a laptop CPU.
