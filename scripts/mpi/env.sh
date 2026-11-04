# Shared roots for every stage. Sourced after `cd`ing to the checkout.
#
# Inputs and outputs both live on the cluster's large scratch filesystem: the
# delivered dataset is ~12 GB, the catalog ~50 GB, and the raw periodograms
# ~915 GB. Set SCRATCH to that filesystem (site-specific; the one line to edit).
export SCRATCH=${SCRATCH:-/path/to/scratch}
export DATA_ROOT=$SCRATCH/project-data/epochalypse
export OUT_ROOT=$SCRATCH/project-outputs/epochalypse

# The characterization reads the catalog the generator wrote -- so its
# --catalog-root IS $OUT_ROOT -- and writes its own products beside it.
export PGRAM_ROOT=$OUT_ROOT/periodograms
