#!/usr/bin/env bash
# Fetch the epochalypse catalog from its Zenodo record into this directory, in
# the layout the generator writes (data/, data/index/, periodograms/). Resumable
# (curl -C -): rerun to finish an interrupted copy.
#
#   ./download_catalog.sh                          the tables (~13.5 GB): what the
#                                                  figure notebooks and the tutorial
#                                                  read, apart from the epochs
#   ./download_catalog.sh --shards 1_companion     also the epoch shards of that
#                                                  population (a 10 GB tar, unpacked
#                                                  into data/simulated_astrometry/)
#   ./download_catalog.sh --shards all             all three populations (28.6 GB)
#
# The record holds every table flat plus one simulated_astrometry_<pop>.tar per
# population; this script puts each where the notebooks expect it.
set -uo pipefail
RECORD="23789041"           # the Zenodo record number (doi 10.5281/zenodo.23789041)
if ! [[ "$RECORD" =~ ^[0-9]+$ ]]; then
  echo "download_catalog.sh: set RECORD at the top of this script to the Zenodo record number" >&2
  exit 1
fi
BASE="https://zenodo.org/records/$RECORD/files"
cd "$(dirname "$0")"

# served name -> local path
TABLES=(
  "manifest.json periodograms/manifest.json"
  "calibration.json periodograms/calibration.json"
  "period_grid.parquet periodograms/period_grid.parquet"
  "stars_index.parquet data/index/stars_index.parquet"
  "scanlaw_index.parquet data/index/scanlaw_index.parquet"
  "injected_solutions_1_companion_high_snr.parquet data/injected_solutions_1_companion_high_snr.parquet"
  "injected_solutions_2_companion_high_snr.parquet data/injected_solutions_2_companion_high_snr.parquet"
  "characterization_1_companion_high_snr.parquet periodograms/characterization_1_companion_high_snr.parquet"
  "characterization_2_companion_high_snr.parquet periodograms/characterization_2_companion_high_snr.parquet"
  "injected_solutions_0_companion.parquet data/injected_solutions_0_companion.parquet"
  "injected_solutions_1_companion.parquet data/injected_solutions_1_companion.parquet"
  "injected_solutions_2_companion.parquet data/injected_solutions_2_companion.parquet"
  "characterization_0_companion.parquet periodograms/characterization_0_companion.parquet"
  "characterization_1_companion.parquet periodograms/characterization_1_companion.parquet"
  "characterization_2_companion.parquet periodograms/characterization_2_companion.parquet"
  "stars.csv data/stars.csv"
)
SHARDS=()
if [ "${1:-}" = "--shards" ]; then
  shift
  if [ $# -eq 0 ] || [ "$1" = "all" ]; then SHARDS=(0_companion 1_companion 2_companion); else SHARDS=("$@"); fi
fi

fetch() {  # fetch <served name> <local path>: resumed if partial, then size-checked
  local name=$1 dest=$2 want have
  local url="$BASE/$name?download=1"
  want=$(curl -sSI -L "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
  have=$(stat -f %z "$dest" 2>/dev/null || stat -c %s "$dest" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$have" = "$want" ]; then echo "ok    $dest"; return 0; fi
  curl -sS -f -L -C - --retry 5 --retry-all-errors --create-dirs -o "$dest" "$url"
  have=$(stat -f %z "$dest" 2>/dev/null || stat -c %s "$dest" 2>/dev/null || echo 0)
  if [ "$have" = "$want" ]; then echo "done  $dest  ($have bytes)"; return 0; fi
  echo "FAIL  $dest  have=$have want=$want"; return 1
}

status=0
for entry in "${TABLES[@]}"; do
  fetch $entry || status=1
done
for pop in ${SHARDS[@]+"${SHARDS[@]}"}; do   # (empty-array-safe under set -u on bash 3.2)
  tarf="simulated_astrometry_$pop.tar"
  if fetch "$tarf" "$tarf"; then
    echo "unpacking $tarf into data/ ..."
    tar -xf "$tarf" -C data && rm -f "$tarf" && echo "done  data/simulated_astrometry/$pop/"
  else
    status=1
  fi
done
[ $status -eq 0 ] && echo "ALL DOWNLOADS COMPLETE" || echo "SOME DOWNLOADS FAILED (rerun to resume)"
exit $status
