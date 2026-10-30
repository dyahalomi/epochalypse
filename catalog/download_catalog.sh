#!/usr/bin/env bash
# Fetch the parts of the epochalypse catalog the paper's figures read, mirroring
# the server's layout. Resumable (curl -C -): rerun to finish an interrupted copy.
#
# Deliberately NOT fetched: periodograms/power/ (711 GB of raw curves),
# data/simulated_astrometry/ (28.6 GB of epoch shards), harv*/ and detectability/.
# The figure notebooks read the few epoch or power rows they need straight off
# the server.
set -uo pipefail
BASE=https://users.flatironinstitute.org/~apricewhelan/epochalypse
cd "$(dirname "$0")"
FILES=(
  periodograms/manifest.json
  periodograms/calibration.json
  periodograms/period_grid.parquet
  data/index/stars_index.parquet
  data/injected_solutions_1_companion_high_snr.parquet
  data/injected_solutions_2_companion_high_snr.parquet
  periodograms/characterization_1_companion_high_snr.parquet
  periodograms/characterization_2_companion_high_snr.parquet
  data/injected_solutions_0_companion.parquet
  data/injected_solutions_1_companion.parquet
  data/injected_solutions_2_companion.parquet
  periodograms/characterization_0_companion.parquet
  periodograms/characterization_1_companion.parquet
  periodograms/characterization_2_companion.parquet
  data/stars.csv
)
fetch() {  # one file, resumed if partial, then size-checked against the server
  local f=$1 want have
  want=$(curl -sSI -L "$BASE/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
  curl -sS -f -L -C - --retry 5 --retry-all-errors --create-dirs -o "$f" "$BASE/$f"
  have=$(stat -f %z "$f" 2>/dev/null || echo 0)
  if [ "$have" = "$want" ]; then echo "done  $f  ($have bytes)"; else echo "FAIL  $f  have=$have want=$want"; fi
}
n=0
for f in "${FILES[@]}"; do
  fetch "$f" &
  n=$((n+1)); [ $((n % 4)) -eq 0 ] && wait
done
wait
echo "ALL DOWNLOADS COMPLETE"
