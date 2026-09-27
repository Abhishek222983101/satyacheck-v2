#!/usr/bin/env bash
# Build Nyaya's multi-branch samples for val (fit), cal (temperature + conformal), test (report). 2 shards per split in parallel.
cd "$(dirname "$0")/../.."; . .venv/bin/activate
for sp in val cal test; do
  python -m ml.train.nyaya_data --split $sp --clips 200 --part 0/2 > logs/nyaya_${sp}_0.log 2>&1 &
  python -m ml.train.nyaya_data --split $sp --clips 200 --part 1/2 > logs/nyaya_${sp}_1.log 2>&1 &
  wait
done
echo NYAYA_DATA_DONE
