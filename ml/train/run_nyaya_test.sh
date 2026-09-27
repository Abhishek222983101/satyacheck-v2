#!/usr/bin/env bash
cd "$(dirname "$0")/../.."; . .venv/bin/activate
python -m ml.train.nyaya_data --split test --clips 200 --part 0/2 > logs/nyaya_test_0.log 2>&1 &
python -m ml.train.nyaya_data --split test --clips 200 --part 1/2 > logs/nyaya_test_1.log 2>&1 &
wait; echo NYAYA_TEST_DONE
