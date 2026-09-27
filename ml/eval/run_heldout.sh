#!/usr/bin/env bash
cd "$(dirname "$0")/../.."; . .venv/bin/activate
python -m ml.eval.eval_heldout --part 0/2 --gens 0,1,3,5 > logs/heldout_0.log 2>&1 &
python -m ml.eval.eval_heldout --part 1/2 --gens 0,1,3,5 > logs/heldout_1.log 2>&1 &
wait; python -m ml.eval.eval_heldout --report > logs/heldout_report.log 2>&1; echo HELDOUT_DONE
