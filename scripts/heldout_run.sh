#!/bin/bash
PY=/c/projects/vesuvius/.venv-win/Scripts/python.exe; SP=$(dirname "$0")
for d in "$@"; do
  [ -d $d/pooled.zarr ] || $PY -m vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input $d/sv.zarr $d/pooled.zarr --level 2 --workers 8 2>&1 | tail -1
  [ -f $d/pred_reverse.tif ] || $PY -m vesuvius.ink_detection.inference.infer $d/pooled.zarr /e/vesuvius/checkpoints/seed42_step75000.pth $d/pred.tif \
      --ensemble-checkpoint /e/vesuvius/checkpoints/seed43_step75000.pth --direction both --no-compile --batch-size 32 --num-workers 2 2>&1 | grep -E "Traceback|Error"
  $PY $SP/heldout_eval.py $d $d/eval.json
done
