#!/bin/bash
# v3 follow-ups, two queues on the GPU at once:
#   A) every kept v3 checkpoint at 100 solver steps (epochs 7 and 11 also with the denoising trajectory)
#   B) overfitting check: the same checkpoints on training fruits, to compare with the val numbers
# Re-runnable: finished evals are skipped.
set -uo pipefail
P=~/crop_monitoring/paulie
CODE=$P/code/fruits3d
DS=$P/data/harvester_ds_v1
R2=$P/runs/harvester_v2
R3=$P/runs/harvester_v3
log() { echo "[$(date +%F' '%T)] $*"; }
DOCKER="docker run --rm --gpus all --shm-size=8g --user $(id -u):$(id -g) -e HOME=/tmp -v $CODE:/workspace
  -v $DS/syn:/cache:ro -v $DS:/ds:ro -v $R3:/workspace/pcdiff/experiments -v $R2:/r2:ro pcdiff:cu128"
CK3=/workspace/pcdiff/experiments/harvester_v3/lightning_logs/version_0/checkpoints
CK2=/r2/harvester_v2/lightning_logs/version_0/checkpoints

ev() {  # name ckpt extra-args...
  local name=$1 ck=$2; shift 2
  [ -f $R3/eval/$name/summary.json ] && return
  log "eval $name"
  $DOCKER python3 /workspace/harvester/eval_harvester.py --diff $ck --data /ds --out /workspace/pcdiff/experiments/eval/$name "$@" > $R3/eval/$name.log 2>&1
}

queue_a() {
  ev v3_best_epoch_11_T100 $CK3/harvester_v3_best_epoch=11.ckpt --n-syn 300 -T 100 --traj-every 10
  ev v3_best_epoch_07_T100 $CK3/harvester_v3_best_epoch=07.ckpt --n-syn 300 -T 100 --traj-every 10
  ev v3_best_epoch_09_T100 $CK3/harvester_v3_best_epoch=09.ckpt --n-syn 300 -T 100
  ev v3_epoch_19_T100 $CK3/harvester_v3_epoch=19.ckpt --n-syn 300 -T 100
  ev v3_epoch_39_T100 $CK3/harvester_v3_epoch=39.ckpt --n-syn 300 -T 100
}
queue_b() {
  ev train_v3_best_epoch_07 $CK3/harvester_v3_best_epoch=07.ckpt --n-syn 300 --syn-split train --real-split train --n-real 157
  ev train_v3_best_epoch_11 $CK3/harvester_v3_best_epoch=11.ckpt --n-syn 300 --syn-split train --real-split train --n-real 157
  ev train_v3_epoch_39 $CK3/harvester_v3_epoch=39.ckpt --n-syn 300 --syn-split train --real-split train --n-real 157
  ev train_v2_epoch_04 "$CK2/harvester_v2_epoch=04.ckpt" --n-syn 300 --syn-split train --real-split train --n-real 157
  ev train_v2_epoch_59 "$CK2/harvester_v2_epoch=59.ckpt" --n-syn 300 --syn-split train --real-split train --n-real 157
}
queue_a & queue_b & wait
log "ALL DONE"
