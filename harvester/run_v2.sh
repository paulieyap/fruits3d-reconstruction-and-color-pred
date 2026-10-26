#!/bin/bash
# v2 (v-prediction) on the same dataset, started while v1 (epsilon) is still training:
#   stop v1 after its epoch-30 checkpoint -> train v2 (60 epochs) -> evaluate v1 (centroid fix) and v2.
# Re-runnable: finished stages are skipped.
set -uo pipefail
P=~/crop_monitoring/paulie
CODE=$P/code/fruits3d
DS=$P/data/harvester_ds_v1
R1=$P/runs/harvester_v1
R2=$P/runs/harvester_v2
log() { echo "[$(date +%F' '%T)] $*"; }
mkdir -p $R2

# 1) stop v1 once epoch 30 is saved (in the background, so v2 starts now)
(
  until ls $R1/harvester_v1/lightning_logs/version_0/checkpoints/*epoch=29*.ckpt >/dev/null 2>&1; do sleep 120; done
  sleep 30
  for c in $(docker ps -q --filter ancestor=pcdiff:cu128); do
    docker inspect --format '{{join .Args " "}}' $c | grep -q config_harvester_v1 && docker stop $c && log "stopped v1 container $c"
  done
  tmux kill-session -t paulie-harvester 2>/dev/null && log "closed tmux paulie-harvester"
) &

DOCKER="docker run --rm --gpus all --shm-size=16g --user $(id -u):$(id -g) -e HOME=/tmp -v $CODE:/workspace
  -v $DS/syn:/cache:ro -v $DS:/ds:ro -v $R1:/r1 -v $R2:/workspace/pcdiff/experiments pcdiff:cu128"

# 2) train v2
if [ ! -f $R2/TRAIN_DONE ]; then
  last=$(ls -t $R2/harvester_v2/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
  resume=""
  [ -n "$last" ] && resume="-ckpt ${last/$R2//workspace/pcdiff/experiments}"
  log "training v2 (resume: ${resume:-none})"
  $DOCKER python3 train.py -c config/config_harvester_v2.yaml $resume && touch $R2/TRAIN_DONE
fi
wait

# 3) evaluate: v1 with the centroid fix above t=300 (epsilon drift), v2 as is
for ck in $(ls $R1/harvester_v1/lightning_logs/version_0/checkpoints/*.ckpt | grep -E "epoch=(09|19|29)"); do
  name=v1_$(basename $ck .ckpt | sed 's/.*epoch=/epoch_/')_x0fix300
  [ -f $R2/eval/$name/summary.json ] && continue
  log "eval $name"
  $DOCKER python3 /workspace/harvester/eval_harvester.py --diff ${ck/$R1//r1} --data /ds \
    --out /workspace/pcdiff/experiments/eval/$name --n-syn 300 --x0-fix-tc 300
done
for ck in $(ls $R2/harvester_v2/lightning_logs/version_*/checkpoints/*.ckpt | grep -E "epoch=(09|19|29|39|49|59)"); do
  name=v2_$(basename $ck .ckpt | sed 's/.*epoch=/epoch_/')
  [ -f $R2/eval/$name/summary.json ] && continue
  log "eval $name"
  $DOCKER python3 /workspace/harvester/eval_harvester.py --diff ${ck/$R2//workspace/pcdiff/experiments} --data /ds \
    --out /workspace/pcdiff/experiments/eval/$name --n-syn 300
done
$DOCKER bash -c "python3 tools/export_scalars.py experiments/harvester_v2 experiments/scalars.csv; python3 tools/export_scalars.py /r1/harvester_v1 /r1/scalars.csv"
log "ALL DONE"
