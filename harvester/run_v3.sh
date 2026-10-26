#!/bin/bash
# v3 = v2 + EMA weights, cosine LR, checkpoints picked by val Chamfer (v2 sampled worse the longer it trained).
#   wait for v2 training to finish (GPU) -> train v3 (40 epochs) -> evaluate the best-3 and every 10th epoch
#   -> re-run the best one (by synthetic Chamfer) with the denoising trajectory for the results page.
# Re-runnable: finished stages are skipped.
set -uo pipefail
P=~/crop_monitoring/paulie
CODE=$P/code/fruits3d
DS=$P/data/harvester_ds_v1
R2=$P/runs/harvester_v2
R3=$P/runs/harvester_v3
log() { echo "[$(date +%F' '%T)] $*"; }
mkdir -p $R3/eval

until [ -f $R2/TRAIN_DONE ]; do sleep 120; done

DOCKER="docker run --rm --gpus all --shm-size=16g --user $(id -u):$(id -g) -e HOME=/tmp -v $CODE:/workspace
  -v $DS/syn:/cache:ro -v $DS:/ds:ro -v $R3:/workspace/pcdiff/experiments pcdiff:cu128"
CK=/workspace/pcdiff/experiments/harvester_v3/lightning_logs/version_0/checkpoints

# 1) train
if [ ! -f $R3/TRAIN_DONE ]; then
  last=$(ls $R3/harvester_v3/lightning_logs/version_0/checkpoints/last.ckpt 2>/dev/null || true)
  resume=""
  [ -n "$last" ] && resume="-ckpt $CK/last.ckpt"
  log "training v3 (resume: ${resume:-none})"
  $DOCKER python3 train.py -c config/config_harvester_v3.yaml $resume && touch $R3/TRAIN_DONE
fi

# 2) evaluate (50 steps, w = 1, as v2)
for ck in $(ls $R3/harvester_v3/lightning_logs/version_0/checkpoints/*.ckpt | grep -E "best|epoch=(09|19|29|39)"); do
  name=v3_$(basename $ck .ckpt | sed 's/harvester_v3_//; s/epoch=/epoch_/')
  [ -f $R3/eval/$name/summary.json ] && continue
  log "eval $name"
  $DOCKER python3 /workspace/harvester/eval_harvester.py --diff $CK/$(basename $ck) --data /ds \
    --out /workspace/pcdiff/experiments/eval/$name --n-syn 300
done

# 3) best by synthetic Chamfer, again with the denoising trajectory (100 synthetic + all real)
best=$(python3 -c "
import json, glob, os
s = {os.path.basename(os.path.dirname(f)): json.load(open(f)) for f in glob.glob('$R3/eval/v3_*/summary.json') if '_traj' not in f}
print(min(s, key=lambda k: s[k]['syn']['ddpm_cd_mm']))")
log "best: $best"
if [ ! -f $R3/eval/${best}_traj/summary.json ]; then
  ck=$(python3 -c "import json; print(json.load(open('$R3/eval/$best/summary.json'))['ckpt'])")
  $DOCKER python3 /workspace/harvester/eval_harvester.py --diff $ck --data /ds \
    --out /workspace/pcdiff/experiments/eval/${best}_traj --n-syn 100 --traj-every 10
fi
$DOCKER bash -c "python3 tools/export_scalars.py experiments/harvester_v3 experiments/scalars.csv"
log "ALL DONE (best $best)"
