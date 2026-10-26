#!/bin/bash
# Unattended harvester pcdiff v1 pipeline on the 5090 box (k-ws5090-01), run in tmux:
#   segment remaining views -> build dataset -> norm stats -> train (Docker pcdiff:cu128) -> eval
# Each stage is skipped when its output already exists, so the script can be re-run after a crash.
set -euo pipefail
P=~/crop_monitoring/paulie
CODE=$P/code/fruits3d
DS=$P/data/harvester_ds_v1
RUN=$P/runs/harvester_v1
CFG=$CODE/pcdiff/config/config_harvester_v1.yaml
log() { echo "[$(date +%F' '%T)] $*"; }

# 1) wait for the extraction / first segmentation pass, then segment views extracted since
while tmux has-session -t paulie-harvester-prep 2>/dev/null || tmux has-session -t paulie-harvester-extract 2>/dev/null; do sleep 60; done
log "segmenting remaining views"
~/miniconda3/envs/sam3-ft/bin/python $CODE/harvester/segment_views.py $P/data/views 2>&1 | grep -v Warning

# 2) dataset
if [ ! -f $DS/index.csv ]; then
  log "building dataset"
  $P/envs/prep/bin/python $CODE/harvester/build_harvester_dataset.py $P/data/views $DS -j 28
fi
log "dataset: $(ls $DS/syn/train | wc -l) syn train, $(ls $DS/syn/val | wc -l) syn val, $(ls $DS/real/val 2>/dev/null | wc -l) real val pairs"

# 3) XYZRGB normalisation (partial-centred), from 3000 random train samples
if grep -q "norm_mean: null" $CFG; then
  log "norm stats"
  $P/envs/prep/bin/python - "$DS" "$CFG" <<'EOF'
import glob, sys, numpy as np, re
ds, cfg = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(ds + "/syn/train/*.npz"))
fs = np.random.default_rng(0).permutation(fs)[:3000]
pts = []
for f in fs:
    z = np.load(f)
    a = z["aligned_data"].astype(np.float64)
    a[:, :3] -= z["rgbd_pcd"][:, :3].astype(np.float64).mean(0)
    pts.append(a[::4])
pts = np.concatenate(pts)
rgb = pts[pts[:, 4] != -1, 3:]
mean = np.r_[pts[:, :3].mean(0), rgb.mean(0)]
std = np.r_[pts[:, :3].std(0), rgb.std(0)]
fmt = lambda v: "[" + ", ".join(f"{x:.8f}" for x in v) + "]"
s = open(cfg).read()
s = re.sub(r"norm_mean: null[^\n]*", f"norm_mean: {fmt(mean)} # harvester syn train, recentred (3000 samples)", s)
s = re.sub(r"norm_std: null", f"norm_std: {fmt(std)}", s)
open(cfg, "w").write(s)
print("mean", fmt(mean), "std", fmt(std))
EOF
fi

# 4) train
until docker image inspect pcdiff:cu128 >/dev/null 2>&1; do log "waiting for pcdiff:cu128 image"; sleep 120; done
mkdir -p $RUN
DOCKER="docker run --rm --gpus all --shm-size=16g --user $(id -u):$(id -g) -e HOME=/tmp
  -v $CODE:/workspace -v $DS/syn:/cache:ro -v $DS:/ds:ro -v $RUN:/workspace/pcdiff/experiments pcdiff:cu128"
if [ ! -f $RUN/TRAIN_DONE ]; then
  last=$(ls -t $RUN/harvester_v1/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
  resume=""
  [ -n "$last" ] && resume="-ckpt ${last/$RUN//workspace/pcdiff/experiments}"
  log "training (resume: ${resume:-none})"
  $DOCKER python3 train.py -c config/config_harvester_v1.yaml $resume
  touch $RUN/TRAIN_DONE
fi

# 5) evaluate the last checkpoint and a few earlier ones
for ck in $(ls $RUN/harvester_v1/version_*/checkpoints/*.ckpt | grep -E "epoch=(19|39|59)|last" ); do
  name=$(basename $ck .ckpt | tr '=' '_')
  out=$RUN/eval/$name
  [ -f $out/summary.json ] && continue
  log "eval $name"
  $DOCKER python3 /workspace/harvester/eval_harvester.py --diff ${ck/$RUN//workspace/pcdiff/experiments} --data /ds \
    --out ${out/$RUN//workspace/pcdiff/experiments} --n-syn 300
done
log "ALL DONE"
