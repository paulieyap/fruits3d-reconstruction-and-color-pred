"""Fill page_template.html with page_data.json (+ run status) -> a self-contained HTML page.

  python3 build_page.py page_data.json status.json out.html
status.json: {"state": "running"|"done", "epoch": int, "max_epoch": int, "updated": str, "commit": str}
"""
import json, os, sys

REPRO = """# GPU box k-ws5090-01 (ubuntu@100.108.110.126), everything under ~/crop_monitoring/paulie/
data/snapshots_raw/<day>/        execution snapshots (fleet 003, copied from NAS fleet/003.hrvst.tmto.01.kwl.nld.rz.etrnl)
data/views/<snapshot>/<cam>/     color.jpg depth.npy conf.png intrinsic.json pose.txt inst.png inst.json
data/harvester_ds_v1/            syn/{train,val}/*.npz (pcdiff cache), real/{train,val}/*.npz, index.csv
runs/harvester_v1/               TensorBoard + checkpoints (harvester_v1/lightning_logs/version_*), eval/<ckpt>/
code/fruits3d/                   repo, branch harvester-pcdiff

# whole pipeline (re-runnable, skips finished stages), in tmux:
tmux new -s paulie-harvester 'bash code/fruits3d/harvester/run_all.sh 2>&1 | tee -a runs/harvester_v1/run_all.log'      # v1 (epsilon)
mkdir -p runs/harvester_v2; tmux new -s paulie-harvester-v2 'bash code/fruits3d/harvester/run_v2.sh 2>&1 | tee -a runs/harvester_v2/run_v2.log'   # v2 (v-prediction) + evals

# TensorBoard
docker run --rm -p 6006:6006 -v $PWD/runs/harvester_v1:/r pcdiff:cu128 tensorboard --logdir /r --bind_all

# single steps
envs/prep/bin/python code/fruits3d/harvester/extract_snapshots.py "data/snapshots_raw/<day>/*" data/views
~/miniconda3/envs/sam3-ft/bin/python code/fruits3d/harvester/segment_views.py data/views
envs/prep/bin/python code/fruits3d/harvester/build_harvester_dataset.py data/views data/harvester_ds_v1 -j 28
docker run ... pcdiff:cu128 python3 train.py -c config/config_harvester_v1.yaml
docker run ... pcdiff:cu128 python3 /workspace/harvester/eval_harvester.py --diff <ckpt> --data /ds --out <dir>"""


def main():
    data = json.load(open(sys.argv[1]))
    data['status'] = json.load(open(sys.argv[2]))
    data['repro'] = REPRO
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'page_template.html')).read()
    blob = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    open(sys.argv[3], 'w').write(tpl.replace('/*__DATA__*/', blob))
    print(sys.argv[3], os.path.getsize(sys.argv[3]) // 1024, 'KB')


if __name__ == '__main__':
    main()
