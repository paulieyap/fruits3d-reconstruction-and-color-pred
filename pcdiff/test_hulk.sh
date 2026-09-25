#block(name=[plants_diff_classfree_test], threads=10, memory=48000, subtasks=1, gpus=1, hours=500)
python3 train.py -w experiments/Plants_ClassFree/default/version_0/checkpoints/last.ckpt --test

