"""Precompute the per-fruit training samples of ShapeCompletionDataset into a cache.

Fusing the RGB-D frames, ICP and transferring colors onto the ground truth is
deterministic, so it only has to run once per fruit instead of every epoch:

  python3 tools/build_cache.py DATA_ROOT CACHE_DIR [--splits train val] [--workers 16]

Writes CACHE_DIR/<split>/<fruit_id>.npz; existing files are skipped. Point
`data.cache_dir` in the config at CACHE_DIR to train from it (the raw data is
then not needed anymore).
"""
import contextlib
import io
import os
import multiprocessing

import click

from pcdiff.dataloader import ShapeCompletionDataset


_datasets = {}  # one dataset per split and worker process


def build(args):
    root, cache_dir, split, idx = args
    with contextlib.redirect_stdout(io.StringIO()):
        if split not in _datasets:
            _datasets[split] = ShapeCompletionDataset(data_source=root, split=split, cache_dir=cache_dir)
        ds = _datasets[split]
        fid = list(ds.fruit_list)[idx]
        if not os.path.isfile(ds.cache_path(fid)):
            ds[idx]
    return split, fid


@click.command()
@click.argument('data_root')
@click.argument('cache_dir')
@click.option('--splits', multiple=True, default=('train', 'val', 'test'))
@click.option('--workers', type=int, default=os.cpu_count())
def main(data_root, cache_dir, splits, workers):
    jobs = []
    for split in splits:
        if not os.path.isdir(os.path.join(data_root, split)):
            continue
        n = len(os.listdir(os.path.join(data_root, split)))
        jobs += [(data_root, cache_dir, split, i) for i in range(n)]

    # spawn, not fork: forking after torch/open3d have started their thread pools deadlocks the workers
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for k, (split, fid) in enumerate(pool.imap_unordered(build, jobs), 1):
            print(f'[{k}/{len(jobs)}] {split}/{fid}', flush=True)


if __name__ == '__main__':
    main()
