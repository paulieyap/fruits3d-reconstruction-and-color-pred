"""XYZRGB mean/std of the ground-truth samples in a cache split, for data.norm_mean / data.norm_std.

  python3 tools/norm_stats.py CACHE_DIR [--split train]

Uses every GT point of every fruit (`aligned_data`: GT XYZ + colours transferred from the partial);
RGB statistics only over points that received a colour (not -1).
"""
import glob
import os

import click
import numpy as np


@click.command()
@click.argument('cache_dir')
@click.option('--split', default='train')
def main(cache_dir, split):
    pts = np.concatenate([np.load(f)['aligned_data'].astype(np.float64)
                          for f in sorted(glob.glob(os.path.join(cache_dir, split, '*.npz'))) if '.tmp' not in f])
    xyz, rgb = pts[:, :3], pts[:, 3:]
    rgb = rgb[rgb[:, 1] != -1]
    mean = np.r_[xyz.mean(0), rgb.mean(0)]
    std = np.r_[xyz.std(0), rgb.std(0)]
    fmt = lambda a: '[' + ', '.join(f'{v:.8f}' for v in a) + ']'
    print(f'# {split}: {len(pts)} points')
    print(f'    norm_mean: {fmt(mean)}')
    print(f'    norm_std: {fmt(std)}')


if __name__ == '__main__':
    main()
