"""Render partial observation and ground truth overlaid (3 orthographic views) for visual checks.

  python3 tools/render_overlay.py CACHE_DIR SPLIT OUT_DIR fid [fid ...] [--pred NAME=DIR ...]

Partial points are drawn in their RGB colour, ground truth in grey, and an optional completion
(<DIR>/<fid>.ply) in blue. Works from the sample cache (tools/build_cache.py).
"""
import os

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

VIEWS = [('x', 'y', 0, 1), ('x', 'z', 0, 2), ('y', 'z', 1, 2)]


def thin(a, n, rng):
    return a if len(a) <= n else a[rng.choice(len(a), n, replace=False)]


@click.command()
@click.argument('cache_dir')
@click.argument('split')
@click.argument('out_dir')
@click.argument('fids', nargs=-1, required=True)
@click.option('--pred', multiple=True, help='NAME=DIR of <fid>.ply completions to overlay')
def main(cache_dir, split, out_dir, fids, pred):
    os.makedirs(out_dir, exist_ok=True)
    preds = dict(p.split('=', 1) for p in pred)
    rng = np.random.default_rng(0)
    for fid in fids:
        c = np.load(os.path.join(cache_dir, split, fid + '.npz'))
        gt = thin(c['gt_points'], 20000, rng)
        part = thin(c['rgbd_pcd'], 20000, rng)
        comps = {k: thin(np.asarray(o3d.io.read_point_cloud(os.path.join(d, fid + '.ply')).points), 20000, rng)
                 for k, d in preds.items() if os.path.isfile(os.path.join(d, fid + '.ply'))}

        fig, axes = plt.subplots(1 + len(comps), 3, figsize=(15, 5 * (1 + len(comps))), squeeze=False)
        rows = [('partial + GT', None)] + list(comps.items())
        for r, (title, comp) in enumerate(rows):
            for ax, (a, b, i, j) in zip(axes[r], VIEWS):
                ax.scatter(gt[:, i], gt[:, j], s=0.3, c='0.6', label='GT')
                if comp is None:
                    ax.scatter(part[:, i], part[:, j], s=0.3, c=np.clip(part[:, 3:6], 0, 1), label='partial')
                else:
                    ax.scatter(comp[:, i], comp[:, j], s=0.3, c='tab:blue', alpha=0.4, label=title)
                    ax.scatter(part[:, i], part[:, j], s=0.3, c='k', alpha=0.3, label='partial')
                ax.set_xlabel(a + ' [m]')
                ax.set_ylabel(b + ' [m]')
                ax.set_aspect('equal')
                ax.set_title(f'{fid}: {title} ({a}{b})')
        axes[0][0].legend(markerscale=20, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{fid}.png'), dpi=80)
        plt.close(fig)
        print(fid, 'gt', c['gt_points'].shape, 'partial', c['rgbd_pcd'].shape)


if __name__ == '__main__':
    main()
