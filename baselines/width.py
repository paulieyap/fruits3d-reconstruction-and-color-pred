"""What does a per-fruit width measurement buy? (e.g. two keypoints on the fruit's widest horizontal chord
plus depth from the camera)

The measured width is simulated from the ground truth: the largest horizontal (xy) diameter of the GT
points, times (1 + noise), noise ~ N(0, sigma). Methods that use it:

  width_sphere        sphere of diameter w: volume pi/6 w^3; only the centre is fitted
  ellipsoid_aa_width  axis-aligned ellipsoid with both horizontal radii = w/2; centre and height fitted
  <pred>_width        an existing completion (e.g. the DDPM) scaled about its centroid to width w

  python3 -m baselines.width --cache CACHE --split val --out OUT [--sigma 0 --sigma 0.03] [--pred ddpm=DIR]

Writes OUT/results.csv in the evaluate.py format plus columns sigma, width_gt_mm, width_meas_mm.
Assumes z is the vertical axis (true for the shape completion challenge data).
"""
import csv
import multiprocessing
import os
import time
import zlib

import click
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, cKDTree

from baselines.evaluate import FIELDS, VISIBLE_THRESH, hull_volume, poisson_volume, score, to_pcd
from baselines.geometric import (N_POINTS, Completion, _bounds_x0, _ellipsoid_points, _radial_resid, _rng,
                                 _subsample, _unit_sphere, fit_sphere, largest_cluster, FIT_POINTS)


def horizontal_width(xyz, trim=0.0):
    """Largest horizontal (xy) diameter; trim drops the farthest fraction of points from the median first."""
    xy = np.asarray(xyz)[:, :2]
    if trim > 0:
        d = np.linalg.norm(xy - np.median(xy, 0), axis=1)
        xy = xy[d <= np.quantile(d, 1 - trim)]
    h = xy[ConvexHull(xy).vertices]
    return float(np.max(np.linalg.norm(h[:, None] - h[None], axis=-1)))


def width_sphere(partial, w, n_points=N_POINTS, seed=0):
    xyz = _subsample(largest_cluster(np.asarray(partial, dtype=np.float64)[:, :3]), FIT_POINTS, seed)
    r = w / 2
    c0, _, _ = fit_sphere(xyz, seed=seed)
    res = least_squares(lambda c: np.linalg.norm(xyz - c, axis=1) - r, c0, loss='huber', f_scale=0.002)
    return Completion(res.x + r * _unit_sphere(n_points, _rng(seed)), 4 / 3 * np.pi * r ** 3, {'center': res.x})


def ellipsoid_aa_width(partial, w, n_points=N_POINTS, seed=0):
    xyz = _subsample(largest_cluster(np.asarray(partial, dtype=np.float64)[:, :3]), FIT_POINTS, seed)
    r = w / 2
    c0, _, _ = fit_sphere(xyz, seed=seed)

    def resid(p):  # centre (3) + vertical radius
        return _radial_resid(xyz - p[:3], np.array([r, r, abs(p[3]) + 1e-6]))
    lb, ub = [-np.inf] * 3 + [0.3 * r], [np.inf] * 3 + [2.0 * r]
    res = least_squares(resid, _bounds_x0(np.r_[c0, r], lb, ub), bounds=(lb, ub), loss='huber', f_scale=0.002)
    radii = np.array([r, r, res.x[3]])
    pts = _ellipsoid_points(res.x[:3], np.eye(3), radii, n_points, _rng(seed))
    return Completion(pts, 4 / 3 * np.pi * radii.prod(), {'center': res.x[:3], 'radii': radii})


def scale_to_width(points, w):
    """Scale a completion about its centroid so its (lightly trimmed) horizontal width is w."""
    c = points.mean(0)
    s = w / horizontal_width(points, trim=0.01)
    return c + (points - c) * s, s


def run(args):
    path, sigmas, preds, out = args
    fid = os.path.basename(path)[:-4]
    c = np.load(path)
    gt = c['gt_points'].astype(np.float64)
    partial = c['rgbd_pcd'][:, :3].astype(np.float64)
    vol_gt = poisson_volume(gt)
    w_gt = horizontal_width(gt)
    common = {'fruit_id': fid, 'subset': 'lab' if fid.startswith('lab') else 'greenhouse',
              'vol_gt_ml': vol_gt * 1e6, 'vol_gt_hull_ml': hull_volume(gt) * 1e6,
              'visible_frac': float((cKDTree(partial).query(gt, k=1)[0] < VISIBLE_THRESH).mean()),
              'n_partial': len(partial), 'width_gt_mm': w_gt * 1000}
    pred_pts = {name: np.asarray(o3d.io.read_point_cloud(os.path.join(d, fid + '.ply')).points) for name, d in preds.items()
                if os.path.isfile(os.path.join(d, fid + '.ply'))}
    rows = []
    for sigma in sigmas:
        # same noise draw per fruit and sigma for every method
        rng = np.random.default_rng(zlib.crc32(f'{fid}:{sigma}'.encode()))
        w = w_gt * (1 + rng.normal(0, sigma)) if sigma > 0 else w_gt
        todo = [('width_sphere', lambda: width_sphere(partial, w)), ('ellipsoid_aa_width', lambda: ellipsoid_aa_width(partial, w))]
        for name, pts in pred_pts.items():
            todo.append((f'{name}_width', lambda pts=pts: (lambda sp: Completion(sp[0], None, {'scale': sp[1]}))(scale_to_width(pts, w))))
        for name, f in todo:
            t = time.time()
            comp = f()
            vol = comp.volume if comp.volume is not None else poisson_volume(comp.points)
            vol_hull = hull_volume(comp.points)
            rows.append(dict(common, method=name, sigma=sigma, width_meas_mm=w * 1000, **score(gt, comp.points),
                             vol_pred_ml=vol * 1e6, vol_pred_hull_ml=vol_hull * 1e6,
                             vol_err_ml=(vol - vol_gt) * 1e6, vol_err_pct=(vol / vol_gt - 1) * 100,
                             vol_hull_err_pct=(vol_hull / vol_gt - 1) * 100, n_used=len(partial),
                             runtime_s=time.time() - t))
    return rows


@click.command()
@click.option('--cache', required=True)
@click.option('--split', default='val')
@click.option('--out', required=True)
@click.option('--sigma', 'sigmas', multiple=True, type=float, default=(0.0, 0.03, 0.06), help='relative width noise')
@click.option('--pred', multiple=True, help='NAME=DIR of <fid>.ply completions to rescale to the measured width')
@click.option('--workers', type=int, default=8)
def main(cache, split, out, sigmas, pred, workers):
    os.makedirs(out, exist_ok=True)
    preds = dict(p.split('=', 1) for p in pred)
    d = os.path.join(cache, split)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz') and '.tmp' not in f)
    rows = []
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run, [(p, sigmas, preds, out) for p in paths]), 1):
            rows += r
            print(f'[{k}/{len(paths)}]', flush=True)
    fields = FIELDS + ['sigma', 'width_gt_mm', 'width_meas_mm']
    rows.sort(key=lambda r: (r['method'], r['sigma'], r['fruit_id']))
    with open(os.path.join(out, 'results.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f'wrote {len(rows)} rows')


if __name__ == '__main__':
    main()
